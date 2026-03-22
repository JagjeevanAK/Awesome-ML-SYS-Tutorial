# SGLang-veRL Server: From Engine to Server, we need a more flexible RLHF rollout interface

## Preface

This article is the result of yitianlian's participation in SGLang RL. The whole process involved the cooperation of jhinpan and fzyzcjy. Finally, zhaochenyang20 completed the review. Thanks to each participant for their contribution.

I am ashamed to say that about a month and a half ago, in order to support the [openR1 project](https://github.com/huggingface/open-r1), jin and I brought together Qiujiang, jinwei, Xiaotong and xuting to add SGLang support to the three sub-projects of HuggingFace. For distillation, we support [distilabel](https://github.com/argilla-io/distilabel), and for evaluation, we support [lighteval](https://github.com/huggingface/lighteval/blob/main/docs/source/use-sglang-as-backend.mdx). Finally, on the training engine, we support trl’s [grpo](https://github.com/huggingface/trl/pull/2981). The work of supporting open-r1 across the board is really hard, but it can be considered a complete success. The only regret is that trl was delayed for a long time. At the beginning, jin wrote a version that started [SGLang Sever](https://docs.sglang.ai/backend/send_request.html). Then I refuted that everyone uses [Engine](https://docs.sglang.ai/backend/offline_engine_api.html), why do we use Server? I did not respect his opinions and work at the time, and just arbitrarily believed that we should be consistent with the community, so I rejected the method of using Server. This actually unreasonable decision delayed his work for two weeks. In the end, we did successfully support the SGLang engine on trl, but now that HuggingFace accelerate is aware of some of their problems, they still haven't merged our PR.

Although it did delay jin for a long time, it was worth the effort. Now he is a strong collaborator (and creditor, wry smile) of mine. From this, I also learned a problem: **Even for a system that looks very complex and advanced, the design of many details may be decided by community developers in the early stages. If I think a design is weird, I should think about it seriously instead of "everyone does it, so we have to do it too." ** Regarding rollout, just because everyone is comfortable using Engine, almost all open source RLHF frameworks use Engine for rollout. In the long run, for interaction with the environment and rollout flexibility, we really need Sever as a more flexible interface.## Why is Server required to complete rollout?

In order to cooperate with the training of agentic LLM, there is a strong demand to change from single turn rollout to environment interactive multi-turn rollout based on the existing PPO/GRPO algorithm.

In this process, there is an absolutely non-negligible delay in the interaction between policy and environment, and the waiting time between turns is very long. If you keep using `Engine` for rollout (`engine.generate`), you may not even be able to set up continuous batching. Therefore, the need to switch to server to do rollout through https is immediately apparent. In fact, this is the most natural way for SGLang to work. In addition, environment interaction is often completed through https requests. For example, many coding sandboxes use the environment to start a server to expose a port, and then send requests to it to achieve interaction.

In short, in order to maintain good communication and interaction among the three sub-processes of training engine, rollout and environment, it is imperative to choose server.

## An abandoned case that gained a lot

Initially, to achieve this goal, we rewritten SGLang's `launch_server` function into `launch_server_from_verl_engine`, and tried to reuse its `TokenizerManager` and `SchedulerInfo` based on the existing `VerlEngine` initialization. The original intention of doing this is to avoid repeatedly creating communication pipelines. In SGLang, TokenizerManager and SchedulerInfo have established a complete inter-process communication (IPC) mechanism (such as ZMQ socket channel). If these components are recreated instead of reused, redundant communication pipelines may be established, consuming more system resources (memory, file descriptors, etc.) and increasing system burden.

However, this solution eventually led to a more serious problem: resource conflicts occurred when components were reused. After discussion with fzyzcjy, we found that the root cause is that TokenizerManager is accessed by two threads at the same time - one is the original `_engine` thread, and the other is the new `server_from_verl_engine` thread. The current design of SGLang does not support concurrent access to TokenizerManager, causing confusion in the communication pipeline. This conflict ultimately manifests itself as "the `update_weights_from_tensor` function gets stuck after the second call": the first call succeeds, but the second call blocks permanently in the inter-process communication link, eventually triggering an NCCL timeout error.Although the plan failed, we still show the development process of this abandoned project. Review the experience of failure and move towards a successful life (wry smile).

### `launch_server_from_verl_engine`

As mentioned above, we add the `launch_server_from_verl_engine` function as the entry point of VerlEngine. This function is similar to [`launch_server`](https://github.com/sgl-project/sglang/blob/ef9a378a209d970e0b5c48ae3eac6f2660d43faf/python/sglang/srt/entrypoints/http_server.py#L659), but allows external passing of existing `tokenizer_manager` and `scheduler_info` and start HTTP Server from inside `VerlEngine`.

<details>
<summary>The endpoint of canceling the case and starting the verl server</summary>

```python
def launch_server_from_verl_engine(
    tokenizer_manager: TokenizerManager,
    scheduler_info: Dict,
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):

    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            scheduler_info=scheduler_info,
        )
    )

    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Add prometheus middleware
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    # Send a warmup request - we will create the thread launch it
    # in the lifespan after all other warmups have fired.
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(
            server_args,
            pipe_finish_writer,
            _global_state.tokenizer_manager.image_token_id,
            launch_callback,
        ),
    )
    app.warmup_thread = warmup_thread

    try:
        # Update logging configs
        set_uvicorn_logging_configs()
        app.server_args = server_args
        # Listen for HTTP requests
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        warmup_thread.join()
```
</details>

### Modify `VerlEngine.__init__`

I started a new thread to run `launch_server_from_verl_engine` in the process with `tp_rank == 0`, so as not to block the initialization logic of the main thread. And avoid port conflicts by setting `server_args.port` to `30000 + tp_rank`.

<details>
<summary>Calling launch_server_from_verl_engine on tp rank 0</summary>

```python
class VerlEngine:
    def __init__(
        self,
        device_mesh_cpu: DeviceMesh,
        nnodes: int = 1,
        **kwargs,
    ):
        self._device_mesh_cpu = device_mesh_cpu
        self._tp_rank = device_mesh_cpu.get_local_rank()
        self._tp_size = device_mesh_cpu.size()
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        if first_rank_in_node:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = Engine(
                **kwargs, tp_size=self._tp_size, node_rank=node_rank, nnodes=nnodes
            )
        else:
            self._engine = None

        if self._tp_rank == 0:
            import copy

            new_server_args = copy.deepcopy(self._engine.server_args)
            new_server_args.port = 30000 + self._tp_rank
            print(f"launch_server_from_verl_engine {new_server_args.port}")

            def server_thread_wrapper(tokenizer_manager, scheduler_info, server_args):
                print(f"Server thread begin")
                launch_server_from_verl_engine(
                    tokenizer_manager=tokenizer_manager,
                    scheduler_info=scheduler_info,
                    server_args=server_args,
                )

            server_thread = threading.Thread(
                target=server_thread_wrapper,
                args=(
                    self._engine.tokenizer_manager,
                    self._engine.scheduler_info,
                    new_server_args,
                ),
                daemon=True,
            )
            server_thread.start()

        dist.barrier(group=self._device_mesh_cpu.get_group())
```
</details>

### Specific issues regarding the abolition of cases

As we mentioned before, Failed can successfully load the model and start the Server, and the first `update_weights_from_tensor` call can also successfully complete the parameter update. However, when calling this method for the second time, the program will get stuck and eventually report an NCCL timeout error. After debugging, we found:

- `scheduler` called `send_to_tokenizer.send_pyobj(output)` after processing the update task;
- Although the `handler_loop` of `tokenizer_manager` is still running, it cannot receive the message and the main process has been blocked;
- If you comment out the startup logic of the Server (do not call `launch_server_from_verl_engine` when `VerlEngine` is initialized), the above problem disappears completely, indicating that some components of the server have affected the original communication logic.

Based on this, we found that there were problems with the previous multi-threaded design:

1. Thread safety issues:
    - TokenizerManager may not be thread-safe, causing race conditions when accessed by multiple threads simultaneously.

2. IPC channel interference:
    - The event loop created by the Server thread (FastAPI/Uvicorn) interferes with the ZMQ communication channel of the main thread.
    - This can cause one end of the pipe to be closed unexpectedly, as reported above with the `BrokenPipeError: [Errno32] Broken Pipe` error.

3. Blocking under GIL restrictions:
    - Python's GIL (Global Interpreter Lock) does not switch threads in time when handling I/O-intensive tasks.
    - When the Server thread occupies the GIL for a long time, the TokenizerManager will not be able to respond to ZMQ messages in a timely manner.

4. Resource allocation conflicts:
    - Two threads operating network resources at the same time lead to port contention, which may be related to the broken pipe error that occurs on the server.

## Actual implementation

In view of the problems that occurred in the first design idea, we adopted a completely different architectural solution - completely separating the HTTP server and the Engine to avoid any resource sharing and concurrent access conflicts. Specifically, in the abolishment case, we reused `VerlEngine` at the `_engine` level, and the type of `_engine` is actually `Engine`. In the actual implementation plan, we completely decouple `Engine` and `HTTP Service`, making them two independent and homogeneous classes. We introduced `HttpServerEngineAdapter` as an alternative, the key points of which are:1. **Completely separate rather than share resources**: Unlike the scrapped case, we gave up the idea of ​​resource reuse and no longer tried to share the TokenizerManager in different threads of the same process. Instead, we established a completely independent server process and interacted through HTTP communication.

2. **Replace the original Engine object**: Refer to the partial implementation of HttpServerEngineAdapter below. By replacing the `_engine` attribute inside VerlEngine with HttpServerEngineAdapter, we have achieved the decoupling of training and inference services. This design allows the training process to focus on model updates, while the inference service runs independently in the HTTP server, avoiding resource competition and the complexity of state synchronization. HttpServerEngineAdapter is actually an implementation of the proxy mode. It does not contain the `_engine` attribute, but communicates directly with the independently running server through HTTP requests. If the original Engine object is retained, the training process and the inference service will access the same resources, leading to the concurrency conflict problems we encountered in the cancellation case, such as communication channel confusion and process blocking.

3. **HTTP request instead of direct call**: When `VerlEngine.update_weights_from_tensor()` is called externally, the operation will be forwarded internally to an independent server process through HTTP request, which completely avoids the problem of sharing resources between threads.

<details>
<summary>Implementation of HttpServerEngineAdapter</summary>

```python
if first_rank_in_node and "launch_server" not in kwargs:
    os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
    self._engine = Engine(
        **kwargs, tp_size=self._tp_size, node_rank=node_rank, nnodes=nnodes
    )
elif "launch_server" in kwargs and kwargs["launch_server"]:
    del kwargs["launch_server"]
    if "server_args" in kwargs:
        # Directly load server_args
        server_args = kwargs["server_args"]
    else:
        # Construct server_args from kwargs
        if "log_level" not in kwargs:
            # Do not print logs by default
            kwargs["log_level"] = "error"
        server_args = ServerArgs(**kwargs)
    if self._tp_rank == 0:
        self._engine = HttpServerEngineAdapter(server_args)
    else:
        self._engine = None
else:
    self._engine = None
```
</details>

### The core difference with the abolished case

1. **Resource sharing in the abolished case causes conflicts**: In the abolished case, `TokenizerManager` is accessed by two threads at the same time: `_engine` thread and `server_from_verl_engine` thread. Since `TokenizerManager` is not thread-safe, communication confusion occurs.

2. **Complete isolation of the new solution ensures stability**: In the new solution, `HttpServerEngineAdapter` is just a proxy object and does not contain any resources shared with the original `Engine`. It communicates with the independently running server process through HTTP requests. The server process has its own independent `TokenizerManager` and `SchedulerInfo`.

3. **Request forwarding mechanism**: On the master node (`tp_rank=0`), `VerlEngine` will collect the tensor data of all nodes, and then send HTTP requests to the server through `HttpServerEngineAdapter`. The design of this client-server architecture solves the problem of resource conflicts that arise in the waste case and ensures the stability and reliability of the system. Client-Server architecture is a distributed architecture that divides system functions into service providers (Server) and service consumers (Client). Its advantages include clear separation of responsibilities, easy expansion, resource isolation, and stronger fault isolation capabilities; its disadvantages are the introduction of additional network communication overhead, possible increase in system response delays, and the need for a fault-tolerant mechanism to handle network failures.

<details>
<summary>update_weights_from_tensor in VerlEngine</summary>

```python
# update_weights_from_tensor in VerlEngine
if self._tp_rank == 0: # Only the master node sends HTTP requests
    self._engine.update_weights_from_tensor(
        named_tensors=[(name, LocalSerializedTensor(values=gathered_serialized_tensors))],
        # Other parameters...
    )
```
</details>

### Why don’t we use `update_weights_from_distributed` to update Server parameters

In SGLang, there are two main methods for updating server parameters: `update_weights_from_distributed` and `update_weights_from_tensor`. Their core differences lie in applicable scenarios and communication methods.

`update_weights_from_distributed` relies on NCCL (NVIDIA Collective Communications Library) for efficient inter-GPU communication, which is suitable for full distributed training scenarios.
The original design of `update_weights_from_tensor` is to support weight update operations on the same node and across processes, which is especially suitable for inference scenarios such as VerlEngine where HybridEngine is deployed.
`update_weights_from_tensor` actually transfers Tensor directly between CPU or GPU through the inter-process sharing mechanism, and the receiving end directly copies it into the model parameters.
**The actual data transfer does not rely on NCCL, but is completed based on pointer transfer + explicit copy. ** Therefore, there is no "transmitting tensor data via HTTP or NCCL" process in `update_weights_from_tensor`.
In this way, during the update weights process, parameters will not be transmitted through HTTP. After we switch from veRL Engine to veRL Server, we will only increase the transmission overhead of meta data, and we do not need to worry about the additional overhead of parameter transmission.
Compared with the programming flexibility brought by the server, this meta data transmission overhead is completely acceptable to us.

Finally, we chose to use `update_weights_from_tensor` in this version, which has the following benefits:

1. **Compatible with existing logic of VerlEngine**: Complete model weight synchronization directly between different processes in the same node to avoid interference with existing reasoning logic.

2. **Data copying is explicitly handled by the system**: What we pass in `update_weights_from_tensor` is the pointer position corresponding to the tensor. Therefore, since the model in the Server is on the GPU, the transferred tensor must also be placed on the GPU (the same goes for the CPU). Therefore, there is no need for NCCL transfers, and there is no need to worry about passing GPU pointers across processes. How to transfer across devices here requires follow-up research.3. **`update_weights_from_distributed` has a design conflict with the VeRL framework**: The current implementation of `update_weights_from_distributed` relies on NCCL to communicate between different placement groups, but VeRL's resource scheduling is mainly hybrid, that is, training and Inference share placement groups. How to leverage NCCL for both sending and receiving on the same placement group remains to be studied.

To sum up, `update_weights_from_tensor` is more in line with our current deployment needs in terms of compatibility, stability and implementation flexibility.

### Serialization method

When implementing `update_weights_from_tensor`, we need to pass parameters via HTTP POST request. However, since the custom `UpdateWeightsFromTensorReqInput` class cannot be passed directly as a parameter, it needs to be serialized.

In the early implementation, we used `pickle` plus `base64` to package the entire class. Although this method is feasible, there are some problems: first, it seems unnatural and inconsistent with common serialization habits; second, using `pickle` is not necessary in this scenario, and may bring additional complexity and potential security risks.

In the latest version of the implementation, we have extended the functionality of `MultiprocessingSerializer`. When data needs to be transmitted through HTTP, we will encode the byte type data (`byte`) into a string (`str`) through `base64` so that it can be successfully transmitted through an HTTP POST request. During deserialization, if it is detected that the input data type is a string, it will first be decoded back to a byte type using `b64decode`, and then handed over to the original processing logic of `MultiprocessingSerializer` to continue processing.

With this improvement, we not only simplify the code logic reuse of `update_weights_from_tensor`, but also make the entire process more standardized, clear and easy to understand. The new implementation method avoids unnecessary dependencies (such as `pickle`), is more in line with common development habits, and improves the readability and maintainability of the code.

The pseudo code is as follows:
<details>
<summary>Extensions to MultiprocessingSerializer</summary>

```python
class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        """Serialized object

        Args:
            obj: the object that needs to be serialized
            output_str: whether to return a string representation (base64 encoding)
                        Useful when serialized data needs to be transferred via JSON
        """
        #Original serialization logic
output = original serialized output

# New: Support string output
        if output_str:
            return base64.b64encode(output).decode('utf-8')
        return output

    @staticmethod
def deserialize(data, other parameters):
        """Deserialize object

        Args:
            data: serialized data (bytes or base64-encoded string)
        """
        # New: Automatically detect and process string input
        if isinstance(data, str):
            data = base64.b64decode(data.encode('utf-8'))

#Original deserialization logic
```
</details>

### Multi-node scenario support

To better support multi-node deployments, node awareness capabilities can be added. The pseudocode is currently as follows, but host should be replaced with ip later:

<details>
<summary>Discussion on Multi-nodes</summary>

```python
class HttpServerEngineAdapter:
    def __init__(self, server_args, node_aware: bool = False):
        """Initialize the HTTP server engine adapter

        Args:
            server_args: server parameters
            node_aware: Whether to enable node awareness function
        """
        self.host = server_args.host
        self.port = server_args.port
        self.node_aware = node_aware

        # If node awareness is enabled, collect all available node information
        if node_aware:
            self.nodes_info = self._discover_nodes()

    def _discover_nodes(self):
        """Discover and log all nodes in the cluster"""
        # Node discovery logic
        pass
```
</details>

## Final effect test

Start a new virtual environment. We don’t use docker here, but we still use uv.

```bash
cd ~
python3 -m venv ~/.python/veRL-server
source ~/.python/veRL-server/bin/activate
python3 -m pip install uv

# Install sglang
git clone https://github.com/yitianlian/sglang-fork.git
cd sglang-fork
git checkout feature/http_server_engine

# IMPORTANT: Install the modified package
cdpython
pip install .
pip install torch_memory_saver

# Test veRL Server
cd ../test/srt
python test_verl_engine_server.py
```