#AgentLoop source code analysis

Recently, Mr. Wu Xibin from the RL sys circle designed AgentLoop on verl to decouple rollout and tool calls, realizing free and flexible mutli-turn RL. Inside each AgentLoop, the rollout engine only provides a token-in-token-out interface to the outside world, and tool calls are implemented through `ToolAgentLoop`. I personally prefer such a decoupled design. At the same time, the code structure of AgentLoop is also relatively clear. After I personally studied the entire code, I felt that the design of AgentLoop was very good, but the historical baggage of `ActorRolloutRefWorker` was still heavy.

This article briefly analyzes the source code of agent loop and gives some of its own opinions.

If we regard the entire `ActorRolloutRefWorker` as a `sgl.Engine`, there are two layers of `AsyncSGLangServer` and `AsyncLLMServerManager` wrapped in AgentLoop. `AsyncSGLangServer` is equivalent to wrapping `fastapi` on `sgl.Engine` to become a server, while `AsyncLLMServerManager` wraps a layer of router on the server for load balancing, which is equivalent to sglang's router. Both levels of design are reasonable. The main trouble is `ActorRolloutRefWorker`, which is called layer by layer. Finally, it takes a total of 7 classes to transfer to `sgl.Engine`. Recently, the verl team is also working on reconstructing this worker class, so stay tuned. Finally, for the three layers of `AgentLoopManager`, `AgentLoopWorker` and `AgentLoop`, I think `AgentLoopWorker` may not be necessary, and the other two layers are quite reasonable.

## Author

Changyi Yang(CMU), Huapeng Zhou(UW), Chenyang Zhao(LMSYS)

## Related Resources

**Script**

https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/agent_loop.md

**Related PR**

https://github.com/volcengine/verl/pull/2124

**Design Docs**

https://github.com/volcengine/verl/pull/2563

https://github.com/volcengine/verl/pull/A2598

**Commit we are looking at**

https://github.com/volcengine/verl/tree/c5b189a1af496d0bc68320cd1d5bd7a1f1e3638a

## Using AgentLoop

Install the latest version of verl-sglang:

```bash
cd ~
git clone https://github.com/volcengine/verl.git
cd verl

python -m uv pip install wheel setuptools
python3 -m uv pip install -e ".[sglang]" --prerelease=allow
python3 -m uv pip install -r ./requirements.txt --no-build-isolation
python3 -m uv pip install torch_memory_saver
```
Specifically implement your own agent loop (see analysis below), and then configure the config file:

```yaml
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.name=sglang \
```
Note that if `actor_rollout_ref.rollout.mode=async` is not used, the mutli-turn function managed by SGLangRollout itself will be enabled, which is exactly the same as AgentLoop in effect.

Finally, add a new `agent_name` field during the data set construction process. For example, we append `"agent_name": "tool_agent"` in `~/verl/examples/data_preprocess/gsm8k_multiturn_w_tool.py`:

```python
def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                # new column for agent loop
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        #...
                    }
                ]
            }
            return data

        return process_fn

```
## Call overview


`main_ppo.py -> RayPPOTrainer(fit)-> AgentLoopManager(async) -> AgentLoopWorker -> AsyncLLMServerManager -> AsyncSGLangServer -> AsyncActorRolloutRefWorker -> SGLangRollout -> AsyncEngine -> sgl.Engine`

- `TaskRunner` starts training and calls `RayPPOTrainer.fit()`.
- `RayPPOTrainer` manages the training process, calls `AgentLoopManager.generate_sequences()` to start calling down layers, and initializes `AsyncActorRolloutRefWorker` at the same time.
- `AgentLoopManager` initializes dp `AsyncSGLangServer`, and subsequently, initializes `num_rollout_workers` `AgentLoopWorker`.
- Next, each `AgentLoopWorker` initializes `train_batch_size / num_rollout_workers` `AgentLoop` instances managed by itself from the pre-registered `_agent_loop_registry` according to `agent_name`. For GRPO, `train_batch_size` needs to be multiplied by the group size. Users can register new `AgentLoop` according to their own needs. Currently, `ToolAgentLoop` completely covers the tool call management based on `_req_level_generate_sequences` in `SGLangRollout`. In other words, the previous tool state management of multi-turn RL was implemented in `SGLangRollout`, and `AgentLoop` abstracted this layer of management. `SGLangRollout` was just packaged upward into `AsyncSGLangServer` to complete token-in-token-out.
- After `AgentLoop` is initialized, it manages the various states of tool calls, and based on the return status of the policy, calls `AsyncLLMServerManager` -> `AsyncSGLangServer` -> `AsyncActorRolloutRefWorker` -> `SGLangRollout` -> `AsyncEngine` -> `sgl.Engine` to the lower layers to obtain the model output. After output is returned, the `AgentLoop` life cycle ends.
- `AgentLoopWorker` collects all `AgentLoop` return values, hands them to `AgentLoopManager`, and waits for the next call.
- `AgentLoopManager` collects all return values ​​of `AgentLoopWorker` and returns.![](../imgs/agentLoop.png)

## AgentLoopManager

The top-level manager of AgentLoop is responsible for managing the life cycle of AgentLoopWorker and LLM servers. The core method is `generate_sequences`: it is called to the lower layers to obtain the trajectories of the policy model in a given agent loop environment.

### Core API

Initialized in `RayPPOTrainer`:

```Python
if self.config.actor_rollout_ref.rollout.mode == "async":
    from verl.experimental.agent_loop import AgentLoopManager

    self.async_rollout_mode = True
    self.async_rollout_manager = AgentLoopManager(
        config=self.config,
        worker_group=self.actor_rollout_wg,
    )
```
The specific initialization is very simple:

**`__init__`**

```Python
    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        self.sleep()
```
- Pass in the worker group corresponding to ActorRolloutRefWOrker, which is used to find the corresponding RolloutWorker in `_initialize_llm_servers`;
- Initialize llm server and agent loop workers;

**`_initialize_llm_servers`**

- Calculate dp size: `self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size`
- Obtain the server class through `async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)`, such as `Async``SGLang``Server`, which serves as the transfer layer for communication with the lower `sgl.Engine`.
- Initialize dp size servers with ray and create server instances for each dp rank.
- Obtain and record the address of each server through `ray.get(server.get_server_address.remote())`
- Call `ray.get([server.init_engine.remote() for server in self.async_llm_servers])`; the server queries ray through the prefix and gets all its corresponding SGLang engines in the initialized ray actor. 

```Python
def _initialize_llm_servers(self):
    # Calculate dp size
    self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
    self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

    # Get worker information for node affinity scheduling
    register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
    workers_info = ray.get(register_center.get_worker_info.remote())
    assert len(workers_info) == self.worker_group.world_size

    self.async_llm_servers = [None] * self.rollout_dp_size
    self.server_addresses = [None] * self.rollout_dp_size

    # Get the corresponding server according to config, e.g., AsyncSGLangServer
    if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
        server_class = async_server_class(
            rollout_backend=self.config.actor_rollout_ref.rollout.name,
            rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
            rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
        )
    else:
        server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

    # Initialize dp rank AsyncServer with ray
    unready_dp_ranks = set(range(self.rollout_dp_size))
    while len(unready_dp_ranks) > 0:
        servers = {
            rollout_dp_rank: server_class.options(
                # Make sure AsyncServer and the corresponding worker are on the same node
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                    soft=False,
                ),
                name=f"async_llm_server_{rollout_dp_rank}",
            ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
            for rollout_dp_rank in unready_dp_ranks
        }

        # Record server address
        for rollout_dp_rank, server in servers.items():
            try:
                address = ray.get(server.get_server_address.remote())
                self.server_addresses[rollout_dp_rank] = address
                self.async_llm_servers[rollout_dp_rank] = server
                unready_dp_ranks.remove(rollout_dp_rank)
            exceptException:
                ray.kill(server)
                print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # Initialize the server. This initialization is when the server gets all the workers corresponding to its own dp from ray.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])
```

**`_init_agent_loop_workers`**

Initialize `rollout.agent.num_workers` `AgentLoopWorker` on ray:

```Python
def _init_agent_loop_workers(self):
    self.agent_loop_workers = []
    for i in range(self.config.actor_rollout_ref.rollout.agent.num_workers):
        self.agent_loop_workers.append(
            AgentLoopWorker.options(
                name=f"agent_loop_worker_{i}",
            ).remote(self.config, self.async_llm_servers)
        )
```
**`generate_sequences`**

- If `free_cache_engine` is configured, call `self.wake_up()` first
- `chunkes = prompts.chunk(len(self.agent_loop_workers))` Chunks the input batch by the number of AgentLoopWorkers.
- Each agentLoopWorker processes its own chunk, executes it in parallel through `ray.get([worker.generate_sequences.remote(chunk) for ...])` and obtains the result;
- After processing is completed, call `self.sleep()` to put the server into sleep state to release the video memory
- Calculate performance metrics for generated sequences and tool calls
- Combine the output of all `AgentLoopWorker` and return

Code link [[here](https://github.com/volcengine/verl/blob/c5b189a1af496d0bc68320cd1d5bd7a1f1e3638a/verl/experimental/agent_loop/agent_loop.py#L486)]

```Python
def generate_sequences(self, prompts: DataProto) -> DataProto:
    if self.config.actor_rollout_ref.rollout.free_cache_engine:
        self.wake_up() # Wake up all LLM servers

    chunkes = prompts.chunk(len(self.agent_loop_workers)) # Divide into chunks according to the number of workers

    outputs = ray.get(
        [worker.generate_sequences.remote(chunk) for worker, chunk in zip(self.agent_loop_workers, chunkes)]
    ) # Distribute to each AgentLoopWorker in parallel

    output = DataProto.concat(outputs) # Aggregate the output of all workers

    if self.config.actor_rollout_ref.rollout.free_cache_engine:
        self.sleep() # Put the server into sleep state and release the video memory

    # Calculate performance indicators
    metrics = [output.meta_info["metrics"] for output in outputs]
    timing = self._performance_metrics(metrics, output)

    output.meta_info = {"timing": timing}
    return output
```

## AsyncSGLangServer

Asynchronous server implementation based on SGLang, inherited from `AsyncServerBase`. Runs as a Ray remote actor and is responsible for forwarding received requests to the underlying SGLang Engine. Due to the design of SGLang, when calling `generate`, you only need to call the master worker (verl's inference tp 0).

### Core API

**`init_engine`**

Asynchronously initialize the SGLang engine:

- Find all matching actors through `ray.util.list_named_actors`;
- Parse actor names according to the naming rule `self.wg_prefix + "WorkerDict_"`;
- Allocate actors according to dp_rank and tp_size, determine master worker (tp rank 0) 

```Python
async def init_engine(self):
    if self.workers:
        # avoid init twice
        return
    all_actors = ray.util.list_named_actors(all_namespaces=True)
    matched_actors = [
        actor for actor in all_actors if actor.get("name", None).startswith(self.wg_prefix + "WorkerDict_")
    ]

    gpu_per_node = len(set([actor["name"].split(":")[1] for actor in matched_actors]))
    # total gpu num
    assert len(matched_actors) == self._dp_size * self._tp_size

    for matched_actor in matched_actors:
        fields = matched_actor["name"].split(":")
        assert len(fields) == 2, f"invalid actor name: {matched_actor['name']}"
        pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])

        current_global_rank = gpu_per_node * pg_index + local_rank
        worker_dp_rank = current_global_rank // self._tp_size
        worker_tp_rank = current_global_rank % self._tp_size

        if worker_dp_rank == self._dp_rank:
            worker = ray.get_actor(**matched_actor)
            self.workers.append(worker)

            if worker_tp_rank == 0:
                self.master_worker = worker
```
**`chat_completion`**

Handle `chat_completion` request:

```Python
async def chat_completion(self, raw_request: Request):
    request = await raw_request.json()
    output_future = self.master_worker.chat_completion.remote(request)
    [outputs] = await asyncio.gather(output_future)
    return JSONResponse(outputs)
```
- Forward the request to the master worker for processing
- Returns response in JSON format

**`generate`**

Token in token out to obtain the inference result of SGLang Engine:

```Python
async def generate(self, prompt_ids: List[int], sampling_params: Dict[str, Any], request_id: str) -> List[int]:
    return await self.master_worker.generate.remote(prompt_ids, sampling_params, request_id)
```
- Directly call the master worker's generation method
-Support custom sampling parameters

## AsyncLLMServerManager

Manage multiple OpenAI-compatible LLM servers (e.g. `AsyncSGLangServer`), providing load balancing and session stickiness capabilities. Supports the least request load balancing algorithm to ensure that multiple rounds of conversations are sent to the same server for automatic prefix caching. It can be thought of as a simple router/load balancer layer.

**Initialization**

- Configure the server handle list and randomly shuffle it
- Initialize the minimum request load balancer: `self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]`
- Create LRU cache: `self.request_id_to_server = LRUCache(maxsize=max_cache_size)` for request_id to server mapping 

```Python
def __init__(self, config: DictConfig, server_handles: List[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)
```

**`_choose_server`**

```Python
def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
    if request_id in self.request_id_to_server:
        return self.request_id_to_server[request_id] # Session stickiness
    
    server = self.weighted_serveres[0][1][1] # The least requested server
    self.weighted_serveres[0][0] += 1 # Increase request count
    heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
    self.request_id_to_server[request_id] = server
    return server
```
- **Session Stickiness**: The same `request_id` is sent to the same `server`
- **Least Request**: New requests are assigned to the currently least loaded `server`
- **Dynamic Update**: Use heap structure to maintain server load status

**`generate`**

```Python
@rollout_trace_op
async def generate(self, request_id, *, prompt_ids: List[int], sampling_params: Dict[str, Any]) -> List[int]:
    server = self._choose_server(request_id)
    output = await server.generate.remote(
        request_id=request_id,
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
    )
    return output
```

- Select `server` based on `request_id`
- Asynchronously call the server's generation interface, token-in-token-out
- Support performance tracking

## AgentLoopWorker

`AgentLoopWorker` is responsible for receiving data and sending it down to the specific `AgentLoop`. Although the name is worker,

1. From the perspective of ray, `AgentLoopWorker` is stateful and is a ray actor, not a ray worker.
2. The core function `generate` is a layer-by-layer shell that calls other classes; for example, `single_turn_agent_loop` and `tool_agent_loop` come to `generate` (of course, `generate` of these two classes are also called downwards, which will be discussed below)

**`__init__`**

```Python
@ray.remote
class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle]):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles)

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )
```

- `config` and `server_handles` passed from upstream are used as parameters to initialize `AsyncLLMServerManager`, and then this `self.server_manager` will be passed to the downstream;
- Set `model_path, local_path, tokenizer` according to `config`**`.`**`actor_rollout_ref`**`.`**`model`**`.`**`path` of `config`
- Configure `RolloutTraceConfig` for tracing trajectories

**`generate_sequences`**

```Python
async def generate_sequences(self, batch: DataProto) -> DataProto:
    """Generate sequences from agent loop.

    Args:
        batch (DataProto): Input batch.

    Returns:
        DataProto: Output batch.
        - prompts: [bsz, prompt_length], prompt token ids from dataset.
        - responses: [bsz, response_length], output token ids include response tokens
          from LLM generation and observation tokens from tool_calls.
        - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
        - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
          and response tokens.
        - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
        - position_ids: [bsz, prompt_length + response_length], incremental position ids.

        For multi-turn conversations:
        responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
        response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
    """
    config = self.config.actor_rollout_ref.rollout
    sampling_params = dict(
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=1.0,
    )

    # override sampling params for validation
    if batch.meta_info.get("validate", False):
        sampling_params["top_p"] = config.val_kwargs.top_p
        sampling_params["temperature"] = config.val_kwargs.temperature

    # by default, we assume it's a single turn agent
    if "agent_name" not in batch.non_tensor_batch:
        batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

    tasks = []
    agent_names = batch.non_tensor_batch["agent_name"]
    raw_prompts = batch.non_tensor_batch["raw_prompt"]
    if "index" in batch.non_tensor_batch:
        index = batch.non_tensor_batch["index"]
    else:
        index = np.arange(len(raw_prompts))

    trajectory_info = await get_trajectory_info(
        batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
    )

    for agent_name, messages, trajectory in zip(agent_names, raw_prompts, trajectory_info, strict=True):
        tasks.append(
            asyncio.create_task(self._run_agent_loop(agent_name, messages.tolist(), sampling_params, trajectory))
        )
    outputs = await asyncio.gather(*tasks)

    output = self._postprocess(outputs)
    return output
```

- Use the `config` passed from upstream to create `sampling_params` for downstream use; use validation parameters for validation batch.
- Use batch's `meta_info` to obtain `agent_name, raw_prompts, index`. Then use this `meta_info` to process to obtain `trajectory_info`; that is, use the index just now to calculate the number of times each prompt is rolled out in each step, and then store it in a list to obtain the trace of the entire rollout;
- Use `agent_names, raw_prompts, trajectory_info` to execute `_run_agent_loop` concurrently.
- Within the `_run_agent_loop` function, the `agent_loop` corresponding to `agent_name` must be instantiated, and the corresponding run function of `agent_loop` must be called to generate.
- In `_postprocess`, post-processing will be carried out based on the previously calculated output (encapsulated into the `AgentLoopOutput` format); padding, mask added, and finally encapsulated into a `DataProto` for return.

```Python
async def _run_agent_loop(
    self,
    agent_name: str,
    messages: list[dict[str, Any]],
    sampling_params: dict[str, Any],
    trajectory: dict[str, Any],
) -> AgentLoopOutput:
    with rollout_trace_attr(
        step=trajectory["step"],
        sample_index=trajectory["sample_index"],
        rollout_n=trajectory["rollout_n"],
        validate=trajectory["validate"],
        name="agent_loop",
    ):
        assert agent_name in _agent_loop_registry, (
            f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
        )

        agent_loop_config = _agent_loop_registry[agent_name]
        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=_DummyConfig(config=self.config),
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
        )
        output = await agent_loop.run(messages, sampling_params)
        return output

def _postprocess(self, inputs: list[AgentLoopOutput]) -> DataProto:
    # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
    # prompts: left pad
    # responses: right pad
    # input_ids: prompt + response
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
    # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

    # prompts
    self.tokenizer.padding_side = "left"
    outputs = self.tokenizer.pad(
        [{"input_ids": input.prompt_ids} for input in inputs],
        padding="max_length",
        max_length=self.config.actor_rollout_ref.rollout.prompt_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

    # responses
    self.tokenizer.padding_side = "right"
    outputs = self.tokenizer.pad(
        [{"input_ids": input.response_ids} for input in inputs],
        padding="max_length",
        max_length=self.config.actor_rollout_ref.rollout.response_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

    # response_mask
    outputs = self.tokenizer.pad(
        [{"input_ids": input.response_mask} for input in inputs],
        padding="max_length",
        max_length=self.config.actor_rollout_ref.rollout.response_length,
        return_tensors="pt",
        return_attention_mask=False,
    )
    response_mask = outputs["input_ids"]
    assert response_ids.shape == response_mask.shape, (
        f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
    )
    response_mask = response_mask * response_attention_mask

    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

    batch = TensorDict(
        {
            "prompts": prompt_ids,  # [bsz, prompt_length]
            "responses": response_ids,  # [bsz, response_length]
            "response_mask": response_mask,  # [bsz, response_length]
            "input_ids": input_ids,  # [bsz, prompt_length + response_length]
            "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
            "position_ids": position_ids,  # [bsz, prompt_length + response_length]
        },
        batch_size=len(input_ids),
    )

    num_turns = np.array([input.num_turns for input in inputs], dtype=np.int32)
    metrics = [input.metrics.model_dump() for input in inputs]
    return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns}, meta_info={"metrics": metrics})
```
## AgentLoop

Finally entering the specific agent loop, we observe two specific AgentLoops.

## SingleTurnAgentLoop

This `agent_loop` is the default single-round dialogue, processing simple questions and answers, and does not support tool calls; the most important thing is naturally the `run` function:

1. The `messages` we passed into `agent_loop` are actually the `raw_prompt` we obtained from `batch`, here we call `apply_chat_template`;
2. Call the `generate` function in `server_manager` to calculate `response_ids`;
3. Calculate `response_mask`, intercept according to `response_length`, encapsulate these results into `AgentLoopOutput`, and padding is done in `_postprocess` of the upper layer `AgentLoopManager`;


```Python
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, config, server_manager, tokenizer):
        super().__init__(config, server_manager, tokenizer)
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        )

        with simple_timer("generate_sequences", metrics):
            response_ids = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
            )
        response_mask = [1] * len(response_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=2,
            metrics=metrics,
        )
        return output
```
## ToolAgentLoop

Finally reached the core place. `ToolAgentLoop` supports multiple rounds of conversations and tool invocations. Currently `ToolAgentLoop` can completely cover the [tool call implemented based on `_async_rollout_a_request` in `SGLangRollout` Management](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme-2.md#_async_rollout_a_request). But the number of states and the transition relationship are simpler. In other words, the previous tool state management of multi-turn RL was implemented within `SGLangRollout`, and `AgentLoop` abstracted this layer of management in advance.

**`init_class`**

The following only introduces the functions of some key parameters:

1. **`tool_response_truncate_side`: **Control the truncation method when the tool response content is too long.
   * `"left"`: Truncate from the left, retain the beginning part + "...(truncated)";
   * `"right"`: Truncate from the right side, keep the end part, and add "(truncated)..." in front;
   * Other values: truncate from the middle, keep the beginning and end, and add "...(truncated)..." in the middle
2. **`tool_config_path`**: Specify the location of the configuration file containing tool definition and configuration information, used to initialize the list of available tools, such as `verl/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml

```YAML
tools:
  - class_name: "verl.tools.gsm8k_tool.Gsm8kTool"
    config: 
      type: native
    tool_schema:
      type: "function"
      function:
        name: "calc_gsm8k_reward"
        description: "A tool for calculating the reward of gsm8k. (1.0 if parsed answer is correct, 0.0 if parsed answer is incorrect or not correctly parsed)"
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
              description: "The model's answer to the GSM8K math problem, must be a digits"
          required: ["answer"]
```
**`tool_list`**, **`tool_schemas`**: Parse and create tool instances from the configuration file through the `initialize_tools_from_config(tool_config_path)` function.

**`tool_parser`**: By setting parameters like `actor_rollout_ref.rollout.multi_turn.format=hermes`, you can get the corresponding `tool_parser`; for example, `HermesToolParser` extracts the content between `<tool_call></tool_call>` and returns the corresponding `function_call` (`function_name` and `function_arguments`), and except `content` other than `tool_call` content.
 
```Python
@classmethod
def init_class(cls, config, tokenizer):
    if cls._class_initialized:
        return
    cls._class_initialized = True
    print("Performing class-level ToolAgentLoop initialization")

    # Initialize tools from config file
    cls.tokenizer = tokenizer
    cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
    cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
    cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
    cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
    cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
    tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
    tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
    cls.tools = {tool.name: tool for tool in tool_list}
    cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
    cls.tool_parser = cls.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format)
    print(f"Initialized tools: {cls.tools}")

    cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
    cls.response_length = config.actor_rollout_ref.rollout.response_length
    cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
```
**`run`**

- Same as `single_turn_agent_loop`, for prompts `apply_chat_template`;
- Initialize `user_turns, assistant_turns` and enter the multi-turn loop until exiting:
  - Send `prompt_ids` to `server_manager` and get the corresponding `response_ids`; append the `response_ids` returned in this round to `prompt_ids` to prepare as input for the next round, and `assistant_turns += 1`
  - Handle boundary conditions, such as prompts too long, no tool call, or max turns exceeded;
  - Asynchronous execution of `_call_tool`: extract Function Call from response, then `tool`**`.`**`execute(instance_id`**`,`**` tool_args)` to obtain the corresponding `tool_response`, and then truncate and return. The specific `_call_tool` will be analyzed later.
  - `tool_responses` then `apply_chat_template` gets `tool_response_ids`, which is also appended to `prompt_ids`, and then `user_turns += 1`, entering the next cycle;
- After exiting the tool agent loop, construct `AgentLoopOutput`. Note num_turns=user_turns+assistant_turns +1, because prompt is also counted as a user turn

```Python
@rollout_trace_op
async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
    metrics = {}
    request_id = uuid4().hex
    prompt_ids = await self.loop.run_in_executor(
        None,
        lambda: self.tokenizer.apply_chat_template(
            messages, tools=self.tool_schemas, add_generation_prompt=True, tokenize=True
        ),
    )
    response_mask = []

    user_turns, assistant_turns = 0, 0
    while True:
        with simple_timer("generate_sequences", metrics):
            response_ids = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
            )
        prompt_ids += response_ids
        response_mask += [1] * len(response_ids)
        assistant_turns += 1

        # reach max response length
        if len(response_mask) >= self.response_length:
            break

        # reach max assistant turns
        if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
            break

        # reach max user turns
        if self.max_user_turns and user_turns >= self.max_user_turns:
            break

        # no tool calls
        tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
        if not tool_calls:
            break

        # call tools
        tasks = []
        for tool_call in tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call))
        with simple_timer("tool_calls", metrics):
            tool_responses = await asyncio.gather(*tasks)
        if any(isinstance(item, Exception) for item in tool_responses):
            break

        # append tool_response_ids
        tool_response_ids = await self.loop.run_in_executor(
            None,
            lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        tool_response_ids = tool_response_ids[len(self.system_prompt) :]

        # NOTE: last turn should not be user turn, or the EOS token reward
        # can't be propagated to previous token in GAE.
        if len(response_mask) + len(tool_response_ids) >= self.response_length:
            break

        prompt_ids += tool_response_ids
        response_mask += [0] * len(tool_response_ids)
        user_turns += 1

    response_ids = prompt_ids[-len(response_mask) :]
    prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

    output = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids[: self.response_length],
        response_mask=response_mask[: self.response_length],
        num_turns=user_turns + assistant_turns + 1,
        metrics=metrics,
    )
    return output
```

**`call_tool`**

To call a tool based on the tool in the tool list, such as `calc_gsm8k_reward` configured in the previous config, the arguments obtained from the tool parser can be substituted into the operation to obtain the corresponding `tool_response`. If the tool is called successfully, the resources occupied by the tool will be released, and finally `tool_response` will be truncated accordingly according to `tool_response_truncate_side`.

```Python
async def _call_tool(self, tool_call: FunctionCall) -> dict[str, str]:
    """Call tool and return tool response."""
    tool, instance_id = None, None
    try:
        # TODO: append malformed tool_call to the prompt: invalid function name or arguments
        tool_name = tool_call.name
        tool_args = json.loads(tool_call.arguments)
        tool = self.tools[tool_name]

        instance_id = await tool.create()
        tool_response, _, _ = await tool.execute(instance_id, tool_args)
    except Exception as e:
        logger.exception(f"Error when executing tool: {e}")
        return e
    finally:
        if tool and instance_id:
            await tool.release(instance_id)

    if len(tool_response) > self.max_tool_response_length:
        if self.tool_response_truncate_side == "left":
            tool_response = tool_response[: self.max_tool_response_length] + "...(truncated)"
        elif self.tool_response_truncate_side == "right":
            tool_response = "(truncated)..." + tool_response[-self.max_tool_response_length :]
        else:
            length = self.max_tool_response_length // 2
            tool_response = tool_response[:length] + "...(truncated)..." + tool_response[-length:]

    return {
        "role": "tool",
        "content": tool_response,
    }
```