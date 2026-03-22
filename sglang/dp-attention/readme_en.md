#Data Parallelism attention

This article will introduce the principle and implementation of DP Attention in detail. If you already understand SGLang's tensor parallelism (TP), data parallelism (DP) mechanisms and their execution links, it will be relatively easy to understand the content of this article. If you are not familiar with the relevant background, it is recommended to first read "Sglang Source Code Study Notes (3) - Distribution and Parallelism (taking deepseek as an example) (WIP) - Attack of Bruce's article - Zhihu" to establish a foundation. https://zhuanlan.zhihu.com/p/1890082781461207006

For the official introduction of SGLang DP Attention and optimization effects, please refer to: https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models

The official description is:

> The most common parallelism strategy for inference is tensor parallelism. However, it might not be the most efficient strategy for certain models. For example, DeepSeek models use MLA and only have one KV head. If we use tensor parallelism on 8 GPUs, it will lead to duplicated KV cache and unwanted memory usage.
>
>
> To overcome this, we've implemented data parallelism (DP) for the multi-head latent attention (MLA) mechanism to improve throughput for DeepSeek models. By adopting DP for the attention component, the KV cache is significantly reduced, allowing for larger batch sizes. In our DP attention implementation, each DP worker handles different types of batches (prefill, decode, idle) independently. The attention-processed data will be all-gathered among all workers before entering the Mixture-of-Experts (MoE) layer, and after processing through the MoE, the data will be redistributed back to each worker. The figure below illustrates this idea.Optimization effect: With data parallelism attention enabled, we have achieved up to 1.9x decoding throughput improvement compared to the previous version.

Note: This optimization improves peak throughput in high-volume scenarios when the server is limited by KV cache capacity, but is not recommended for low-latency, small-volume scenarios. Ref: https://github.com/sgl-project/sglang/blob/main/docs/references/deepseek.md

**Please note: In order to facilitate understanding of the core idea of DP Attention, this article will mainly analyze its first version implementation code (see PR #1970), which requires DP_SIZE to be equal to TP_SIZE. ：https://github.com/sgl-project/sglang/pull/1970**

This article was written in May 2025. At that time, SGLang's DP Attention had developed to support more flexible configurations such as `1 < DP_SIZE ≤ TP_SIZE` and `MOE-DENSE-TP-SIZE=[1, None]` to adapt to more application scenarios. Since these new features increase the complexity of implementation, this article will not go into depth and focus on the first version design for easier understanding.

# Why DP Attention is needed

The release notes for SGLang v0.4 mention:

> The most common parallelism strategy for inference is tensor parallelism. However, it might not be the most efficient strategy for certain models. For example, DeepSeek models use MLA and only have one KV head. If we use tensor parallelism on 8 GPUs, it will lead to duplicated KV cache and unwanted memory usage.
>

As stated in the release notes, the num_kv_heads (number of KV heads) of the multi-head latent attention (MLA) mechanism in the DeepSeek model is 1 (reducing the KV Cache to compress the video memory footprint and thereby optimize the inference speed).

In the QKVParallelLinear (linear layer for attention mechanism QKV transformation) implementation of SGLang (and other inference engines), its parallel strategy is shown in the following code snippet:```cpp
if tp_size >= self.total_num_kv_heads:
    self.num_kv_heads = 1
    self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
else:
    self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
    self.num_kv_head_replicas = 1
```
KV heads are copied when the number of Tensor Parallel (TP) processes (tp_size) is greater than or equal to the total number of KV heads (total_num_kv_heads). Specifically, each original KV header group will be copied tp_size / total_num_kv_heads times, and each TP_Rank is responsible for processing one of the copied KV headers.

KV heads are split when the number of Tensor Parallel (TP) processes (tp_size) is less than the total number of KV heads (total_num_kv_heads). At this time, there is only one copy of each original KV header group, and different TP_Rank processes total_num_kv_heads / tp_size KV headers respectively.

Given that MLA's num_kv_heads is 1, if the above tensor parallel sharding strategy is adopted, when tp_size is greater than 1, KV Cache will be copied tp_size times. This copying significantly increases unnecessary video memory usage, so the traditional tensor parallel segmentation method is not suitable for models with smaller num_kv_heads (such as MLA).

Therefore, SGLang proposes a solution to optimize MLA processing efficiency by using data parallelism (DP).

It is worth noting that similar problems exist in other models. For example, the num_key_value_heads of Qwen/Qwen3-235B-A22B is 4. During large-scale deployment, if tp_size is large, it may also cause KV Cache redundancy. Therefore, SGLang has also introduced DP Attention support for the Qwen3-MOE model. For details, see PR #6121: https://github.com/sgl-project/sglang/pull/6121

Note: This optimization improves peak throughput in high-volume scenarios when the server is limited by KV cache capacity, but is not recommended for low-latency, small-volume scenarios. Ref: https://github.com/sgl-project/sglang/blob/main/docs/references/deepseek.md



# How to implement DP Attention?> To overcome this, we've implemented data parallelism (DP) for the multi-head latent attention (MLA) mechanism to improve throughput for DeepSeek models. By adopting DP for the attention component, the KV cache is significantly reduced, allowing for larger batch sizes. In our DP attention implementation, each DP worker handles different types of batches (prefill, decode, idle) independently. The attention-processed data will be all-gathered among all workers before entering the Mixture-of-Experts (MoE) layer, and after processing through the MoE, the data will be redistributed back to each worker. The figure below illustrates this idea.
>

As explained in the aforementioned SGLang blog, the core ideas of DP Attention mainly include the following points:

1. By employing data parallelism for the attention component, the KV cache is significantly reduced
2. Each DP worker unit processes different types of batches independently (e.g. prefill, decode, idle)
3. All-gathered across all work units before entering the Mixed Expert (MoE) layer
4. After being processed through the MoE layer, the data will be distributed back to each work unit again

Combined with the following figure, we can understand the implementation of this mechanism more intuitively:

![image](https://github.com/user-attachments/assets/83d2b68c-8436-4c56-8828-ee336f98241f)


Based on data parallelism (DP), its core design is as follows:

1. For parts of the model other than the MLP layer (such as Embedding, Self-Attention), the tensor parallel scale (TP_Size) inside each data parallel unit (DP_Rank) is set to 1, that is, each DP_Rank calculates these parts independently.
2. For the MLP layer, all DP_Ranks together form a large tensor parallel group (TP_Group), the size of which is equal to the scale of data parallelism (DP_Size).When performing Embedding and Self-Attention calculations, each `DP_Rank` independently processes the data shards it is responsible for.

When the calculation process proceeds to the MLP layer, all `DP_Rank` will aggregate the `hidden_states` of their respective batches through the `all_gather` operation. The complete `hidden_states` tensor after aggregation is then sent to the `TP_Group` composed of all `DP_Rank` in a tensor parallel manner to the MLP layer for calculation.

After the MLP calculation is completed, its output results will be sliced ​​according to the data boundaries of each `DP_Rank`, and the corresponding parts will be distributed back to each `DP_Rank`. This process corresponds to the slice stage illustrated in the SGLang blog.

![image](https://github.com/user-attachments/assets/1f94d8f7-30a9-4fd6-9869-1f30c7a3f066)


Corresponding code:```cpp
hidden_states, start_idx, end_idx = all_gather(
    hidden_states, forward_batch, self.tp_rank, self.tp_size, self.tp_group
)
hidden_states = self.mlp(hidden_states)
hidden_states = hidden_states[start_idx:end_idx]
```
# SGLang new version of DP Attention

DP Attention in subsequent versions of SGLang has been further enhanced and supports flexible configuration of `1 < dp-size <= tp-size`.

In addition, SGLang also supports the `moe_dense_tp_size=[1, None]` configuration option for Dense FFNs in the MoE model. In particular, when this parameter is set to `1` (that is, data parallelism is adopted for these Dense FFN layers), the common computing unit fragmentation problem under high tensor parallelism can be effectively avoided, while memory usage efficiency is optimized, communication overhead is reduced, and the scalability and performance of the overall system are improved. For a more detailed description of this configuration, see: https://lmsys.org/blog/2025-05-05-large-scale-ep/#dense-ffns



# Code explanation:

https://github.com/sgl-project/sglang/pull/1970

`python/sglang/srt/managers/data_parallel_controller.py````cpp
class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(self, server_args, port_args) -> None:
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.dp_size)
        self.recv_from_tokenizer = get_zmq_socket(
            self.context, zmq.PULL, port_args.scheduler_input_ipc_name
        )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Start data parallel workers
        base_gpu_id = 0
        self.workers = []
        scheduler_pipe_readers = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name

            if server_args.enable_dp_attention:
                # Share workers for DP and TP
```
```cpp
                send_to, reader = self.launch_tensor_parallel_process(
                    server_args,
                    tmp_port_args,
                    base_gpu_id,
                    dp_rank,
                )
                base_gpu_id += 1
                scheduler_pipe_readers.append(reader)
            else:
                send_to = self.launch_tensor_parallel_group(
                    server_args,
                    tmp_port_args,
                    base_gpu_id,
                    dp_rank,
                )
                base_gpu_id += server_args.tp_size
            self.workers.append(send_to)
```
When `server_args.enable_dp_attention` is `True`, the controller calls the `launch_tensor_parallel_process` method to launch each unit of data parallel work.```cpp
def launch_tensor_parallel_process(
    self,
    server_args: ServerArgs,
    port_args: PortArgs,
    base_gpu_id: int,
    dp_rank: int,
):
    reader, writer = mp.Pipe(duplex=False)
    gpu_id = base_gpu_id
    tp_rank = dp_rank
    proc = mp.Process(
        target=run_scheduler_process,
        args=(server_args, port_args, gpu_id, tp_rank, dp_rank, writer),
    )
    proc.start()
    send_to = get_zmq_socket(
        self.context, zmq.PUSH, port_args.scheduler_input_ipc_name
    )

    return send_to, reader
```
In this function, you can see that it starts a separate scheduler process (`run_scheduler_process`) for each `dp_rank`. The key is that the `tp_rank` parameter passed to the process is set to the current `dp_rank`, and since the process is started separately for each DP worker (rather than multiple ranks within a TP group), it can be understood that the `tp_size` from the perspective of the scheduler process defaults to 1 at this time (or, in other words, it forms a TP group of size 1 by itself until before the MLP layer).

`python/sglang/srt/managers/schedule_batch.py`

`def prepare_for_idle````cpp
def prepare_for_idle(self):
    self.forward_mode = ForwardMode.IDLE
    self.input_ids = torch.empty(0, dtype=torch.int32).to(
        self.device, non_blocking=True
    )
    self.seq_lens = torch.empty(0, dtype=torch.int32).to(
        self.device, non_blocking=True
    )
    self.extend_num_tokens = 0
```
This function is used to prepare a batch in `IDLE` mode. This happens when a `dp_rank` is not assigned to actual requests. However, since all `dp_rank` needs to participate in the `all_gather` and calculation of subsequent MLP layers, even an idle `dp_rank` needs to construct an empty batch and perform the corresponding forward propagation process (although its input token is empty).

`python/sglang/srt/managers/scheduler.py`

`def prepare_dp_attn_batch````cpp
def prepare_dp_attn_batch(self, local_batch: ScheduleBatch):
    # Check if other DP workers have running batches
    if local_batch is None:
        num_tokens = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
    else:
        num_tokens = local_batch.extend_num_tokens

    local_num_tokens = torch.tensor(
        num_tokens, dtype=torch.int64, device=self.device
    )
    global_num_tokens = torch.empty(
        self.tp_size, dtype=torch.int64, device=self.device
    )
    torch.distributed.all_gather_into_tensor(
        global_num_tokens,
        local_num_tokens,
        group=self.tp_worker.get_tp_device_group(),
    )

    if local_batch is None and global_num_tokens.max().item() > 0:
        local_batch = self.get_idle_batch()

    if local_batch is not None:
        local_batch.global_num_tokens = global_num_tokens.tolist()

    return local_batch
```
When DP Attention is enabled, this function (`prepare_dp_attn_batch`) is responsible for preparing the additional required data for the `ScheduleBatch` object.

`num_tokens`: The current number of tokens that `dp_rank` actually needs to process in this batch.

`local_num_tokens`: Convert `num_tokens` into tensor form, indicating the number of tokens that the current `dp_rank` needs to process locally.

`global_num_tokens`: A tensor used to store the number of tokens collected from all `dp_rank` through `all_gather` that need to be processed. Its shape is `torch.Size([self.tp_size])` (in this first version of the code, `dp_size` is equal to `tp_size`, so `self.tp_size` here actually refers to the DP's world_size).

Through the `torch.distributed.all_gather_into_tensor` operation, the `local_num_tokens` of each `dp_rank` will be collected into the `global_num_tokens` tensor. After this operation is completed, each `dp_rank` will have a `global_num_tokens` tensor containing the token number information of all other `dp_rank`.

`python/sglang/srt/model_executor/forward_batch_info.py`

`def init_new````cpp
if ret.global_num_tokens is not None:
    max_len = max(ret.global_num_tokens)
    ret.gathered_buffer = torch.zeros(
        (max_len * model_runner.tp_size, model_runner.model_config.hidden_size),
        dtype=model_runner.dtype,
        device=device,
    )
```
Here a buffer named `gathered_buffer` is initialized for the `ForwardBatchInfo` object. This buffer is dedicated to the `all_gather` operation before the MLP layer calculation and is used to temporarily store the `hidden_states` gathered from all `dp_rank`. Its size is preset to `max(ret.global_num_tokens) * model_runner.tp_size`, ensuring that it can accommodate the data contributed by the `dp_rank` with the largest number of tokens in all `dp_rank`, and multiplied by `tp_size` (i.e. `dp_size`) to cover all `dp_rank` data.

`python/sglang/srt/models/deepseek_v2.py`

`def __init__````cpp
if use_dp:
    # For data parallel attention
    if self.q_lora_rank is not None:
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ReplicatedLinear(
            q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
        )
    else:
        self.q_proj = ReplicatedLinear(
            self.hidden_size,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
        )
    self.kv_b_proj = ReplicatedLinear(
        self.kv_lora_rank,
        self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        bias=False,
        quant_config=quant_config,
    )
    # O projection.
    self.o_proj = ReplicatedLinear(
        self.num_heads * self.v_head_dim,
        self.hidden_size,
        bias=False,
        quant_config=quant_config,
    )
```
When DP Attention is enabled (`use_dp` is `True`), the linear layers (such as Q projection, KV projection, O projection, etc.) in the attention (Attention) module no longer use the traditional tensor parallel segmentation method. Instead, each `dp_rank` will have a complete copy of these layers (such as `q_proj`, `kv_b_proj`, `o_proj`) (implemented through `ReplicatedLinear`) and complete the calculation of these linear transformations independently.

`def all_gather````cpp
def all_gather(
    input_tensor: torch.Tensor, forward_batch: ForwardBatch, rank, world_size, group
):
    if world_size == 1:
        return input_tensor

    all_lens = forward_batch.global_num_tokens
    max_len = max(forward_batch.global_num_tokens)

    padded_tensor = torch.nn.functional.pad(
        input_tensor, (0, 0, 0, max_len - input_tensor.shape[0])
    )

    torch.distributed.all_gather_into_tensor(
        forward_batch.gathered_buffer, padded_tensor, group=group
    )

    gathered_tensors = torch.concat(
        [
            forward_batch.gathered_buffer[i * max_len : i * max_len + all_lens[i]]
            for i in range(world_size)
        ]
    )

    start_index = 0 if rank == 0 else sum(all_lens[:rank])
    end_index = start_index + all_lens[rank]

    return gathered_tensors, start_index, end_index
```
This helper function `all_gather` encapsulates the core logic of global aggregation of `hidden_states` before entering the MLP layer. It receives the `input_tensor` of the current `dp_rank` (that is, part of `hidden_states`), and uses `forward_batch.global_num_tokens` (which records the number of tokens of all `dp_rank`) and the pre-allocated `forward_batch.gathered_buffer` to complete the operation.

Finally, the function returns `gathered_tensors` that aggregates all `dp_rank` data, as well as the start (`start_index`) and end (`end_index`) index of the data corresponding to the current `rank` in this gathered tensor, so that slice distribution can be performed after subsequent MLP calculations are completed.