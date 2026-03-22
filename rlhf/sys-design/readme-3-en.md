# Megatron

>Everyone fears him, but no one can leave him. ——"The Godfather"

True to its name, Megatron is as powerful as Megatron, but difficult to control. No one can escape the five-finger mountain of Megatron. In this article, we have a brief taste of the basic features of Megatron and focus on analyzing the use of Megatron in the RL framework.

The history of Megatron consists of three articles, describing the important features of the three stages of Megatron's development. However, the features of Megatron are naturally much richer than those described in these papers. Even Megatron has evolved into several versions: Megatron, Megatron-Core, M-Bridge and Nemotron.

- [Tensor parallelism](https://arxiv.org/pdf/1909.08053)
- [3D Parallel](https://arxiv.org/pdf/2104.04473)
- [activation recomputation](https://arxiv.org/pdf/2205.05198)

Let’s cover some of Megatron’s important features.

## 3D Parallel

In order to support huge-scale training on thousands of GPUs, Megatron implements the fusion of 3D parallelism:

1. Tensor Parallelism (TP)


2. Pipeline Parallelism (PP)

Divide different layers of the model into multiple stages. Each GPU or GPU group is responsible for a part of the layer, and uses pipeline scheduling to execute multiple micro-batches to improve computing utilization. Using the `--pipeline-model-parallel-size` configuration, interleaved virtual pipeline can be enabled to further improve overlap capabilities.

3. Data Parallelism (DP)

Replicate model copies on multiple workers, distribute different samples, and synchronize gradients through gradient all-reduce. Megatron's DP also supports ZeRO-like distributed optimizer. Use `--use-distributed-optimizer`, `--overlap-grad-reduce`, etc. to control whether to use ZeRO.

## Megatron-Core

Based on Megatron-LM, NVIDIA launched Megatron-Core. Like trt-llm, Megatron-Core has the strongest performance and widely criticized ease of use. In actual experience, FSDP can still handle dense models below 100B; for models above 100B and MOE, Megatron is unique. Judging from the initial experiments of NV, after sacling up to a certain extent, Megatron-Core can actually show super-linear expansion capabilities. The MFU increases from 41% of the smallest model to 47% of the largest model, which is fascinating. This is actually understandable. Although larger training scale means higher intensity of network communication, GEMM (matrix multiplication operation) can allocate greater computing requirements. By expanding the GEMM size, the arithmetic intensity and execution efficiency will be higher, and the overall hardware utilization may also increase.Megatron-Core further supports the following functions:

1. Context Parallelism (CP)

The model is further segmented in the token sequence dimension, which is suitable for long context training tasks (such as Mamba, Llama 3 128k tokens).

2.MoE,EP

Supports optimization mechanisms such as Token-Level MoE routing, expert load balancing, GroupedGEMM and Token Drop.

3. Checkpoint format conversion

Supports interconversion of multiple model formats such as legacy/core/llama_mixtral, and implements checkpoint loading and conversion across TP/PP dimensions through `torch.dist`.

4. Modular API design

Megatron-Core provides components such as `GPTModel` and `TransformerConfig` based on modular construction, which optimizes the original pain points of coupling with the model to a certain extent. ~~But it’s still very serious~~

| Function items | MCore MoE | DeepSpeed |
|------------------------------------------------|----------------|------------------|
| **Architecture (Arch)** | | |
| Token dropless MoE (dMoE) | Supported | Not supported |
| Token drop MoE | Coming soon | Partially supported (Top-1/2) |
| | | |
| **MoE Router** | | |
| Top-K | Support | Partial support (Top-1/2) |
| | | |
| **MoE Load balancing** | | |
| Z-loss | Support | Not support |
| Load balancing loss | Support | Support |
| Sinkhorn | Supported | Not supported |
| | | || **Parallelism** | | |
| EP (Expert Parallel) | Support | Support |
| TP & SP (Tensor Parallel + Sequential Parallel) | Supported | Partially supported (TP only) |
| DP (Data Parallel) | Support | Support |
| PP (Pipeline Parallelism) | Supported | Not supported |
| CP (contextual parallelism) | Coming soon | Not supported |
| Complex mixed parallel support (such as TP+EP+DP+PP) | Supported | Not supported |
| Distributed MoE optimizer | Supported | Not supported |
| | | |
| **Training Utils** | | |
| ZeRO-3 | Support | Support |
| Universal Checkpoint conversion tool (supports HF format) | Supported | Not supported |
| MoE distributed checkpoint | Supported | Not supported |
| | | |
| **Kernel Fusion** | | |
| GroupedGEMM | Supported | Not supported |
| Token (un)permutation | Supported | Not supported |
| Sinkhorn | Supported | Not supported |
| | | |
| **Training Accuracy (Training Dtype)** | | || BF16 | Support | Support |
| FP16 | Support | Support |

## 3D Parallel

### TP

Divide the linear layer parameters within a single-layer Transformer along dimensions (such as MLP weights and attention heads) and distribute them to multiple GPUs for execution, thereby avoiding single-card parameter explosion. Megatron enables TP by default through `--tensor-model-parallel-size`, and cooperates with all-gather and reduce-scatter for cross-GPU communication during execution. In TP, each GPU only retains a part of a tensor, and the aggregation operation is only triggered when some operators require a complete tensor, otherwise only the activations after this layer of tensors are aggregated.

In a TP group, in order to ensure data consistency, only one process will load model parameters from disk, and the remaining processes obtain the same data through broadcast operations to ensure parameter consistency within the group. The same is true in verl, such as [here](https://github.com/volcengine/verl/blob/fcb1e191b758cadd3f45bb9d3ee815d979f4a1ec/verl/models/mcore/loader.py#L85); load complete parameters on rank0 -> split by TP strategy -> broadcast sharding.

What’s more interesting is that we know that FSDP also has a segmentation mechanism similar to TP, but FSDP needs to aggregate the entire weights when performing forwarding calculations, while TP does not. Here, we spend a certain amount of time discussing the differences between TP, FSDP1 and FSDP2 in the form of questions and answers. Readers who are not interested can skip it by themselves.

* **Question 1:** What are the fundamental differences in the core design ideas between TP, FSDP1 and FSDP2?

TP is built for computing, while FSDP is built for storage.

TP's design philosophy is Shard for Computation. The goal of TP is to decompose a single overly large mathematical operation (such as `Y = XW`) to multiple GPUs for collaborative completion. All its designs, such as splitting weights by rows/columns, strictly follow mathematical laws to ensure parallelization of calculations, and its focus is on the single operator level. We will see later how TP cleverly maintains mathematical correctness through alternate segmentation. What needs to be emphasized here is that in order to ensure strict mathematical correctness, TP is very intrusive to the model.The philosophy of FSDP1 (Legacy FSDP, derived from FairScale) is Shard for Storage. Its goal is to solve the problem that a single GPU cannot save the entire model parameters. It distributes the storage pressure to all GPUs by flattening all parameters of a module into a continuous huge tensor (`FlatParameter`), and then slicing this huge tensor. It focuses on the model module level, but its operation method is relatively rough. However, this also makes the code implementation of FSDP simpler and less intrusive to the model.

The philosophy of FSDP2 (Native FSDP, PyTorch native) is in the same vein as FSDP1, and it is still segmented for storage. But it has refined its implementation strategy. It abandons the overall packaging method of `FlatParameter` and instead separates each parameter (Per-Parameter) in the module independently. This makes its operations more granular and more flexible and efficient. Its focus is on a single parameter level. However, FSDP1 and FSDP2 are the same. In order to obtain the complete parameters of each module, multiple AllGather communications are required, resulting in high communication overhead.

* **Question 2:** Taking a 4-layer MLP as an example, what are the differences between the three in the specific implementation of parameter segmentation?

The difference is very obvious, reflected in the structure, granularity and method of segmentation. We assume that the parameters of the four layers are `W1, W2, W3, W4` respectively:

TP will perform structured segmentation. Alternately divide the weights into mathematically meaningful ways. `W1`: Split by **column** $\rightarrow W_1 = [W_{1,0} | W_{1,1} | W_{1,2} | W_{1,3}]$, `W2`: Split by **row** $\rightarrow W_2 = \begin{bmatrix} W_{2,0} \\ W_{2,1} \\ W_{2,2} \\ W_{2,3} \end{bmatrix}$, `W3`, `W4` ... and so on.

FSDP1 (Legacy) performs coarse-grained flattening. It does the following: concatenate `W1, W2, W3, W4` into a long strip in memory, then treat it as a single `FlatParameter`, and then cut it into 4 equal parts. So, GPU 0 might hold the second half of `W1` and the first half of `W2`.

FSDP2 (Native) performs fine-grained independent partitioning. It operates on each parameter independently. The `W1` tensor is sliced ​​into 4 slices, with each GPU holding 1/4. The `W2` tensor is sliced ​​into 4 slices, with each GPU holding 1/4. The same applies to `W3` and `W4`. So, GPU 0 holds exactly slice 0 of `W1`, slice 0 of `W2`, slice 0 of `W3`...* **Question 3:** Why can TP "calculate first and then gather", while FSDP must "gather first and then calculate"? What is the difference between "poly" in FSDP1 and FSDP2?

This is determined by whether their respective segmentation methods support direct calculation.

TP's "compute first, gather later" segmentation follows mathematical rules, allowing calculations to be performed directly on the shards. For example, in column parallelism, $Y = XW = X[W_0|W_1] = [XW_0|XW_1]$. GPU 0 calculates $XW_0$, and GPU 1 calculates $XW_1$. Their results are naturally the two fragments that ultimately output $Y$. The "aggregation" here refers to aggregating part of the results through `All-Reduce` in the parallel parallel layer, but the calculation does occur on the shards.

FSDP's "aggregation first and calculation later" segmentation is purely for storage, and sharding does not have the mathematical meaning of direct calculation. For example, GPU 0 only holds 1/4 of the rows of $W_1$, and it cannot compute any meaningful results just from this data and the input $X$. Therefore, it must temporarily gather the complete $W_1$ locally through `All-Gather` communication before calculation.

Even so, the aggregation of FSDP2 is still more detailed than that of FSDP1. For example, when `Linear1` needs to be calculated, FSDP1 will trigger an `All-Gather` of the entire `FlatParameter` containing `W1, W2, W3, W4`. This results in unnecessary communication (computing `W1` but aggregating `W2,W3,W4`) and higher memory spikes. FSDP2 will only trigger `All-Gather` for the single parameter `W1`. Communication is more accurate, the amount of data is smaller, and it is easier to overlap with the computing pipeline to hide communication delays.

**Question 4:** What is the point of TP using column parallelism interchangeably with row parallelism?

By alternating row parallelism and column parallelism, TP builds an efficient, low-communication computing closed loop. Specifically, we can refer to this figure. In the FFN layer in the figure below, the A matrix is ​​column-divided to obtain $\rightarrow [A_{1} | A_{2}]$, and then each passes through GeLU to obtain $\rightarrow [Y_{1} | Y_{2}]$. Two slices, without any aggregation. Then, the B matrix is row divided to obtain $\begin{bmatrix} B_{1} \\ B_{2} \end{bmatrix}$, and the Y matrix and the B matrix are directly multiplied: $\rightarrow [Y_{1} | Y_{2}] \times \begin{bmatrix} B_{1} \\ B_{2} \end{bmatrix} = \rightarrow [Y_{1}B_{1} | Y_{2}B_{2}]$. Up to this point, no aggregation has been performed.Then, before dropout, $\rightarrow Y_{1}B_{1} | Y_{2}B_{2}$ needs to perform an `All-Reduce` to obtain the complete tensor for dropout. After dropout, the final result $Z$ is obtained. There is only one aggregation in the whole process, and a final result that is strictly consistent in the mathematical sense is obtained. The next layers can be carried out in this way.

<div style="text-align: center;">
  <img src="./pics/cross-shard.png" alt="Cross Shard Diagram" style="width:50%;">
</div>

By alternating columns -> rows -> columns -> ..., the output of the previous layer (whether fragmented or complete) is exactly the input format expected by the latter layer. This design cleverly transforms the need for inter-layer communication from multiple expensive `All-Gather` (used to restore the complete activation value) to a more efficient `All-Reduce` (used to aggregate partial results).

Conversely, if only column splitting is used, the output of each column needs to be All-Gathered once to get the complete tensor before subsequent calculations can be performed.

**Question 5:** Taken together, what are the advantages and disadvantages of TP, FSDP1 and FSDP2?

These three represent different stages and different orientations of technology, and their advantages and disadvantages are very distinct.

| Comparison Dimensions | TP | FSDP1 | FSDP2 |
| :--- | :--- | :--- | :--- |
| **Core Idea** | Segmentation for **computation** | Segmentation for **storage** | Segmentation for **storage** (refinement) |
| **Code intrusiveness** | **High**, model layer needs to be rewritten | **Medium**, wrapper, but `FlatParameter` has side effects | **Very low**, pure wrapper, transparent to model code |
| **Segmentation granularity** | Operator level (row/column) | Module level (the entire module is pressed into one piece) | Parameter level (parameter by parameter) |
| **Communication Content** | Activations | Complete `FlatParameter` | Single `Parameter` |
| **Communication efficiency** | Efficient, mainly `All-Reduce` | **Lower**, communication redundancy | **Medium**, on-demand communication, easy to overlap |
| **Video memory pressure** | **Lowest** (both weights and activation values are fragmented) | **Highest** (the aggregation overhead of `FlatParameter` is very large) | **Medium** (need to aggregate a single parameter) |
| **Combinability** | **Poor**, its own logic is complex | **Poor**, the `FlatParameter` mechanism is not easy to combine | **Extremely strong**, can be seamlessly nested with TP/PP, etc. |
| **Applicable scenarios** | Huge layers that cannot be carried by a single GPU | **(Obsolete)** General model memory optimization | **(Mainstream)** General model memory optimization, large-scale cluster expansion |The best choice among the three is not to choose one out of the three, but to combine them and use a mixture of TP and FSDP2: within the node (Intra-Node), use TP for the largest layers in the model; in the entire cluster (Inter-Node), use FSDP2 to wrap the TP-modified model with another layer to achieve efficient data parallel expansion.

Having discussed so much, let’s take a look at how Megatron implements two TP blocks:

This code splits the MLP along the columns, stores 1/N weights in each rank, and then performs calculations.

<details>
<summary>TP implementation of MLP</summary>

```python 
class ColumnParallelLinear(torch.nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        *,
        ...

    ):
        super(ColumnParallelLinear, self).__init__()
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        self.output_size_per_partition = divide(output_size, world_size)

        # 1. Parameter sharding
        # The original weight is [input_size, output_size], now each GPU only stores [input_size, output_size/N]
        self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition, # Output feature partitions to reduce single card parameters
                        self.input_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )

        
    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        # Step1: Copy input to all GPUs
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Step2: Each GPU independently calculates some results
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=weight,
            bias=bias,
            ...
            ),
        # Step3: Decide whether to aggregate the results as needed
        if gather_output:
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
```
</details>

The following paragraph is the TP segmentation of MHA: the three parameter matrices Q, K, V are segmented according to columns, and the linear layer B is segmented according to rows.

<details>
<summary>TP implementation of Attention Head</summary>

```python
class ParallelAttention(MegatronModule):
...

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None,
                rotary_pos_emb=None):

        # hidden_states: Input tensor with shape [sq, b, h]
        # sq: sequence length
        # b: batch size (batch size)
        # h: Hidden layer dimension (hidden_size)

        ...

        if self.attention_type == AttnType.self_attn:
          
            # Step 1: Column parallel calculation of QKV weight matrix
            # self.query_key_value is a ColumnParallelLinear layer.
            # Its weight matrix has been split by columns (output dimensions) when loading.
            # The shape of the original QKV weights after merging is [h, 3 * h], and the shape of the weights on each GPU after splitting is [h, (3 * h) / N], where N is the TP world size.
            # Therefore, the result of forward calculation mixed_x_layer is also split in the last dimension.
            #Shape of mixed_x_layer: [sq, b, (3 * h) / N]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # Step 2: Reshape tensor to separate attention heads
            # The goal of this step is to reshape the flat, segmented hidden dimensions (3 * h) / N into a structured shape that can distinguish each attention head.
            # This is the most critical and complex step in TP implementation.

            # self.num_query_groups_per_partition: The number of query groups allocated on each GPU.
            # In standard MHA, each Q head is an independent group, so this value = num_attention_heads / world_size.
            # In GQA, multiple Q headers will share a group of K/V headers, this value = num_query_groups_per_partition / world_size.
            
            # self.num_attention_heads_per_partition: The number of Q heads allocated on each GPU. (num_attention_heads / world_size)
            # self.hidden_size_per_attention_head: Dimensions of each attention head (h / num_attention_heads).
            
            # Let's parse the last dimension of new_tensor_shape:
            # (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
            # num_attention_heads_per_partition // num_query_groups_per_partition: The number of Q headers contained in each query group.
            # + 2: It means that each query group also comes with 1 K header and 1 V header.
 
            # The entire expression calculates the total dimension of all Q, K, and V headers within a query group.
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_query_groups_per_partition,
                (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                    * self.hidden_size_per_attention_head
                ),
            )
            
            # Reshape [sq, b, (3 * h) / world_size] to [sq, b, num_groups_per_gpu, group_dim]
            # group_dim represents the total dimension of all Q, K, V headers in a query group.
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # Step 3: Split Q, K, V from the reshaped tensor
            # At this time, we separate Q, K, and V along the last dimension (dim=3), that is, group_dim.
            # This splitting is performed independently on each GPU, operating on data that has been split by TP.
            (query_layer, key_layer, value_layer) = torch.split(
                mixed_x_layer,
                [
                    # Total dimensions of Q: (Number of Q heads per group * Dimension of a single head)
                    (
                        self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                        * self.hidden_size_per_attention_head
                    ),
                    # Total dimensions of K: (1 K head * dimensions of a single head)
                    self.hidden_size_per_attention_head,
                    # Total dimensions of V: (1 V header * dimensions of a single header)
                    self.hidden_size_per_attention_head
                ],
                dim=3)
            
            # After segmentation, the obtained query_layer, key_layer, value_layer are only part of the global Q, K, V (slices).
            # query_layer shape: [sq, b, num_groups_per_gpu, num_q_heads_in_group * head_dim]
            # key_layer shape: [sq, b, num_groups_per_gpu, head_dim]
            # value_layer shape: [sq, b, num_groups_per_gpu, head_dim]

            # Step 4: Organize the shape of Q for attention calculation
            # Recombine the last two dimensions of query_layer and merge "group" and "in-group head" into the total "attention head" dimension.
            #Finally get a shape that is convenient for subsequent attention score calculations.
            # Final shape: [sq, b, num_q_heads_on_this_gpu, head_dim]
            query_layer = query_layer.view(query_layer.size(0),
                                           query_layer.size(1), -1, self.hidden_size_per_attention_head)

            # At this time, key_layer and value_layer also need to undergo similar reshape and preparation (not fully shown in the code).
            # Next, each GPU will use its own Q, K, V slices to independently calculate the attention score of the part it is responsible for.
            # Finally, in the output linear layer of the Attention module (a RowParallelLinear layer), the results of all GPUs are aggregated through All-Reduce to obtain the final complete output.
```
</details>

### PP

Pipeline parallelism is consistent with the concept of pipeline parallelism in the "Principles of Computer Composition" that we studied in our undergraduate studies. Multiple layers of the model are divided on the rank. The data of a batch is divided into smaller micro batches, which are executed in a pipeline on different ranks. The data communication volume of PP is relatively small, and communication is only carried out between adjacent pipeline segments within the PP Group, while TP requires communication within the entire TP group. A common combination strategy is to arrange PP groups to different nodes and prioritize TP on the same node. Pipeline parallelism is a serialized computing process, and the communication type is P2P communication. The amount of data communicated by a single token is small but relatively frequent, and because of the characteristics of the pipeline, pipeline bubbles will occur.

**Ordinary pipeline parallelism**

In the normal pipeline (Forward Then Backward), the model is split into multiple stages (stages), and each GPU is responsible for a part of the layers. The data is split into several micro-batches, and each micro-batch passes through different stages in turn. For example:


<div style="text-align: center;">
  <img src="./pics/fthenb.png" alt="Forward Then Backward Diagram" style="width:50%;">
</div>

Forward propagation and backward propagation are executed serially, that is, the backward propagation is started after the forward propagation is completed. This will lead to low resource utilization between GPUs; and because forward propagation and back propagation are executed serially, the intermediate variables required for back propagation cannot be released, resulting in excessive graphics memory usage or even OOM.

**1F1B**

To solve the problem of FThenB, the 1F1B scheduling strategy is introduced. The full name of 1F1B is 1 Forward 1 Backward, which means forward propagation on one side and backward propagation on the other.

<div style="text-align: center;">
  <img src="./pics/1f1b.png" alt="1F1B Diagram" style="width:50%;">
</div>

In 1F1B, forward propagation and backward propagation are performed alternately, so the forward propagation results can be released immediately when the back propagation of a micro-batch is calculated. This alternate execution mode reduces memory usage and bubbles.

**Interleaved Pipelining**

Although ordinary pipeline parallelism improves the efficiency of parallel computing to a certain extent, there is still a problem when the model parameters continue to increase: some GPUs may become idle while waiting for other stages to complete. To further optimize this, Megatron-LM proposes interleaved pipeline parallelism, which introduces finer-grained model blocking based on the ordinary pipeline.The model is further divided into multiple small chunks (num_model_chunks). Each small chunk has its own micro-batch and small chunks are executed alternately, so that each GPU can process the calculations of different layers at the same time:

<div style="text-align: center;">
  <img src="./pics/chunked_pp.png" alt="Chunked PP Diagram" style="width:50%;">
</div>

### SP

It is still recommended to read [this article](https://zhuanlan.zhihu.com/p/4083427292)

Sequence parallelism is an important optimization technology in Megatron-LM, which is designed to further reduce the memory usage when training long sequence models, especially the memory usage of activation values. It works with TP and is typically deployed within a TP Group. The core idea of ​​SP is to learn from TP's method of dividing the model weights into multiple cards, and also divide the activation values ​​into each card.

Let’s first look at the case of only TP:

<div style="text-align: center;">
  <img src="./pics/tp.png" alt="TP Diagram" style="width:80%;">
</div>

Compared with TP, TP + SP keeps the original TP parallel module unchanged, but does SP (sequence parallel processing) for the input/output part of Attn and MLP.

<div style="text-align: center;">
  <img src="./pics/tp+sp.png" alt="TP + SP Diagram" style="width:80%;">
</div>

For the parts that already have TP, the activation values are inherently cut and saved, so what SP optimizes is actually the LayerNorm and Dropout parts in the picture. These operations are independent along the sequence dimension. For example, Layer Normalization, which normalizes the embedding of each token independently, and Dropout, which applies dropout to each element independently. For these operations, we do not need to have the complete sequence data on each GPU. Sequence parallelism takes advantage of this by splitting the sequence length of the input tensor across GPUs within a tensor parallel group.

Specifically, assume that the size of the TP group is `tp_size`. A typical shape of the input tensor to the Transformer layer might be `(sequence_length, batch_size, hidden_size)`. When performing sequence parallelism, the `sequence_length` dimension of this tensor will be split into `tp_size` parts. Each TP rank only processes a part of the sequence, i.e. `(sequence_length / tp_size, batch_size, hidden_size)`.However, attention is sequence dependent. When calculating Query, Key, Value matrix multiplication, each Query needs to interact with all Keys in the sequence. Therefore, before calculating Attention, the sequence fragments split on each GPU must be all-gathered to restore the complete sequence. After the self-attention calculation is completed, if subsequent operations can be serially parallel, the output of the attention mechanism needs to be split again.

In forward pass, this is usually a simple slice operation, with each GPU taking out its own part of the sequence. In backpropagation, the corresponding operation is usually Reduce-Scatter. The gradient is calculated on the complete sequence first, and then the gradient is distributed and accumulated to the sequence fragments corresponding to each GPU through Reduce-Scatter.

### CP

https://zhuanlan.zhihu.com/p/5502876106

megatron also supports context parallelism. CP and SP are very close: SP only splits Layernorm and Dropout in the sequence dimension, while CP splits all input inputs in the sequence dimension. It can be regarded as an enhanced version of SP, which performs local optimization on the original basis. Turning on CP will overwrite the effects of SP. Except for the Attention module, Layernorm and Dropout are handled the same as SP in CP.

During the attention calculation process, the Q of each token must be calculated together with the K and V of other tokens in the same sequence. Therefore, after starting CP, K and V of all tokens must be obtained through all-gather before calculating Attention. During reverse calculation, the gradient needs to be distributed through reduce_scatter.

In order to reduce the memory usage, each GPU only needs to save a part of the KV block in the forward direction, and obtain all the KV data through all-gather communication in the reverse direction.

CP can better solve the OOM problem of long context training. Each GPU only needs to process a part of the sequence, while reducing communication and calculation by CP times, but keeping TP unchanged, and activation will also be reduced by CP times. The performance reference of CP optimization is as shown in the figure below. It is used in Megatron by specifying `--context-parallel-size`, and finally `world_size = CP * PP * DP * TP`.

<div style="text-align: center;">
  <img src="./pics/tp+cp.png" alt="TP + CP Diagram" style="width:80%;">
</div>### EP

In terms of parallelization, Megatron-Core also supports expert parallelism.

The logic of Expert Parallelism is as shown in the figure below. Each EP rank only contains a part of experts. Tokens on EP ranks often need to be distributed to other EP ranks based on gating results, so this process requires all-to-all communication.

<div style="text-align: center;">
  <img src="./pics/ep.png" alt="ep" style="width:80%;">
</div>

Taking 4 experts, 2 EP ranks and top-k=2 as an example, there are 3 tokens on each EP rank in the figure below:

<div style="text-align: center;">
  <img src="./pics/epall2all.png" alt="epall2all" style="width:80%;">
</div>


The token distribution of EP rank 0 is as follows:
Token 1 → Expert 1 and Expert 2
Token 2 → Expert 1 and Expert 2
Token 3 → Expert 0 and Expert 3
Before all-to-all communication, local tokens need to be permute/grouped according to the gating results, and tokens sent to the same expert should be grouped. Subsequently, these tokens are sent to the corresponding expert rank through all-to-all communication.

After the calculation on local experts, the original ep rank needs to be sent, which is an inverse process of all-to-all, corresponding to the all-to-all combine in the above figure, and the communication volume is consistent with all-to-all dispatch.


### Combination

Combining multiple parallel techniques results in complex interactions. How to combine parallel techniques to maximize the training throughput of large models at a given batch size while ensuring training correctness is an eternal core.

Megatron-LM proposes PTD-P, which leverages a combination of pipeline parallelism across multi-GPU servers, tensor parallelism within multi-GPU servers, and data parallelism to train models with trillions of parameters in an optimized cluster environment with high-bandwidth links between GPUs on the same server and across servers, and is easily scalable:

![image](https://hackmd.io/_uploads/HJqFPNUBgg.png)### Summary of communication methods

- TP: Use all-reduce once for forward, and use all-reduce again for backward propagation;
- PP: The activation and gradient are transferred through Send/Recv in the front and back stages; the communication volume is small and usually deployed across nodes (Infiniband).
- DDP: synchronizes gradients through all-reduce after each layer is reversed; suitable for horizontal expansion to increase throughput.
- FSDP: Forward and reverse use all-gather to aggregate parameters, and then reduce-scatter to aggregate gradient.
- SP: Use all-gather + reduce-scatter once each in forward and reverse directions, which is consistent with the traffic volume using TP alone.
- EP: forward and reverse use all-to-all dispatch to distribute tokens to the GPU where the corresponding expert is based on the routing results, and all-to-all combine to collect the expert calculation results back to the original GPU.

【TODO CP】

When considering how to set up parallel groups, the order we adopted is tp-cp-ep-dp-pp. We believe that the higher the parallel group, the greater the communication volume, so we try to arrange it in one machine. tp-cp-ep-dp-pp is the default sequence of megatron code. Of course we can modify it according to the actual situation, but the premise is to consider the communication volume.


## Megatron source code reading

[megatron code walk through](https://space.keter.top/docs/high_performance/Megatron%E6%BA%90%E7%A0%81%E9%98%85%E8%AF%BB/pretrain_process)

[Quickly summarize this article TODO]

## Usage of Megatron in RL framework

1. slime

[slime](https://github.com/THUDM/slime/blob/main/README_zh.md) uses megatron as the only training backend currently. According to Zilin’s point of view, we hope to join FSDP in the future to support its use in academic circles. Megatron debug is such a pain.

2. AreaL

AreaL natively supports megatron backend, and members of the SGLang RL Group currently support DeepSpeed-AutoTP on AreaL.

3. verl

verl currently uses Megatron-Core.

[Check this paragraph]

## Backend selection

Because the ease of use of Megatron is insane, it is important to consider whether to choose Megatron or not. According to the opinion of a senior I admire, if you choose deepspeed or FSDP compared to Megatron, the loss of training efficiency is within 20%, you should not hesitate to choose a framework that is easier to use; the same is true for the comparison between FSDP and deepspeed.Personally, I think Megatron should be chosen under the following circumstances: the number of models is small, only a few closed-source LLMs are maintained for a long time, and time can be invested in writing LayerSpec and weight mapping; the pursuit of extreme performance requires the highest hardware utilization such as 3D parallelism (TP × PP × ZeRO-DP), Flash-Attn, FP8, etc.; large-scale models and cluster GPUs are far away; and it is maintained by professional AI Infra engineers. In contrast, the scenarios for choosing other frameworks include: there are many models or the structure is frequently modified, and the adaptation cost of rewriting Megatron is too high; the training scale is small, and the model can be directly packaged with PyTorch FSDP; the cross-node bandwidth is limited due to Lacking NVLink, and the TP efficiency is reduced.

[Should the communication between zero and TP be of the same order of magnitude? Or even zero higher?]

The communication volume of ZeRO is higher, especially for models with large parameter amounts. Each layer of Zero requires all-gather parameters + reduce-scatter gradient. The communication volume of TP is relatively fixed, and all-reduce communication of operation results is only performed at specific operation points.


## Megatron implementation in verl

As mentioned before, the model and framework in Megatron cannot be decoupled, and the model needs to be cut and adapted manually. The compatibility is terrible. We can take a closer look at the access to megatron in verl:


1. Model management: `registry.py` provides unified model registration and management;
2. Configuration conversion: `config_converter.py` handles configuration conversion of different models;
3. Model creation: `model_initializer.py` is responsible for model initialization;
4. Forward propagation: `model_forward.py` defines model reasoning logic;
5. Weight processing: `weight_converter.py` and `saver.py` handle weight conversion and saving;
6. Model loading: `loader.py` is responsible for weight loading;
7. Tool support: `util.py` provides sequence processing tools;

In order to support a model in Megatron, the steps required are roughly as follows:

1. Use mcore's `GPTModel` to build the HuggingFace model: Convert HuggingFace's config to mcore's `TransformerConfig`; for example, convert the LLaMA/Qwen configuration (number of layers, dimensions, number of attention heads, etc.) into the field format recognized by mcore. Initialize mcore's GPTModel with this `TransformerConfig`. Load the weights of HuggingFace into GPTModel, paying attention to weight format conversion and dimension reshape.
2. Convert mcore model weights into HuggingFace format for rollout; resolve the differences in weight structure/naming between mcore and HuggingFace, such as `transformer.layers.0.attn.query.weight` and `model.layers.0.q_proj.weight`. Do weight resharding to the rollout engine online. Considering the complex parallelism of mcore, it is necessary to dynamically handle the slicing method, communication strategy, loading order, etc.Let's look at the code in detail. The process of building a megatron training model starts with `ActorRolloutRefWorker` in megatron_worker.py, that is, `ActorRolloutRefWorker.build_model_optimizer`:

1. Load config: First, `build_model_optimizer` will call `MegatronWorker._init_hf_config_and_tf_config` in `verl/single_controller/base/megatron/worker.py`: first read the hf config in the pre-trained model and then call `registry.hf_to_mcore_config`, and convert the hf model according to the corresponding conversion function in the registry cofig is converted into the config (`megatron.core.transformer.TransformerConfig`, ie `tfconfig`) format required by megatron. In this part, `hfconfig` and converted `tfconfig` are obtained.
2. Load the model: `get_model` is the scheduler for the main process of building models in distributed training. It calls `model_provider_func` multiple times to construct each sub-module according to the parallel position of the current process.
It is responsible for actually returning the sub-model structure that the current rank should hold based on the `pre_process` and `post_process` positions, model type, MoE/VLM and other characteristics.
3. Load weights: After the model is initialized, call [load_mcore_dist_weights](https://github.com/volcengine/verl/blob/281ecd4cc167afe676dcbaf1612009b5b81555c1/verl/utils/model.py#L536) and use `megatron.core.dist_checkpointing` to load Megatron. distributed weight. If you use HuggingFace format weights, you can also load them through `load_megatron_gptmodel_weights`, which will further call `_load_hf_model` to load HF weights

【TODO】