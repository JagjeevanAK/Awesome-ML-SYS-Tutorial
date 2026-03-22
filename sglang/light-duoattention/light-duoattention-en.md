# Light-DuoAttention: Efficient long context reasoning using CuTeDSL

## Introduction: The challenge of long context

Imagine that you are using a large language model to process a 100-page technical document, and then ask it a specific question somewhere in the document - this is the famous "Needle in a Haystack" (NIAH) test. Handling such long-context scenarios is both an opportunity and a challenge for modern LLMs.

The standard Attention mechanism has a well-known problem: its computational complexity is O(n²). When the sequence length increases from 2k tokens to 128k tokens, the amount of calculation increases quadratically, which puts great pressure on inference performance.

In order to solve this problem, MIT-Han-Lab proposed a method called DuoAttention [1]. In fact, before DuoAttention was proposed, MIT-Han-Lab had already proposed a series of methods to solve the problem of Long Context Inference reasoning.

For example, StreamingLLM [2] discovered a phenomenon of attention sink, in which a small number of initial tokens will continue to receive a considerable part of the attention score regardless of their semantics. The failure of window attention is caused by the removal of these key attention gathering words. By retaining the key-value state of attention convergence while maintaining the latest tokens, StreamingLLM can maintain stable performance on infinite-length sequences under training-free conditions while reducing the computational complexity of each token to O(1).

Another scheme is Quest [3]. Unlike StreamingLLM, Quest is a dynamic sparse attention scheme, which means that the sparse attention mask M is adaptively calculated at runtime based on some input-related functions. Likewise, Quest is a training-free key-value option. Specifically, Quest divides the KV Cache into fixed-size pages and tracks the upper and lower bounds of attention, using these upper and lower bounds to approximate the highest possible attention score in the page. Only the Top-K pages with the highest estimated attention scores are loaded at runtime to perform sparse attention operations, thereby significantly reducing memory consumption and accelerating attention operations. Because the page score is query-related, if a new query values ​​certain tokens, previously unimportant tokens can also be recalled, allowing Quest to surpass previous query-independent algorithms in terms of attention recall.

DuoAttention is a continuation of StreamingLLM. DuoAttention believes that although StreamingLLM performs well, accuracy is often lost in applications with long contexts. At the same time, it cannot be matched with optimization methods such as GQA[4].![Accuracy of different sparse attention methods in the needle-in-a-haystack test](assets/duoattention-niah.jpg)

**Accuracy of different sparse attention methods in needle-in-a-haystack test**

The above figure shows the accuracy test of other different sparse attention methods such as Full Attention, DuoAttention and StreamingLLM on Needle-in-a-Haystack. However, unlike StreamingLLM, DuoAttention requires partial training to obtain parameters.

## DuoAttention: hybrid attention mechanism

### Core Idea

The key insight of DuoAttention is that LLM can be recognized as two types of heads, called retrieval heads (Retrieval Heads) and streaming heads (Streaming Heads). Retrieval heads only account for a small part, but need to process the entire context; while streaming heads only need to focus on recent tokens and attention sinks. DuoAttention uses different attention for different heads by targeting the dichotomy of retrieval heads and streaming heads, thereby significantly accelerating the Prefill Decode process of LLM Inference and reducing memory usage.

![DuoAttention’s processing flow for Decode and Chunked Prefill](assets/duoattention.jpg)

**DuoAttention’s processing flow for Decode and Chunked Prefill**

DuoAttention has different processing for the Prefill phase and Decode phase:

**Decode**: In the Decode stage, DuoAttention allocates two KV Cache for each layer of LLM, one for the retrieval head, which stores all historical KVs; one for the streaming head, which only stores the attention convergence and the most recent tokens, and keeps the size constant. When processing a new token, Query, Key, and Value will be split along the Head dimension to calculate the complete attention of the retrieval head and the streaming attention of the streaming head respectively. Finally, they are connected according to the Head dimension to obtain the output projection.

**Prefill**: In the Prefill stage, Chunked Prefill technology is used to reduce peak memory usage by prefilling the KV Cache with a fixed-size chunks for a long prompt. When Chunked Prefill calculates Key-Value Pairs, the KV Cache of the streaming header will be pruned immediately. In the Prefill stage, the next incoming token block will only focus on a fixed number of context tokens. Assuming n is the sequence length and c represents the chunk size, the pre-filling complexity of the streaming header is optimized from O(n²) to O(nc), and the memory complexity is reduced from O(n²) to O(nc).In addition, DuoAttention needs to obtain the information of streaming headers and retrieval headers through training. DuoAttention is trained through knowledge distillation, defining the full attention model M_teacher, and defining the model M_student using DuoAttention. The hidden state of the output of the two models is measured through the L2 norm. The smaller the difference between the two models, the more the "student" is like the "teacher". DuoAttention represents this error by defining L_distill.

At the same time, in order to save resources, DuoAttention introduces a gate value to determine which heads enable full calculation and which can be simplified. By introducing the loss function L_sparse through L1 regularization, the model is "forced" to change the gate value to 0 as much as possible during training.

Finally, DuoAttention is trained through the total loss function L = L_distill + λ * L_sparse. The training objective is selected by adjusting the value of λ. When λ is small, it means that the more important goal should be accuracy. When λ is large, it means sacrificing a certain accuracy to maintain sparsity.

## Light-DuoAttention: implemented using CuTeDSL

DuoAttention itself MIT-Han-Lab officially has an implementation [5], in which the training and inference of DuoAttention are implemented by introducing the kernel in Block-Sparse-Attention [6]. However, the current implementation is implemented on the Ampere architecture, so I wrote a new warehouse called Light-DuoAttention:

https://github.com/KuangjuX/light-duoattention

Used to implement Streaming Attention and DuoAttention on SM90 architecture and verified on H800. Finally, light-duoattention was supported in SGLang, and the parameters officially trained by DuoAttention for the LLama-3-8B model were used for verification. Finally, some tests were conducted to verify the effectiveness of the algorithm.

First of all, let me explain that the current Light-DuoAttention is not an extremely optimized implementation (of course I may gradually optimize it in the future), so it may not be suitable for use in a production environment. It is more about helping me understand the entire SGLang process from the Kernel end to the model end, so it is more like an experimental project. The current Light-DuoAttention is modified based on the implementation of FlashAttention in Hopper using CuTeDSL. Next, I will introduce the entire Light-DuoAttention project and how it runs in SGLang.### Implement Streaming Attention Kernel

First, before officially implementing Kernel, we need to use Pytorch to create a reference implementation. Let's first review Streaming Attention, which contains two key components:

1. **Sink Tokens (anchored token)**:
   - Keep the K tokens at the beginning of the sequence (for example: the first 32 or 128)
   - These tokens usually contain important information such as system prompts and instructions
   - All locations can follow these sink tokens

2. **Recent Window**:
   - Keep the most recent W tokens (for example: the most recent 256 or 512)
   - Capture local context information
   - For causal inference, use sliding windows

So the most important part in PyTorch is building the mask and applying the mask to the Attention Score:```python
beyond_causal = col_idx > row_idx
outside_window = torch.logical_and(
   col_idx < row_idx - (local_size - 1), col_idx >= sink_size
)
mask = torch.logical_or(beyond_causal, outside_window)
```
However, FlashAttention3 does not directly apply the mask to the attention score like PyTorch, but calculates Q, K, and V in blocks. Therefore, when we apply the mask, we need to consider both the intra-block mask and the inter-block mask:

- **Intra-block mask**: For intra-block masking, we need to modify `mask.py` to find the absolute position of the register held by the current thread in the entire Attention Score, and apply the mask in place according to the position.

- **Inter-block mask**: For inter-block mask, we need to modify `block_info.py` to obtain blocks that can apply Attention Sink and Recent Tokens.

A simple `apply_streaming_mask` implementation is as follows:```python
# Step 3: Apply masking rules using direct, absolute positions
should_mask = True

# Rule A: Is the key within the sink?
is_in_sink = k_pos < self.sink_size

# Rule B: Is the key within the sliding window?
is_in_window = True  # Default to true if no window size is specified
if cutlass.const_expr(self.window_size_left is not None):
    is_in_window = k_pos >= q_pos - self.window_size_left
    
# Rule C: Is the connection causal?
is_causal = k_pos <= q_pos 

# An element is NOT masked if it's causal AND (it's in the sink OR it's in the window)
if is_causal:
    if is_in_sink or is_in_window:
        should_mask = False

# Rule D: Padding mask (optional, based on sequence lengths)
if cutlass.const_expr(mask_seqlen):
    # Note: self.seqlen_q is the length of the current Q chunk, not total length
    # The q_pos check is implicitly handled by the launch grid.
    # We only need to check k_pos against the total key length.
    if k_pos >= self.seqlen_k:
        should_mask = True
        
# Step 4: Apply the mask if needed
if should_mask:
    acc_S[i] = -cutlass.Float32.inf
```
### Support Chunked Prefill

In order to support Chunked Prefill, we need to introduce an additional variable named `position_ids`, which means the position number in the current Query. This is because when we calculate Chunked Prefill, we can only get the relative position of Query instead of the absolute position for each processing. In this way, when we calculate whether to support Attention Sink, we cannot judge whether the current result is within the area of ​​Attention Sink through the relative position. Therefore, we need `position_ids` to first find the absolute position of the current result, and then perform masking.

## Support DuoAttention in SGLang

Finally, we need to support DuoAttention in SGLang and run the LLama-3-8B model. Currently we use DuoAttention to replace the Prefill stage of native SGLang. The Prefill of native SGLang is implemented in `forward_extend` in `FlashAttentionBackend`. We need to define a new parameter `enable_duo_attn`. When this parameter is turned on, we execute DuoAttention. We need to perform retrieval header and streaming header slicing from Query, Key, and Value, and finally connect (Concat) the results and obtain the final result.

One thing worth noting here is that it is mentioned in the original article:

> Before deployment, we preprocess the model by reordering the output channels of the Query, Key, and Value projection weights according to the attention head assignments. This reordering groups retrieval heads and streaming heads into two distinct, consecutive clusters, allowing for efficient slicing and concatenation operations when managing the KV cache for these two types of heads within a layer, rather than relying on scattering and gathering operations.

In order to perform slicing and joining operations conveniently and efficiently, we need to rearrange Query, Key, Value and Output in advance, and put the streaming header and retrieval header into two different continuous clusters, so that Scatter and Gather operations are not needed. In SGLang, we put this stage into `ModelRunner` for processing:```python
for i, layer in enumerate(layers):
    attn = layer.self_attn
    
    # --- 1. Define Head Patterns and Indices ---
    # kv_pattern defines which KV head group is full (1) or streaming (0)
    kv_pattern = torch.tensor(full_attention_heads[i], device=device, dtype=torch.int)
    assert len(kv_pattern) == num_kv_heads, \
         f"Layer {i}: kv_pattern length mismatch"
    
    # Get the indices for full and streaming KV heads
    kv_full_indices = torch.where(kv_pattern == 1)[0]
    kv_stream_indices = torch.where(kv_pattern == 0)[0]
            
    # Expand the KV pattern to the Q heads
    q_head_pattern = torch.repeat_interleave(kv_pattern, repeats=gqa_group_size)
    q_full_indices = torch.where(q_head_pattern == 1)[0]
    q_stream_indices = torch.where(q_head_pattern == 0)[0]

    # --- 2. Reorder Q, K, V Projection Weights ---
    # For models with a combined QKV projection matrix (e.g., Llama)
    if hasattr(attn, 'qkv_proj'):
        qkv_weight = attn.qkv_proj.weight.data
        q_size = num_heads * head_dim
        k_size = num_kv_heads * head_dim
        v_size = num_kv_heads * head_dim

        # --- Reorder Q weights ---
        w_q = qkv_weight[:q_size, :]
        w_q_reshaped = w_q.view(num_heads, head_dim, hidden_size)
        w_q_reordered = torch.cat([
            w_q_reshaped[q_full_indices],
            w_q_reshaped[q_stream_indices]
        ], dim=0).view(q_size, hidden_size)

        # --- Reorder K weights ---
```
```python
        w_k = qkv_weight[q_size : q_size + k_size, :]
        w_k_reshaped = w_k.view(num_kv_heads, head_dim, hidden_size)
        w_k_reordered = torch.cat([
            w_k_reshaped[kv_full_indices],
            w_k_reshaped[kv_stream_indices]
        ], dim=0).view(k_size, hidden_size)

        # --- Reorder V weights ---
        w_v = qkv_weight[q_size + k_size :, :]
        w_v_reshaped = w_v.view(num_kv_heads, head_dim, hidden_size)
        w_v_reordered = torch.cat([
            w_v_reshaped[kv_full_indices],
            w_v_reshaped[kv_stream_indices]
        ], dim=0).view(v_size, hidden_size)
                
        # Combine back into a single QKV weight
        new_qkv_weight = torch.cat([w_q_reordered, w_k_reordered, w_v_reordered], dim=0)
        attn.qkv_proj.weight.data = new_qkv_weight

    # --- 3. Reorder O Projection Weights ---
    o_weight = attn.o_proj.weight.data
    # Input to o_proj is concatenation of head outputs. We need to reorder the columns.
    o_weight_reshaped = o_weight.view(hidden_size, num_heads, head_dim)
    o_weight_reordered = torch.cat([
        o_weight_reshaped[:, q_full_indices, :],
        o_weight_reshaped[:, q_stream_indices, :]
    ], dim=1).view(hidden_size, q_size)
    attn.o_proj.weight.data = o_weight_reordered
```
## Experimentation and performance evaluation

At the Kernel level, we verified the correctness of Streaming Attention in various situations, including Chunked Prefill, Paged Attention, and the correctness in GQA situations.

At the end-to-end model inference level, we applied the parameters trained by DuoAttention on LLama-3-8B and performed the NIAH test. The execution effect DEMO is shown at the top. In addition, I also tested DuoAttention through knowledge Q&A, English translation, entrepreneurial writing, etc., and achieved good results.

In addition, we also made a simple performance evaluation of Streaming Attention and FlashAttention. The evaluation results are as follows:

![Performance evaluation of Streaming Attention implemented using CuTeDSL compared to FlashAttention](assets/performance.jpg)

The current experimental configuration is a single card H800, sink_size=128, recent_token=256.

## Reference

[1] Xiao G, Tang J, Zuo J, et al. Duoattention: Efficient long-context llm inference with retrieval and streaming heads[J]. arXiv preprint arXiv:2410.10819, 2024.

[2] Xiao G, Tian Y, Chen B, et al. Efficient streaming language models with attention sinks[J]. arXiv preprint arXiv:2309.17453, 2023.

[3] Tang J, Zhao Y, Zhu K, et al. Quest: Query-aware sparsity for efficient long-context LLM inference[J]. arXiv preprint, 2024.

[4] Ainslie J, Lee-Thorp J, De Jong M, et al. Gqa: Training generalized multi-query transformer models from multi-head checkpoints[J]. arXiv preprint arXiv:2305.13245, 2023.

[5] https://github.com/mit-han-lab/duo-attention

[6] https://github.com/mit-han-lab/Block-Sparse-Attention