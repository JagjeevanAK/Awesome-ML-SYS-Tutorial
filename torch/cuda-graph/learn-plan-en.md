# Exploring CUDA Graph again: Dual CUDA Graph optimization in TTS model - learning plan

## Motivation Positioning

The first article in this series ([A brief analysis of CUDA Graph based on torch-memory-savor](readme_en.md)) stays at the perspective of virtual address protection, and the understanding of CUDA Graph is still superficial. Recently, CUDA Graph support was added to the Fish Audio S2 Pro model in the SGLang-Omni framework ([PR #153](https://github.com/sgl-project/sglang-omni/pull/153)), and I discovered the breadth and depth of CUDA Graph. By simultaneously executing the slow head and fast head of S2 Pro through dual CUDA Graph, the TPS increased from 55.6 to 88. Further testing combined with torch.compile:

| Configuration | Startup time | Steady-state throughput |
|---|---|---|
| No compile (CUDA graph only) | ~33s | 88 tok/s |
| Partial compile (fast head only) | ~54s | 121 tok/s |
| Full-model compile | ~137s | 126 tok/s |

> Note: TPS measures the speed at which the TTS model LLM backbone generates speech codec tokens, excluding the vocoder stage.

### Key points of opening style

When writing, start with:
1. **Review the previous articles first** to establish the continuity of the series ("Last August, I briefly wrote about...") instead of the templated beginning of "because of work needs"
2. **The benchmark data is shown at the beginning** (TPS improvement + three configuration comparison table), and the results are used to attract readers
3. **Use a refined numbered list** (within 4 items) for the roadmap. Do not use long paragraphs to describe the structure of the article.
4. **torch.compile discussion is explicitly left for later** - the core of this article focuses on CUDA Graph itself
5. **Feel free and natural** ("Everyone" style), no need to indicate the company/organization
6. **Don't write "This article is analyzed based on commit xxx"** This type of template statement - the commit hash can appear naturally when the code is referenced.

### Core Issues

This PR exposes a series of issues that require in-depth understanding:

1. **deferred graph capture mode**: Why does `factory.py` need to initialize ModelWorker with `disable_cuda_graph=True` first, then call `setup_vq_decode()` and `setup_caches()` before `init_device_graphs()`? What are the CUDA Graph constraints behind this?
2. **persistent buffer design**: The four persistent tensors `_vq_codes`, `_vq_mask`, `_output_codes`, `_output_semantic_ids` in `sglang_model.py` are the core of CUDA Graph security, but why can pre-allocate + `copy_()` ensure the stability of pointers in the graph?3. The rise and fall of **torch.compile**: The third commit of the PR (`c962aa6`) added a line `server_args.enable_torch_compile = True`, and then the launch time expanded from 33s to 137s - Triton autotune was triggered during graph capture. But Ratish1's benchmark shows that partial compile (fast head only) can increase throughput from 88 to 120 tok/s. Where does this 36% increase come from? Why did you choose not to compile in the end?
4. **Coexistence of two KV caches**: the slow head uses SGLang's RadixAttention + paged KV cache, and the fast head uses `FishQwen3AudioDecoder`'s static KV cache (`KVCache` class, each layer is independently pre-allocated `[max_bs, num_codebooks+1, n_heads, head_dim]`). How can two caches coexist in the same CUDA Graph without conflict?
5. **The deep relationship between CUDA Graph and torch.compile**: Why does SGLang choose `max-autotune-no-cudagraphs` instead of `reduce-overhead`? Why can't inductor's CUDAGraph Trees and SGLang's `CudaGraphRunner` coexist? What specific challenges does `fullgraph=True` have for codebook loop traces?
6. **Framework-level compile project blueprint** (Issue #172): How to design the `get_compile_targets()` protocol to allow the model to declare compilable targets? What are the technical challenges and trade-offs of the three-phase plan (partial compile → global compile → mega cache)?

### Series positioning

There are already two CUDA Graph related articles:

- [A brief analysis of CUDA Graph based on torch-memory-savor](readme_en.md) (first article in the series): Basic concepts of CUDA Graph, reasons why it is commonly used for reasoning/less used in training, torch-memory-saver's `cuMemMap` virtual address protection.
- [CUDA Graph vs torch.compile: Practical analysis of S2-Pro TTS model](readme-2-en.md) (the second article in the series, this draft): Comparison of the types of overhead eliminated by the two optimization techniques.Both articles stay at the "usage layer" analysis. This study focuses on the combination of CUDA Graph and specific engineering implementation** - **First establish the five core constraints of CUDA Graph as a conceptual framework**, then use PR #153 as the main narrative line, use the conceptual framework to analyze the challenges of the S2-Pro dual-head architecture, and finally delve into the implementation code of deferred capture, persistent buffer, SGLang CudaGraphRunner and other projects. Regarding the discussion of torch.compile (four modes, CUDAGraph Trees, Issue #172 blueprint), this article only gives an overview, leaving detailed analysis to subsequent articles.

In addition, this topic is also related to the following existing articles:
- [SGLang Code Walk Through](../../sglang/code-walk-through/readme.md): The overall architecture of SGLang, understanding the forward call link of ModelRunner → Model
- [SGLang Worker Architecture](../../sglang/sglang-worker/readme.md): Location of `init_cuda_graphs` in ModelRunner initialization
- [Understanding verl source code in simple terms (initialization)](../../rlhf/verl/multi-turn/code-walk-through/readme_en.md): The initialization process of SGLang rollout engine in verl, involving CUDA Graph memory reservation

## Pre-knowledge check

Before studying this topic, it is recommended to review the following content:

- [A brief analysis of CUDA Graph based on torch-memory-savor](readme_en.md): Understand the DAG structure of CUDA Graph, the basic concepts of capture/replay, and virtual address stability issues
- [SGLang Code Walk Through](../../sglang/code-walk-through/readme.md): Understand SGLang’s Server → Scheduler → ModelRunner → Model call link, and the data flow of `ForwardBatch`
- [CUDA Graph vs torch.compile: Practical analysis of S2-Pro TTS model](readme-2-en.md): Understand the slow head / fast head architecture of S2-Pro, and compare the cost dimensions of the two optimization technologies## Learning Roadmap

> **Core principle one: Concept → Model → Code. ** You must first establish the conceptual framework of CUDA Graph, then introduce the model features and use the conceptual framework to explain its challenges, and finally enter the engineering code. Absolutely not the other way around.
>
> **Core Principle 2: Progressive derivation, not flat listing. ** The content of each step must be naturally derived from the previous step and cannot be an independent knowledge point. The derivation chain is marked with "Derivation from X" in the plan of each step. Constraint mapping should not be made into a stand-alone lookup table, but should be integrated into every engineering discussion as part of the derivation ("This is the embodiment of the 'pointer stability' constraint in Chapter 1").

### Step one: CUDA Graph core mechanism

- **Deep Level**: Understanding Replication (CUDA Graph is the dependent infrastructure)
- **Goal**: Starting from the construction process, derive the constraints, and then derive the inference and video memory sharing mechanism of "one bs, one graph" - forming a progressive logical chain
- **Method**: Progressive derivation (not a list of concepts)
- **Writing Position**: The first text section of the article (enter immediately after the opening)
- **Writing Style Points** (based on actual user rewriting preferences):
  1. **Each stage must be in-depth**: It cannot be summarized in just 1-2 sentences. Capture will talk about "what information each node saves (kernel, grid/block, parameter value is the GPU virtual address), and how to infer the edges (stream submission order + event synchronization)"; Instantiate will be expanded into 3 sub-steps (dependency analysis, parameter binding, legality verification), and use analogies to assist understanding ("recording script vs compiling into executable binary", "baking/welding" and other vivid words)
  2. **Constraints should be deduced from the construction process**: Use transitional sentences such as "The operations performed in the startup phase can be further deduced..." so that the constraints grow from the mechanism and are not independently listed in the checklist.
  3. **Constraints are followed by inferences**: From the constraints, the direct inference that "a graph can only serve one batch size" is derived, with a complete reasoning chain (bs change → all kernel parameters are invalid → SGLang's multi-bs capture strategy → eager fallback explanation)
  4. **Video memory mechanism deserves independent in-depth study**: Don’t mention pool sharing in one sentence. Starting from "CUDA Graph is a kind of cache (space for time)", we will talk about the concept of high-water mark (not the sum of all intermediate tensors), internal and external isolation mental model, and then deduce the multi-graph sharing mechanism
  5. **Allow quotations from real people and conversations**: For example, "A definition that a certain teacher once shared with me"
  6. **Add a summary judgment after each technical description**: such as "In general, CUDA Graph is a relatively fragile static graph operation and needs to be carefully protected."#### 1.1 Construction process

The three stages are unfolded one by one, each stage contains a whole paragraph:

1. **Capture**: After entering recording mode, all operations are recorded as DAG nodes. Explain clearly what each node saves (kernel, grid/block, parameter value = GPU virtual address), how to infer the edges, and finally get `cudaGraph_t` (pure topology description, cannot be executed directly)
2. **Instantiate**: Introduced by analogy ("recording script vs compiling into executable binary"), unfolding three sub-steps: dependency analysis and scheduling, parameter binding and solidification ("baking into executable objects", which naturally leads to "address must remain unchanged"), legality verification
3. **Replay**: One-time submission, CPU overhead drops to zero. Emphasis on "without going through the Python/PyTorch dispatcher"

#### 1.2 Constraints (derived from the construction process)

Use the transitional sentence "The operations performed in the startup phase can further deduce the constraints", and then use a table to present the five constraints. Add summary judgment after the table.

**Then follow the inference**: One graph can only serve one batch size. Reasoning based on constraints (bs change → grid size / tensor shape / memory layout changes → parameter failure) leads to SGLang’s multi-bs capture strategy and eager fallback.

#### 1.3 PyTorch packaging

API mapping table (as concise as possible).

#### 1.4 CUDA Graph memory overhead and sharing mechanism (independent in-depth)

Advance to three levels:
1. **Video memory overhead of a single graph**: Introducing the definition of "CUDA Graph is a cache" to explain that the intermediate tensor address is locked. Introducing the concept of high-water mark (the internal caching allocator of the pool can still be reused, so it is not the sum of all tensors). Use the example of a 32-layer Transformer to illustrate.
2. **Video memory issues for multiple graphs**: 12 bs × one high-water mark = 12 times - unacceptable.
3. **Sharing mechanism**: `pool=...` allows multiple graphs to share the same piece of video memory. The reason for safety is that only one graph is replaying at the same time during the decode phase. The final effect: No matter how many graphs there are, the video memory ≈ a high-water mark of the largest graph.

### Step 2: S2-Pro model architecture and motivation for dual CUDA Graph

- **Depth Level**: Modification and Extension (SGLang-Omni is a self-developed system)
- **Goal**: Let readers first understand the computing characteristics of each of the two heads, and then answer "Why dual CUDA Graph is needed" based on this
- **Method**: Model architecture → Computational feature analysis → Derivation of core issues
- **Writing Order**:
  1. **2.1 Overall architecture + Why is it called slow/fast**: First, let readers know that S2-Pro is a Dual-AR model. A decode step must first run a large transformer (slow head, called slow because it is slow), and then run a small transformer (fast head, called fast because it is fast in a single step, but run 9 times). Explain the origin of the name: slow/fast refers to the speed of a single inference - slow head, a single inference involves 36 layers of large transformers, which is slow; fast head, a single inference with only a few layers of small transformers, is fast, but requires 9 autoregressive runs to generate 9 codebook tokens.2. **2.2-2.3 In-depth analysis of the architecture and computing characteristics of the two heads**
  3. **2.4 Two KV caches coexist**
  4. **2.5 Why CUDA Graph is helpful for this model**: Combined with the concept of the first step, explain that the characteristics of S2-Pro decode (a large number of repeated kernel sequences, fixed control flow) are naturally suitable for CUDA Graph. This may be obvious, but it needs to be pointed out explicitly as a preparation for 2.6
  5. **2.6 Why dual CUDA Graphs are needed**: Core argument - a single graph only covers the slow head is not enough, the launch overhead of the fast head is the bottleneck
  6. **2.7 PR #153 Before and after architecture comparison**
- **Writing Key**: 2.6 is the core argument of this article. But 2.5 (Why CUDA Graph is helpful), although obvious, cannot be skipped - it is the bridge from "concept" to "this model", allowing readers to confirm that "CUDA Graph is suitable for this scenario", and then ask "why do you need to put both heads in"

#### 2.1 Overall architecture of S2-Pro Dual-AR: Why is it called Slow Head / Fast Head

S2-Pro is a **Dual-AR (Dual Autoregressive) TTS model**. The process of each decode step is:

1. **Slow head** (text model): A 36-layer Qwen3 transformer, which inputs the token of the previous step and outputs the logits of the next semantic token. **Called slow because single inference is slow** - 36 layers of large transformer, hidden_size=2560, 4 large GEMMs per layer.
2. **Fast head** (audio decoder): A small transformer that receives the hidden states of the slow head and autoregressively generates 9 codebook tokens. **It’s called fast because a single inference is fast**—there are few layers and small dimensions, and it only takes μs level for a single inference. But it has to be run 9 times in a row (9 codebooks).

The key to naming: slow/fast describes the delay of a single inference, not the total time. Slow head is slow but only runs once; fast head is fast but runs 9 times.

#### 2.2 Slow Head details: `S2ProSGLangTextModel` (based on Qwen3)

Analyze the text model implementation in `sglang_model.py` (commit `cd9aaf3`):- **Model specifications**: 36 layers `S2ProDecoderLayer`, `hidden_size=2560`, `intermediate_size=9728`, `num_heads=32`, `num_kv_heads=8` (GQA), `head_dim=128`
- **Core calculation path**: `embed_tokens` → 36 × (`S2ProAttention` + gate_up_proj/down_proj FFN) → `RMSNorm` → `lm_head`
- **Attention mechanism**: Use SGLang's `RadixAttention` (with paged KV cache), use `QKVParallelLinear` to do tensor parallel, `RoPE` (non-NeoX format, `is_neox_style=False`)
- **Computing characteristics**: Each layer contains 4 large GEMMs (qkv_proj, o_proj, gate_up_proj, down_proj), a single kernel takes a long time (ms level), and the launch overhead is relatively small.
- **Key numbers**: Taking bs=8 as an example, the main GEMM shapes are `mm(8×2560, 2560×6144)` and `mm(8×4096, 4096×2560)` etc. - these have been highly optimized in cuBLAS

#### 2.3 Fast Head Details: `FishQwen3AudioDecoder` (Codebook Loop)

Analyze the implementation of audio decoder:

- **Architecture**: independent small transformer (number of layers is much smaller than text model), including `project_in` (text_dim → fast_dim linear projection), multi-layer `TransformerBlock`, `RMSNorm`, `output` linear header
- **KV Cache**: **static KV cache**, each layer is independently pre-allocated `KVCache(max_batch_size, num_codebooks+1, n_local_heads, head_dim)`, completely independent from SGLang's paged KV cache
- **codebook_embeddings**: `nn.Embedding(vocab_size × num_codebooks, text_dim)` - shared codebook embedding table, used to map VQ codes back to the dimensional space of the text model
- **codebook_offsets**: `torch.arange(num_codebooks) * vocab_size` - offsets for vectorized embedding lookups to avoid separate embedding tables per-codebook
- **`forward_kvcached()` process**: receive the embedding of `[bs, 1, dim]`, implement CUDA Graph safe scalar update through the pre-allocated `input_pos` buffer (`fill_(codebook_idx)`), and perform attention with KV cache → norm → output projection- **Computational characteristics of the 9-step loop**: There is only one small GEMM (`[bs, 1, fast_dim]` level) at each step, and the calculation amount is extremely small (μs level), but it requires a complete sequence of embedding lookup → linear projection → forward_kvcached → argmax → embedding lookup, generating a large number of kernel launches

#### 2.4 Coexistence of two KV Cache

This is a design point worth understanding in depth:

- **SGLang paged KV cache** (slow head): managed by `token_to_kv_pool_allocator`, supports prefix caching and dynamic allocation, attention backend is FlashAttention 3
- **Static KV cache** (fast head): A fixed-size tensor is pre-allocated by `audio_decoder.setup_caches()`, `zero_()` is cleared every time `reset_caches()` is executed, and does not participate in SGLang's memory pool management
- **Key question**: How do the two caches ensure address stability during CUDA Graph capture? The paged cache is managed through SGLang's `req_to_token_pool`, and the static cache is allocated once and does not change after being allocated through `setup_caches()` - the latter is naturally CUDA Graph safe.

#### 2.5 Why CUDA Graph is helpful for this model

This may be obvious, but it needs to be pointed out explicitly as a preparation for 2.6:

- The decode phase of S2-Pro is a **highly repetitive fixed kernel sequence**: each step executes the same 36-layer transformer + 9-step codebook loop, and the control flow is completely static - this is the scenario where CUDA Graph (the "static control flow" constraint in the first step) is best at
- The batch size of the Decode phase is relatively stable over a period of time, and SGLang's multi-bs graph management can cover it well.
- Model inference is latency-sensitive (TTS needs to generate speech in real time), eliminating launch overhead on the CPU side directly helps end-to-end latency

After confirming that "CUDA Graph works with S2-Pro", the next question is: What parts should the graph cover?#### 2.6 Core question: Why dual CUDA Graph is needed

**This is a driver issue for this entire article. **

PR #153 Previously, SGLang's CUDA Graph only covered the slow head (standard LLM transformer forward). Fast head (codebook loop) runs outside the graph as per-request post-processing. Combining the calculation feature analysis of 2.2 and 2.3:

- **Slow head's CUDA Graph has limited benefits**: As analyzed in 2.2, the core of slow head is a large GEMM like `mm(8×2560, 2560×6144)`, and a single kernel takes ms. The relative benefit of eliminating launch overhead is small.
- **Fast head without CUDA Graph is the real bottleneck**: As analyzed in 2.3, each step of the codebook loop is a μs-level small operator, and the 9-step loop generates a large number of kernel launches. Launch overhead accounts for the majority of execution time - this is exactly what CUDA Graph is best at solving.
- **There is also CPU scheduling overhead between the two heads**: After the graph replay of the slow head ends, the CPU needs to retrieve the results, call the codebook loop request by request, and then write it back - this CPU round trip is also overhead.

Therefore, **Core Insight: Unify slow head and fast head into a `forward()`, and let a CUDA Graph capture both** at the same time. TPS jumps from 55.6 to 88, with the gain mainly coming from fast head’s launch overhead elimination.

But "unifying into one graph" introduces huge engineering complexity - which is why the following chapters exist:
- Two KV caches must coexist in the same graph (analyzed in 2.4)
- All dynamic input must be passed in through persistent buffer (→ step 3)
- The initialization sequence must ensure that the graph captures the complete path (→ deferred capture in the third step)
- The looping and sampling of the Codebook loop must satisfy the static constraints of CUDA Graph (→ Step 4)

#### 2.7 PR #153 Architecture comparison before vs after

**Before (separated processing flow)**:
```
Text Model forward → LogitsProcessorOutput
                        ↓
S2ProSGLangOutputProcessor._codebook_loop_impl() ← per-request, not in graph
                        ↓
Codebook codes output
```
**After (unified forward)**:
```
S2ProSGLangModelRunner._update_vq_buffers() ← Write the codes of the previous step into persistent buffers
                        ↓
S2ProSGLangTextModel.forward()
    ├── VQ embedding combination (read from persistent buffers)
    ├── 36-layer Transformer (slow head)
    ├── Logits calculation
    └── _decode_codebooks() (constrained sampling + batched codebook loop)
                        ↓
S2ProSGLangModelRunner._build_outputs() ← Read output codes from persistent buffers
```

**Key Change**: `_decode_codebooks()` changes from an external per-request post-processing to a step inside `forward()`. This means that CUDA Graph can record transformer + sampling + codebook loop at once, eliminating all kernel launch overhead of the entire decode step.

### Step 3: Deferred Graph Capture——Why the initialization order is so important

- **DEPTH LEVEL**: Modify extensions
- **Goal**: Understand the design motivation of deferred graph capture from `factory.py` source code
- **Derivation from the first step**: This step directly corresponds to the fifth constraint of the first step "graph will not be automatically updated after recording". The beginning should explicitly point out this connection ("This directly corresponds to the last of the five constraints"), and then use specific code to show how this constraint drives the design of the initialization timing.
- **Writing Points**:
  - First paste the 6-step timing code of `factory.py`
  - Use questions to elicit core issues ("Why can't you capture it directly in Step 2?")
  - Explicitly quote the constraints in the first step to derive the answer (not simply "because CUDA Graph is static", but "review the fifth constraint in Chapter 1: the graph is static and will not be automatically updated after recording. Even if `setup_vq_decode()` is called later...")

#### 3.1 Initialization timing of `factory.py`

(Keep the original 6-step code + "Why can't it be captured directly in Step 2?" analysis, but the derivation process must explicitly refer back to the constraints of the first step)

#### 3.2 Buffer allocation of `setup_vq_decode()`

- List input/output/auxiliary buffers
- **Derivation from the first step**: The existence of these buffers corresponds to the second constraint of the first step "no dynamic memory allocation" - all tensors must be pre-allocated during capture, so `setup_vq_decode()` allocates all buffers at once before capture
- Security analysis: All buffers only modify their values through local operations - corresponding to the first constraint "pointer stability" of the first step.

### Step 4: Persistent Buffer and CUDA Graph security

- **DEPTH LEVEL**: Modify extensions
- **Goal**: Explain why in-place operations can ensure the security of CUDA Graph, and the read and write protocol of "external write value, graph internal read address"
- **Derivation from the third step**: The buffer is allocated in the third step. This step answers "After allocation, how can we safely read and write these buffers at runtime?"
- **Writing Points**:
  - List of in-place operations (`copy_()`, `fill_()`, `zero_()`, index assignment) - each corresponding to the first step of "pointer stability"
  - buffer read and write protocol (`_update_vq_buffers` → `forward` → `_build_outputs`) - derived the mode of "external write value, graph internal read address"
  - `input_pos.fill_()` mode - corresponds to the two constraints of "pointer stability" + "static control flow"
  - Greedy decoding selection - corresponds to the "no host-device sync" constraint- **Integrate each design choice into its own discussion**, do not have a separate "mapping table" section. Constraint mapping should appear naturally in the text ("This is the embodiment of the 'pointer stability' constraint in Chapter 1"), rather than a separate comparison table

> Note: The fourth step of the old plan is an independent "five-constraint engineering mapping table". This is a list-style writing method, which is inconsistent with the progressive derivation style. The correct approach is to integrate the constraint mapping into each specific discussion in steps three and four, as part of the derivation, rather than as a separate summary table.

### Step 5: SGLang CudaGraphRunner source code analysis

- **Depth level**: Modification and extension (SGLang is a self-developed system)
- **Goal**: Understand how `CudaGraphRunner` manages graph instances with multiple batch sizes, and how it works with the special needs of S2-Pro
- **Derivation from the first step**: The first step 1.2 deduces that "a graph can only serve one batch size", and 1.4 establishes a memory pool sharing mechanism. This step is the source code level implementation of these two concepts - they should be quoted explicitly ("The 'one bs one graph' principle derived in Chapter 1 is reflected in CudaGraphRunner as...")
- **Method**: Source code reading
- **Writing Points**:
  - The reason for the capture order (from large to small) - derived from the memory pool sharing mechanism of 1.4 (capture the big bs first and let the small bs reuse the memory)
  - The necessity of warmup run - corresponding to "no dynamic memory allocation"
  - bs padding strategy - the engineering corollary of "one bs one graph"
  - S2-Pro has additional requirements for capture - the graph is larger than ordinary LLM
  - eager fallback condition - the boundary case of "one bs one graph"

#### 5.1 `CudaGraphRunner` capture process

Analyze SGLang source code `sglang/srt/model_executor/cuda_graph_runner.py`:

- **capture_bs list**: Default `[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]` (12 batch sizes)
- **capture order**: from large to small (largest bs captured first). Reason: Large bs requires more memory. Capturing the large bs first allows the memory allocator to "see" the maximum memory requirement. Subsequent capture of the small bs can reuse the allocated memory.
- **warmup run**: Each bs does an eager forward before capture, triggering all possible memory allocations (including cuBLAS workspace, attention buffer, etc.) to ensure that there will be no unexpected allocations during capture.#### 5.2 The impact of S2-Pro on the capture process

When `text_model._vq_ready = True`, the forward of capture contains:

1. VQ embedding combination (read `_vq_codes`, `_vq_mask`)
2. 36-layer Transformer (RadixAttention + FFN)
3. logits calculation
4. `_decode_codebooks()`: constrained sampling + 9-step codebook loop

**Additional capture constraints**:
- The static KV cache of audio decoder must be allocated through `setup_caches(max_batch_size=max_bs)` before capture
- `self._audio_decoder.reset_caches()` in `_decode_codebooks()` is a `zero_()` operation - in-place operation, graph safe
- `forward_kvcached()` in the codebook loop involves the audio decoder's own attention calculation - these kernels will also be entered into the graph

**Key question**: Capture a graph containing a 9-step codebook loop, which means that the graph contains approximately `36 × 4 (transformer GEMM) + 9 × N (codebook loop kernels)` kernel nodes. This is significantly larger than the graph of a normal LLM. It is necessary to analyze the impact of graph size on replay latency.

#### 5.3 bs padding in Graph Replay

When actual batch size < captured batch size:

- `CudaGraphRunner` copies the actual input to the first `actual_bs` lines of the preallocated buffer
- graph replay still executes the complete captured_bs kernels, and the extra lines produce invalid calculations
- For S2-Pro, padding means that `_decode_codebooks()` also performs a full 9-step codebook loop for the padding lines - which is extra waste. But since the codebook loop is a small matrix operation, the additional overhead is very small#### 5.4 `can_run_cuda_graph` Judgment logic

Analyze under what circumstances S2-Pro will fallback to eager mode:

- prefill stage: sequence length is not fixed → no graph
- decode phase: bs exceeds the maximum capture bs → fallback
- chunked prefill: no graph
- S2-Pro's extend mode (`forward_batch.forward_mode.is_extend()`): go eager

### Step 6: The deep relationship between CUDA Graph and torch.compile - how the two optimization systems coexist

- **Deep Level**: Understanding Replication (CUDA Graph and torch.compile are both dependent infrastructure)
- **Goal**: Understand which level of overhead in the GPU execution pipeline each eliminates, and why SGLang chooses `max-autotune-no-cudagraphs`
- **Deduced from the previous article**: In the second step 2.6, we analyzed the different effects of CUDA Graph on slow head and fast head (large GEMM is not afraid of launch overhead, but small operator chains are). But the benchmark table at the beginning shows that partial compile can also improve CUDA Graph by another 36% - this shows that there are other people who have overhead that CUDA Graph cannot eliminate. This step starts from this data suspense and introduces a five-layer overhead model to explain
- **Method**: Conceptual framework + PyTorch source code analysis
- **Writing Points**:
  - Do not use ASCII diagrams for the five-layer overhead model, use mermaid or tables.
  - The choice of `max-autotune-no-cudagraphs` should be deduced from "SGLang already has CudaGraphRunner" (step 5) - if the inductor also manages graph, there will be "graph within graph"
  - This article is only an overview. In-depth analysis of CUDAGraph Trees and `fullgraph=True` will be left for subsequent articles.

#### 6.1 Essential differences between the two optimization systems

First, a clear conceptual framework needs to be established - CUDA Graph and torch.compile solve the overhead of different stages in the **GPU execution pipeline**:

```
CPU side GPU side
┌─────────────────┐ ┌──────────────────┐
│ Python interpreter │ ①Python overhead │ │
│ ↓ │ │ │
│ PyTorch dispatch│ ②Framework dispatch │ │
│ ↓ │ │ │
│ CUDA API call │ ③launch overhead ──→│ Kernel execution │
│ ↓ │ │ ↓ │
│ Wait for the next op │ │ ④Video memory read and write (bandwidth) │
│ │ │ ↓ │
│ │ │ ⑤Arithmetic calculation (computing power) │
└─────────────────┘ └──────────────────┘
```

| Overhead layer | The effect of CUDA Graph | The effect of torch.compile |
|---|---|---|
| ①Python overhead | Completely eliminated (replay does not go through Python) | Significantly reduced (traced graph bypasses Python dispatch) |
| ②Frame dispatch | Completely eliminated | Significantly reduced |
| ③launch overhead | **completely eliminated** (all kernels are submitted in one launch) | partially reduced (the number of kernels is reduced after fusion) |
| ④Video memory bandwidth | No impact | **Significant optimization** (operator fusion reduces intermediate tensor reading and writing) |
| ⑤ Arithmetic calculation | Does not affect | Possible optimization (Triton kernel may be better than cuBLAS, but it may also be worse) |

**Key Insight**: The two overlap in overhead ③ (both can reduce kernel launch), but only torch.compile is effective in ④. This explains why:
- CUDA Graph alone works extremely well for codebook loop (③ is the bottleneck)
- But CUDA Graph + torch.compile (additional optimization of ④) still has a 36% increase - because the small operator chain of the codebook loop still has a large number of intermediate tensor memory reads and writes

#### 6.2 The relationship between the four modes of torch.compile and CUDA Graph

| Mode | Meaning | CUDA Graph Behavior | S2-Pro Applicability |
|---|---|---|---|
| `default` | Basic optimization, no autotune | inductor does not manage CUDA graph | Available but little benefit |
| `reduce-overhead` | Optimization + automatic management of CUDA graph inside inductor | **inductor own capture/replay graph** (CUDAGraph Trees) | **conflict with SGLang's CudaGraphRunner** |
| `max-autotune` | Fully autotune + inductor to manage CUDA graph | Same as reduce-overhead, inductor manages graph by itself | **conflict** |
| `max-autotune-no-cudagraphs` | Fully autotune, but **not** let the inductor manage the graph | The inductor only generates an optimized kernel and does not touch the graph | **The mode used by SGLang** |**Why does SGLang use `max-autotune-no-cudagraphs`? **

This is a key architectural decision: SGLang has its own `CudaGraphRunner` responsible for graph capture/replay. If the inductor also does graph capture (`reduce-overhead` or `max-autotune` mode), there will be a conflict of **"graph within graph"** - the outer layer is SGLang's graph capture, and the inner layer is the inductor's CUDAGraph Trees. The two compete for control of stream capture.

Therefore, SGLang chooses the `no-cudagraphs` suffix: let the inductor be only responsible for **kernel optimization** (operator fusion + Triton autotune), and **graph management** is left to SGLang itself. This is a "division of labor" model:

```
torch.compile(mode="max-autotune-no-cudagraphs") → generate optimized Triton kernel
                          ↓
SGLang CudaGraphRunner.capture() → Input these optimized kernels into CUDA Graph
                          ↓
SGLang CudaGraphRunner.replay() → Submit all optimized kernels at once
```

#### 6.3 CUDAGraph Trees: Inductor’s internal graph management mechanism

Although SGLang does not use inductor's graph management, understanding its mechanism helps to understand why the two systems cannot be superimposed:

- **CUDAGraph Trees** is the internal implementation of inductor in `reduce-overhead` mode: creating an independent graph for each different execution path (different shape, different control flow branch), organized in a tree structure
- **Memory Pool Sharing**: All graphs share a memory pool, "no additional memory overhead"
- **Graph Break processing**: When torch.compile encounters an operation that cannot be traced (such as `.item()`, dynamic control flow), a graph break will be generated and the code will be divided into multiple graph partitions - each partition is captured independently
- **Execution process**: First call warmup → Second call to record graph → Subsequent call replay

**Conflict with SGLang CudaGraphRunner**:
- SGLang's `CudaGraphRunner` also captures on streams. If the inductor has captured part of the kernel (CUDAGraph Trees) internally, then the outer capture of SGLang will "see" an execution that has been graphed - this may lead to undefined behavior of "nested graphs within graphs"
- Memory pool management: inductor and SGLang manage memory pools respectively, and they do not communicate with each other.

#### 6.4 Constraints of `fullgraph=True` and challenges of S2-Pro

`fullgraph=True` requires torch.compile to trace the entire compiled function into a complete FX graph, without allowing any graph breaks. This means for S2-Pro's `_decode_codebooks()`:

**Requirements that must be met**:
1. `for cb_idx in range(1, self._num_codebooks)` loop must be fully expanded by torch.compile (`num_codebooks` is a constant → can be expanded)
2. `zero_()` in `self._audio_decoder.reset_caches()` must be an operation supported by torch.compile → Yes, in-place operations can be traced
3. There cannot be graph break in `self._audio_decoder.forward_kvcached()` → It is necessary to verify whether the KV cache index operation in attention is compile-clean
4. `torch.argmax` must be traceable → yes**Potential graph break risk** (Issue #172 Phase 2 needs to be resolved):
- `RadixAttention` (slow head): dynamic index involving paged KV cache → possible graph break
- Dynamic property access for `ForwardBatch` → possible graph break
- tensor parallel communication of `MergedColumnParallelLinear` → requires validation

This is why Issue #172 ranks backbone compile (Phase 2) after auxiliary module compile (Phase 1) - **fast head's `_decode_codebooks` is easier to satisfy `fullgraph=True`, while slow head has a lot of SGLang-specific dynamic operations**.

#### 6.5 Can the Triton kernel generated by the inductor be recorded by an external CUDA Graph?

This is the core assumption in `no-cudagraphs` mode: the Triton kernel generated by the inductor must be a "normal" CUDA kernel that can be recorded normally by SGLang's `CudaGraphRunner`.

**The answer is: Yes, but there are conditions**:
- The Triton kernel output by the inductor in `no-cudagraphs` mode is a standard CUDA kernel (via Triton's PTX codegen) and can be recorded by stream capture like the cuBLAS kernel.
- But the **guard mechanism** of the inductor may trigger recompilation during graph replay - if the shape/stride/dtype of the input tensor is different from that of the trace, the inductor will try to recompile, which is not allowed during CUDA Graph replay
- SGLang circumvents this problem through `CudaGraphRunner`'s **fixed bs + padding strategy**: each captured bs corresponds to a set of fixed shape inputs, and guard will not trigger

### Step 7: The iteration story of PR #153 - the rise and fall of torch.compile in S2-Pro

- **DEPTH LEVEL**: Modify extensions
- **Goal**: Through 7 commit and review discussions of PR, demonstrate how the five-layer cost model guides actual engineering decisions
- **Derivation from step six**: Step six establishes the five-layer overhead model and the `no-cudagraphs` division of labor principle. In this step, benchmark data is used to verify the model - 36% of the gain comes from overhead ④ (memory bandwidth), the 4% difference shows that overhead ⑤ (cuBLAS) is close to the optimal, and the 103s startup time is the inherent cost of autotune.
- **Method**: PR Archeology + Data Analysis
- **Writing Points**:
  - Benchmark data interpretation must refer back to the five-layer overhead model in step 6 ("Reviewing the overhead layer analysis in step 6: CUDA Graph has eliminated ①②③, but the overhead ④...")
  - The three-level decision-making logic is not an independent judgment, but a natural conclusion from the previous analysis.#### 7.1 The narrative lines of seven commits

| Serial number | Commit | Content | Meaning |
|---|---|---|---|
| 1 | `c153ae9` | "unified slow/fast head gaining huge efficiency gain" | Core implementation: unified forward + persistent buffers |
| 2 | `f621355` | "lint" | Code specifications |
| 3 | `c962aa6` | "torch.compile added in" | **Turning point**: Adding `server_args.enable_torch_compile = True`, triggering launch time problem |
| 4 | `78aafc7` | "setup_vq_decode before CUDA graph capture" | **Critical fix**: deferred graph capture mode |
| 5 | `dccf122` | "[refactor] tts eval for voice cloning" | Benchmark refactoring |
| 6 | `cf9396d` | "[feature] export server output of tokens" | Output interface adjustment |
| 7 | `20be04a` | "Acknowledge torch.compile discussion" | **Final decision**: Remove torch.compile, record analysis |

**Key turning point of iteration**:
- Commit 3 → 4: Found that torch.compile triggers Triton autotune for the entire model forward during graph capture. SGLang's `CudaGraphRunner` will call `torch.compile(model.forward, mode="max-autotune-no-cudagraphs")` when `enable_torch_compile=True` is set, and benchmark 18 candidate kernels for each GEMM shape of 36 layers of transformer × 12 bs.
- Commit 7: Based on the benchmarking results of Ratish1, torch.compile is officially removed.

#### 7.2 In-depth analysis of Ratish1’s three configuration Benchmark| Configuration | Health Ready | Graph Capture | Throughput (TTS) | Throughput (Voice Clone) |
|---|---|---|---|---|
| CUDA Graph only | 33.3s | 3.3s | 88.1 tok/s | 87.7 tok/s |
| Partial compile (fast head only) | 54.4s | 16.4s | 120.6 tok/s | 118.7 tok/s |
| Full-model compile | 137.0s | 107.0s | 125.7 tok/s | 122.5 tok/s |

**Interpret these data in conjunction with the theoretical framework of step 5**:

1. Where does the 36% throughput improvement of **Partial compile come from? ** Review the overhead layer analysis in 5.1: CUDA Graph has eliminated overhead ①②③, but each step of the 9-step loop of the codebook loop includes embedding lookup → linear projection → multi-head attention → RMSNorm → output projection. The intermediate tensor between these small operators still needs to be read and written through the video memory (overhead ④). The inductor of torch.compile fuses these operators into fewer Triton kernels, reducing the GPU-side memory round-trip. **Even with zero launch overhead, there is still a 36% gain from bandwidth optimization**.
2. **Full compile vs Partial compile only 4% difference**: Large GEMMs (`mm(8×2560, 2560×6144)`, etc.) in the transformer part are highly optimized by cuBLAS. The autotune log confirms that cuBLAS beats the Triton kernel in most shapes (overhead ⑤ is close to optimal). The only benefit of torch.compile on transformer is to integrate small operator chains such as layernorm + residual, but this part accounts for a small proportion.
3. **103.7s additional startup time**: `max-autotune-no-cudagraphs` mode does Triton autotune for each GEMM shape × each bs, total = 12 bs × 36 layers × ~4 linear layers × 18 candidates ≈ 31,000+ benchmark runs. This is an inherent cost of autotune and has nothing to do with CUDA Graph.
4. The startup time of **Partial compile is only +21s**: only fast head (a small number of small operators of the codebook decoder) is compiled, the autotune search space is much smaller than the full model, `54.4s - 33.3s = 21.1s` is acceptable.#### 7.3 Why did you choose not to compile in the end? ——Three-layer decision-making logic

**zhaochenyang20’s judgment** (from PR review):

1. **Abstraction level mismatch**: "Hard-coding mega cache into a single model file isn't the right abstraction... should live at the framework level" - The optimization of torch.compile should be the SGLang-Omni framework level capability, not a hack of a single model. Combined with the mode analysis of 5.2: Choosing `max-autotune-no-cudagraphs` itself is a framework-level decision (let SGLang manage the graph and let the inductor manage the kernel) and should not be hard-coded in the model code
2. **Interaction Complexity**: Combining the analysis of 5.4 and 5.5 - the interaction between torch.compile's guards/recompilation and CUDA Graph needs to be very careful. The codebook loop's `for cb_idx in range(1, self._num_codebooks)` needs to be fully expanded by `fullgraph=True`. Any graph break will cause compilation failure or performance degradation. And the `RadixAttention` of slow head is the hardest hit area of graph break.
3. **Granularity problem**: "Compiling the entire model forward is wasteful" - Combining the data of Ratish1, the only real benefit is the fusion of small operators in the fast head (36% gain), and the 4% increment of the slow head is not worth the 103s additional startup time.

**sdli1995's mega cache suggestion**: Use `torch.compiler.load_cache_artifacts()` / `save_cache_artifacts()` to cache the compilation results of the inductor (FX graph + Triton kernel binary), allowing subsequent startups to skip autotune. Actual measurements can reduce the compile time to 2s (LLM) + 10s (dual AR loop). But this has been deferred to [Issue #172](https://github.com/sgl-project/sglang-omni/issues/172).### Step 8: Issue #172 - Project blueprint of Framework-Level torch.compile

- **Depth Level**: Modification and Extension (SGLang-Omni is a self-developed system)
- **Goal**: Understand how the torch.compile optimization deferred in PR #153 is systematically implemented at the framework level
- **Derivation from step seven**: The conclusion of step seven is "Don't do torch.compile here, postpone it to the framework level". This step then asks "What should we do at the framework level?" - the three-stage plan is a systematic response to the three-layer decision-making logic of the seventh step.
- **Method**: Issue analysis + architectural design review
- **This article is only a summary**: List the three-phase plan and core design principles, and leave detailed analysis in subsequent articles.

#### Core issues of 8.1 Issue #172

[Issue #172](https://github.com/sgl-project/sglang-omni/issues/172) is titled "Framework-level fine-grained `torch.compile` + Mega Cache for Omni models". Three obstacles it addresses:

1. **Startup overhead**: Full-model compile takes 2~5 minutes, which is unacceptable in the production environment
2. **Abstraction Missing**: Currently you need to hardcode the compile logic in `factory.py` of each model
3. **Graph conflict**: The framework already manages CUDA Graph capture, and torch.compile must coexist cleanly with it

These three obstacles correspond to the three technical constraints analyzed in the fifth step: mode selection in 5.2, fullgraph requirement in 5.4, and guard mechanism in 5.5.

#### 8.2 Three-Phase Implementation Plan

**Phase 1: Partial Compile (auxiliary modules only)**

- **Goal**: Compile auxiliary modules (such as S2-Pro's codebook decoder), and the backbone remains eager + CUDA Graph
- **Framework API protocol**: The model implements the `get_compile_targets()` method, returning `dict[str, Callable]
```python
# Model side: declare "what is compilable"
class S2ProSGLangTextModel(nn.Module):
    def get_compile_targets(self) -> dict[str, Callable]:
        if not self._vq_ready:
            return {}
        return {"decode_codebooks": self._decode_codebooks_impl}
```

```py
# Framework side: decide "how to compile" (new file engines/omni/compile.py)
def apply_compile_targets(model, compile_mode="max-autotune-no-cudagraphs"):
    if not hasattr(model, "get_compile_targets"):
        return []
    compiled = []
    for name, fn in model.get_compile_targets().items():
        compiled_fn = torch.compile(fn, mode=compile_mode, fullgraph=True)
        setattr(model, f"_compiled_{name}", compiled_fn)
        compiled.append(name)
    return compiled
```

**Key Design Decision Analysis**:
- **`fullgraph=True` is mandatory** - combined with the analysis of 5.4, graph break will lead to unpredictable behavior in the CUDA Graph environment
- **compile mode is fixed to `max-autotune-no-cudagraphs`** - combined with the analysis of 5.2, graph management rights must be left to SGLang
- **The model only declares the target and does not call `torch.compile`** - implements the abstraction of "the model does not know that it has been compiled"
- **compile after `setup_vq_decode()` but before `init_device_graphs()` - Reuse deferred capture timing from PR #153

**Expected results**: S2-Pro ~121 tok/s, startup ~54s (compared to 88 tok/s / 33s of CUDA Graph only)

**Phase 2: Global Compile (full model forward)**

- **Goal**: Compile the entire `model.forward()` to get the remaining 4% throughput increment
- **Precondition**: SGLang's `RadixAttention`, `ForwardBatch`, `MergedColumnParallelLinear`, and RoPE cache modes must all be compile-clean (no graph break)

**Two coexistence strategies require benchmark**:

| Strategy | Implementation | Advantages | Disadvantages |
|---|---|---|---|
| **Layered** (layered management) | `mode="max-autotune-no-cudagraphs"` + SGLang's CUDA Graph capture | SGLang maintains full control over the graph | Need to ensure that the inductor kernel is fully transparent to SGLang graph capture |
| **Unified** (unified management) | `mode="reduce-overhead"`, let the inductor manage CUDA Graph | Deeper optimization (inductor can do cross-kernel memory planning) | Lose SGLang's multi-bs graph management, memory pool control and other fine capabilities |

**Key Challenges**:
- `RadixAttention` involves dynamic indexing of the paged KV cache - this is where graph breaks are most likely to occur
- The dynamic properties of `ForwardBatch` (`forward_mode`, `extend_seq_lens`, etc.) change in different prefill/decode modes
- S2-Pro's `_vq_ready` conditional branch is solidified during capture, but torch.compile's trace needs to process two paths

**Configuration interface**:

```bash
--compile-level none # CUDA graph only (default, zero startup overhead)
--compile-level partial # Phase 1: auxiliary modules only
--compile-level full # Phase 2: full model forward
```

**Phase 3: Mega Cache (eliminates startup overhead)**

- **Goal**: Cache the inductor's compilation products (FX graph + Triton kernel binary + autotune results), so that the second startup can skip all compile overhead
- **Cache Key Design**:
```python
cache_key = hash(
    model_path, # model weight identifier
    compile_level, # "partial" or "full"
    max_batch_size, # affects CUDA Graph shape
    torch.__version__, # inductor codegen may vary
    cuda_runtime_version, # CUDA runtime version
    gpu_arch, # sm_80 vs sm_90 produces different kernels
)
```
- **Two implementation options**:
  1. **Inductor-native**: Set `TORCHINDUCTOR_CACHE_DIR` to use inductor’s built-in FX graph + kernel cache
  2. **`torch.export` + `torch._inductor.aot_compile`**: pre-compiled into `.so` dynamic library (more control, more complex)

- **Invalidation policy**: Recompile when any cache key changes; provide `--clear-compile-cache` CLI command to manually reset
- **Expected results**: Under warm cache, even if `compile_level=full`, the startup time is close to baseline's ~33s

#### 8.3 The Five Design Principles of Issue #172 – Echoing the Decisions of PR #153

| Design Principles | Implications | Corresponds to lessons from PR #153 |
|---|---|---|
| The compile call does not appear in the model file | The model declares the target, and the framework determines the compilation strategy | Hardcoding `enable_torch_compile=True` in `factory.py` in PR #153 leads to an unmaintainable hack |
| Compile target must be tensor-in tensor-out | The compiled function cannot have external state access | `_decode_codebooks` accesses `self._audio_decoder` - needs to be refactored into a pure function `_decode_codebooks_impl` |
| `fullgraph=True` mandatory | graph break is not allowed | the `for` loop and `forward_kvcached` calls of the codebook loop must all be traceable |
| Eager-first readability | compile is optional acceleration, not the default behavior | PR #153 Finally choose CUDA Graph only as default and compile as opt-in |
| Configure the driver | Control it through the `ServerArgs` switch, no need to change the code | PR #153 Control the graph behavior through `server_args.disable_cuda_graph`, compile should do the same |

#### 8.4 Scope of application: not just S2-Pro

The framework design of Issue #172 is model-agnostic:| Model | Backbone | Auxiliary Module | Compile Opportunity |
|---|---|---|---|
| **S2-Pro** | Qwen3 + RadixAttention | Codebook decoder | +37% (aux); +43% (full) |
| **Qwen3-Omni** | Qwen3 thinker | Talker, encoders | Awaiting benchmark |
| **Future Models** | Any SGLang-backed LLM | Model-specific decoder | Automatically get compile support |

This means that learning this design is valuable not only for S2-Pro, but also for understanding how torch.compile is systematically managed by inference frameworks in general.

#### 8.5 Open questions requiring in-depth understanding

1. **Phase 2 RadixAttention compile feasibility**: Can the dynamic page table index of paged KV cache be processed by `fullgraph=True`? Or does it need to be modified on the SGLang side?
2. **Layered vs Unified strategy benchmark**: measured the throughput difference and memory footprint difference of the two strategies on S2-Pro
3. **Mega cache invalidation accuracy**: `torch.__version__` granularity is too coarse (minor version may not affect codegen), GPU arch granularity is too fine (same arch, different GPUs may have different optimal performance kernels)
4. **Purely functional transformation of compile target**: `_decode_codebooks` currently accesses `self._audio_decoder`, `self._semantic_bias` and other external states - how to reconstruct it to satisfy the "tensor-in tensor-out" constraint?

### Step 9: PyTorch CUDA Graph encapsulation layer and Memory Pool mechanism

> Note: The core concepts of this step (API mapping table + video memory sharing mechanism + high-water mark) have been deeply developed in the first step 1.3-1.4. There is no longer a need for a separate section in the article, but plan retains this step as a reference and its content has been integrated into steps one and five.

- **Depth level**: Understanding recurrence
- **Goal**: Supplement the details of PyTorch packaging, mainly as supporting material for the first and fifth steps
- **Method**: Code Analysis#### 9.1 Core API Mapping

| PyTorch API | CUDA Runtime API | Usage scenarios in S2-Pro |
|---|---|---|
| `torch.cuda.CUDAGraph()` | `cudaGraph_t` + `cudaGraphExec_t` | `CudaGraphRunner` maintains an instance for each bs |
| `graph.capture_begin()` | `cudaStreamBeginCapture()` | Before capturing starts, you need to ensure that all buffers have been allocated |
| `graph.capture_end()` | `cudaStreamEndCapture()` | Get the complete DAG after capture ends |
| `graph.replay()` | `cudaGraphLaunch()` | The actual execution of each decode step |
| `torch.cuda.graph()` context manager | Encapsulation begin + end | SGLang does not use this directly, but lower-level control |

#### 9.2 Memory Pool and graph sharing

- `torch.cuda.graph(pool=...)` allows multiple graphs to share the same memory pool, which is the basis for SGLang to manage 12 bs graphs
- **The cost of not sharing the pool**: Each graph independently allocates intermediate tensor memory, and each of the 12 graphs holds a copy → 12 times the intermediate tensor memory
- **Prerequisites for shared pool**: Only one graph is replaying at the same time (the decode phase of SGLang meets this condition)
- Meaning for S2-Pro: 12 bs graphs share the KV cache intermediate result memory of the audio decoder, and do not need 12 independent allocations
- **CUDA 12.4+ improvements**: The device memory of each kernel launch is reduced from 64KB, reducing the memory overhead of multiple graphs

#### 9.3 Interaction between torch.compile and PyTorch CUDA Graph package

When `mode="max-autotune-no-cudagraphs"`:
- The compiled function generated by the inductor is an ordinary Python callable, which internally calls the optimized Triton/cuBLAS kernel
- This callable can be captured normally by `torch.cuda.graph()` or SGLang’s `CudaGraphRunner`
- inductor does not do any graph capture internally (no CUDAGraph Trees)When `mode="reduce-overhead"` (not used by SGLang):
- inductor uses CUDAGraph Trees internally and manages capture/replay by itself
- If the outer layer is wrapped with another layer of `torch.cuda.graph()`, undefined behavior of nested capture will occur.

### Step 10: Design Review - Complete Trade-off Analysis of Unified Graph

- **DEPTH LEVEL**: Modify extensions
- **Goal**: Connect all the previous knowledge in series and conduct a complete review of the optimization path of S2-Pro
- **Derivation from the full text**: Each decision in the design decision matrix should refer back to a certain step of analysis in the previous article. The overhead eliminated by each layer in the optimization path table corresponds to the five-layer model in step six. This is not a new analysis, but the conclusion of the full-text logical chain.
- **Method**: Comprehensive table + summary judgment
- **Writing Points**:
  - Add a column of "corresponding CUDA Graph constraints" to the design decision matrix - let the constraints run through to the end
  - Optimize paths using markdown tables (no ASCII word art)
  - Add summary judgment at the end ("The five constraints are the outline, and all engineering design is the purpose - the outline is the outline" style)

#### 10.1 Design Decision Matrix

| Decision | Choice | Trade-off | Reasons | Future Evolution (Issue #172) |
|---|---|---|---|---|
| Unified vs separated graph | Unified | Single large graph vs two small graphs + intermediate data transmission | Unified to eliminate the CPU scheduling overhead between slow → fast | Maintain unified - compile does not change the graph structure |
| Greedy vs Sampling | Greedy (`torch.argmax`) | Loss of sampling diversity | CUDA Graph compatibility requirements; greedy is acceptable in TTS scenarios | Explore graph-safe sampling |
| Persistent buffers vs Dynamic tensors | Persistent | Additional memory usage (~several MB) | CUDA Graph requires address stability | Unchanged - compile does not affect buffer design |
| torch.compile | Off (defer to framework) | Give up 36% throughput improvement | Startup time trade-off + abstraction level + maintainability | Phase 1: partial compile restores 36% gain; Phase 3: mega cache eliminates startup overhead |
| Deferred capture | First init → setup_vq → capture | Increase initialization complexity | Ensure graph contains complete decode path | compile warmup inserted between setup_vq and capture |
| Graph management rights | SGLang CudaGraphRunner | Abandon inductor's CUDAGraph Trees | Maintain fine multi-bs graph control | `max-autotune-no-cudagraphs` unchanged; do not consider `reduce-overhead` |#### 10.2 The complete optimization stack of S2-Pro (from eager to final state)

```
Eager baseline (no optimization)
    │ Eliminate ①②③ → CUDA Graph only (PR #153, 88 tok/s, 33s startup)
    │
    │ Eliminate ④ for fast head → + Partial compile (Issue #172 Phase 1, ~121 tok/s, ~54s startup)
    │
    │ Eliminate ④ for slow head → + Full compile (Issue #172 Phase 2, ~126 tok/s, ~137s startup)
    │
    │ Eliminate compile startup → + Mega cache (Issue #172 Phase 3, ~126 tok/s, ~33s startup)
    ▼
Final state: CUDA Graph + Full compile + Mega cache (126 tok/s, 33s startup)
```

Each layer of optimization is **orthogonal and stackable**, thanks to the clear division of labor of "inductor tube kernel, SGLang tube graph" implemented by `max-autotune-no-cudagraphs` mode.

#### 10.3 Future optimization directions

1. **Phase 1 Implementation** (Issue #172): `get_compile_targets()` protocol + `apply_compile_targets()` framework function + S2-Pro’s `_decode_codebooks_impl` registration
2. **Mega cache integration** (Issue #172 Phase 3): `torch.compiler.save_cache_artifacts()` / `load_cache_artifacts()`, or `TORCHINDUCTOR_CACHE_DIR` environment variable
3. **CUDA Graph conditional nodes** (CUDA 12.4+): May allow conditional execution inside the graph, valuable for early stopping (ends early when encountering `im_end_id`) - but requires PyTorch level support
4. **Graph Update API**: `cudaGraphExecUpdate()` may allow modifying some parameters in the graph without re-capturing, reducing the overhead of multi-bs capture
5. **Sampling Recovery**: Explore the implementation of stochastic sampling in a CUDA Graph-compatible manner (pre-allocated random state + graph-safe `multinomial`)
6. **RadixAttention compile-clean transformation**: This is the core precondition of Phase 2 and requires the cooperation of SGLang upstream

### Step 11: Piecewise CUDA Graph - another path to the SGLang main repository

- **Depth level**: Modification and extension (SGLang is a self-developed system)
- **Goal**: Understand the design motivation and implementation mechanism of Piecewise CUDA Graph, and its comparison with monolithic graph
- **Derivation from the previous article**: The whole article has been discussing the engineering implementation of monolithic graph so far. But the "one bs one graph" limitation derived in step 1.2, the prefill fallback to eager in step five, and the graph break problem of RadixAttention in step eight Issue #172 Phase 2 - these limitations naturally lead to a question: "Is there a more flexible CUDA Graph solution than monolithic?" Piecewise is the answer to this question from the SGLang main repository.
- **Method**: Conceptual framework + source code reading
- **Writing Position**: Before design review. From the specific case of S2-Pro to the SGLang framework-level perspective
- **Writing Points**:
  - Three limitations are derived from the constraints of the first step (uncaptured operation → constraint three, prefill dynamic shape → "one bs one graph" inference, video memory pressure → high-water mark of 1.4)
  - Piecewise's split point mechanism should explain "at what operations it is split and why these operations cannot be captured"
  - Comparison table with monolithic should cover key dimensions- Point out that piecewise has embedded torch.compile - this is the framework-level implementation of the division of labor in the sixth step of "inductor manages kernel, SGLang manages graph"
  - Implications for S2-Pro: Not currently required, but may be required in the future

#### 11.1 Motivation: Limitations of Monolithic Graph

The solution in PR #153 is **monolithic CUDA Graph** - capturing the entire `forward()` as a graph. This works well for the decode stage of S2-Pro (fixed bs, fixed kernel sequence), but for the wider scenarios faced by the SGLang main warehouse, the monolithic graph has fundamental limitations:

1. **Uncatchable operations**: Operations such as FlashAttention, MoE dispatch (DeepEP, etc.) themselves cannot or are not suitable to be captured by CUDA Graph - they require dynamic shape or have internal host-device sync. Monolithic graphs cannot bypass these operations.
2. **Dynamic shape of Prefill/Extend**: The number of tokens in the Prefill stage varies widely (from a few to thousands), and it is impossible to pre-capture a monolithic graph for each token number.
3. **Video memory pressure**: Monolithic graph holds a complete graph and intermediate tensor for each batch size, which takes up a lot of video memory.

These limitations are exactly what Piecewise CUDA Graph aims to solve.

#### 11.2 Core idea: segmentation + segmented capture

**The core idea of ​​Piecewise CUDA Graph**: instead of treating the entire forward as a graph, it is divided at the boundary of **uncaptureable operations** (such as attention kernel, MoE dispatch), and the forward is split into several small subgraphs, and each subgraph is captured independently.

```
Monolithic Graph (PR #153 solution):
┌────────────────────────────────────────────────────┐
│ The entire forward() as a graph │
│ VQ combine → 36-layer Transformer → logits → codebook │
└─────────────────────────────────────────────────────┘

Piecewise Graph (SGLang main warehouse solution):
┌──────────┐ eager ┌──────────┐ eager ┌───────────┐
│ Subgraph │→ attn →│ Subgraph │→ attn →│ Subgraph │→ ...
│ (FFN, etc.) │ kernel │ (FFN, etc.) │ kernel │ (FFN, etc.) │
└──────────┘ └──────────┘ └───────────┘
```

Each subgraph covers the captureable part "between two non-captureable operations" (such as FFN, layernorm, residual, etc.). Uncatchable operations (attention, MoE dispatch) are still executed in eager mode.

#### 11.3 Split Points and three-phase execution

**Split Points Mechanism**:
- Declare split points through the `@register_split_op` decorator (such as MoE forward dispatch)
- Automatically cut the FX graph at these locations during compilation to generate several subgraphs
- Each subgraph is compiled independently (`torch.compile`) and captured

**Three-stage execution model** (per subgraph):
1. **Compilation**: `torch.compile` compiles subgraph and processes dynamic shape
2. **CUDA Graph Capture**: capture each subgraph with predefined token length (4, 8, 12, ..., 2048+)
3. **Steady-State Replay**: Find the latest captured size during runtime and replay after pad

**Capture Size Schedule** (default):

```
4-32: Step size 4
48-256: Step size 16
288-512: Step size 32
640-1024: Step size 64
1280-4096: Step size 256
4608-max: step size 512
```

The design logic of this schedule: small token numbers (decode phase) require finer granularity to reduce padding waste; large token numbers (prefill phase) are not sensitive to granularity.

#### 11.4 Comparison with PR #153 Monolithic Graph

| Dimensions | Monolithic Graph (PR #153) | Piecewise CUDA Graph |
|---|---|---|
| **Capture range** | Entire forward | Each layer/section independent |
| **Attention processing** | Contained by graph (RadixAttention) | Executed eagerly at split point |
| **Applicable stage** | decode only (fixed bs) | Decode + Prefill (multiple token numbers) |
| **Uncatchable operation** | Must be bypassed or replaced | Naturally supported at split points |
| **Memory Pool** | Each bs and a graph share a pool | Global shared pool, shared by all subgraphs + all capture sizes |
| **Relationship with torch.compile** | Orthogonal (compile first and then capture) | **Inline** (compile first and then capture for each subgraph) |
| **Complexity** | Simple and direct | The framework layer is complex, but transparent to model developers |

**Key Insight**: The monolithic graph of PR #153 works because the decode phase of S2-Pro satisfies all five constraints of CUDA Graph - especially RadixAttention happens to be captured in the decode phase (fixed bs, fixed seq position). But for the wider scenarios faced by the SGLang main warehouse (multiple attention backends, MoE models, variable length prefills), the piecewise solution is a more general solution.

#### 11.5 Key points of source code reading

Key code files:
- `sglang/srt/model_executor/piecewise_cuda_graph_runner.py` - main runner, manages capture and replay
- `sglang/srt/compilation/cuda_piecewise_backend.py` - compile + capture per-subgraph
- `sglang/srt/compilation/backend.py`——graph segmentation logic
- `sglang/srt/compilation/compilation_config.py`——split points, capture sizes configuration
- `sglang/srt/server_args.py`——`--enable-piecewise-cuda-graph` and other CLI parametersDesign points that need attention:
1. **Reverse capture order**: Same as `CudaGraphRunner`, capture from large token number to small token number, reuse memory pool
2. **Global shared memory pool**: All subgraph × all capture sizes share a pool
3. **Tight integration with torch.compile**: Each subgraph is first optimized by `torch.compile` (inductor), and then captured as CUDA Graph - this is exactly the framework-level implementation of the "inductor manages kernel, SGLang manages graph" division of labor model discussed in PR #153 and Issue #172
4. **Eager fallback**: automatic fallback when capture fails or exceeds max tokens

#### 11.6 Implications for S2-Pro / SGLang-Omni

Can the ideas of Piecewise CUDA Graph be applied to SGLang-Omni?

- **Not required for current S2-Pro decode**: the monolithic graph in the decode stage is enough (fixed bs, all operations can be captured, performance is already very good)
- **But the prefill stage may benefit**: S2-Pro's prefill is currently eager. If prefill becomes a bottleneck, the piecewise graph can cover the part other than attention in prefill.
- **Broader significance**: When SGLang-Omni connects to more complex models (multimodal, MoE) in the future, the piecewise solution may become a required option
- **Issue #172 Phase 2 correlation**: Phase 2 needs to compile the entire `model.forward()`, but RadixAttention may graph break - piecewise's "split at uncatchable operations" idea is an elegant solution

#### 11.7 Related Issues and Status

- [Feature Roadmap for Prefill (Piecewise) CUDA Graph - Issue #11490](https://github.com/sgl-project/sglang/issues/11490)
- [TODO: Piecewise CUDA Graph Default Enable - Issue #18130](https://github.com/sgl-project/sglang/issues/18130)
- [Docs: Add doc for piecewise CUDA graph - Issue #18267](https://github.com/sgl-project/sglang/issues/18267)
- Current status: **Enabled by default**, can be turned off via `--disable-piecewise-cuda-graph`## Recommended resources

### Official Documentation
- [NVIDIA CUDA Programming Guide - CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [CUDA Runtime API - Graph Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)
- [PyTorch CUDA Graphs](https://pytorch.org/docs/stable/cuda.html#cuda-graphs)
- [PyTorch torch.compile Documentation](https://pytorch.org/docs/stable/torch.compiler.html)
- [PyTorch CUDAGraph Trees](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html) - Understand the graph management mechanism inside the inductor, and why SGLang chooses the `no-cudagraphs` mode

### Code repository (need to determine commit hash)
- SGLang `CudaGraphRunner` (monolithic): `sglang/srt/model_executor/cuda_graph_runner.py`
- SGLang `PiecewiseCudaGraphRunner`: `sglang/srt/model_executor/piecewise_cuda_graph_runner.py`
- SGLang Piecewise compilation backend: `sglang/srt/compilation/cuda_piecewise_backend.py`, `sglang/srt/compilation/backend.py`
- SGLang-Omni PR #153 (merge commit `cd9aaf3`): https://github.com/sgl-project/sglang-omni/pull/153
  - `sglang_omni/models/fishaudio_s2_pro/sglang_model.py`——Unified model implementation
  - `sglang_omni/models/fishaudio_s2_pro/factory.py`——deferred graph capture
  - `sglang_omni/models/fishaudio_s2_pro/runtime/s2pro_sglang_ar.py`——buffer reading and writing protocol
  - `sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/modeling.py`——audio decoder (`FishQwen3AudioDecoder`)- SGLang-Omni Issue #172: https://github.com/sgl-project/sglang-omni/issues/172 - Three-stage project blueprint for Framework-level torch.compile
- torch-memory-saver: https://github.com/fzyzcjy/torch_memory_saver
- PyTorch CUDA Graph wrapper: `torch/cuda/graphs.py`
- PyTorch Inductor CUDAGraph Trees: `torch/_inductor/cudagraph_trees.py` - Understand how the inductor internally manages the graph in `reduce-overhead` mode

### Community Articles
- [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) — PyTorch official blog
- [CUDA Graphs in the Deep Learning Ecosystem](https://developer.nvidia.com/blog/cuda-graphs/) — NVIDIA Developer Blog
- [torch.compile, the missing manual](https://docs.google.com/document/d/1y811KBmTLBEaYMCFBfbgLzrdSN0VRz51CkLMpTk7vAU) — Internal documentation for torch.compile from the PyTorch team

## Article structure suggestions

- **Article type**: code-walkthrough + sys-design hybrid (with PR #153 as the main narrative line, CUDA Graph engineering implementation as the core, and torch.compile discussion as the summary)
- **Suggested path**: `torch/cuda-graph/readme-3.md`
- **Suggested Title**: `CUDA Graph Revisited: Dual CUDA Graph Optimization in TTS Model`
- **Series Attribution**: CUDA Graph series, third article (the first two articles are brief analysis and S2-Pro comparative analysis respectively)
- **Opening structure** (refer to the user's actual writing style):
  1. First review the first article in the series, admit that the understanding was shallow at the time, and introduce the engineering practice of this PR.
  2. **Then the benchmark data table is displayed** (three configuration comparison + TPS improvement number), and attract readers with the results
  3. Short roadmap (numbered list of 4 items, no more than one sentence/item)
  4. Acknowledge in a casual style
  5. **Do not write templated statements** (such as "This article is analyzed based on commit xxx")
- **Core principles for chapter order: Concept → Model → Code**:
  - The conceptual framework of CUDA Graph must be established first (capture/replay three phases, pointer stability, static control flow constraints) so that readers can get the "analysis tools"
  - Re-introducing the characteristics of the S2-Pro model and using a conceptual framework to explain "why this model poses challenges to CUDA Graph"
  - Finally enter the code implementation to show how the concept is implemented in the project
  - **Never the other way around** - without a conceptual basis, the reader cannot understand the design choices in the code
- **Estimated Chapters**:1. **CUDA Graph conceptual framework**: capture → instantiate → replay three-stage mechanism, pointer stability constraints (graph records GPU virtual address), static control flow requirements, no dynamic memory allocation, no host-device sync - to establish the reader's "analysis toolbox"
  2. **S2-Pro model: why it challenges CUDA Graph**: Dual-AR dual-head architecture (slow head + fast head), small kernel launch bottleneck of 9-step codebook loop, coexistence of two KV caches - use the conceptual framework in Chapter 1 to explain the CUDA Graph constraints brought by each feature
  3. **Deferred Graph Capture**: Initialization sequence of `factory.py` - why setup must be done first and then capture (mapped to the concept of "graph is static" in Chapter 1)
  4. **Persistent Buffer design**: pointer stability of pre-allocate + `copy_()` (mapped to the address stability constraints of Chapter 1), buffer read and write protocol
  5. **SGLang CudaGraphRunner source code reading**: multiple bs graph management, memory pool sharing, eager fallback conditions
  6. **The relationship between CUDA Graph and torch.compile**: five-layer overhead model, four compile modes, and the division of labor philosophy of `no-cudagraphs` (summary, detailed analysis will be left for subsequent articles)
  7. **The rise and fall of torch.compile in S2-Pro**: Ratish1 benchmark, three-layer decision logic
  8. **Issue #172 Summary**: Introduction to the three-phase plan (detailed analysis will be left for subsequent articles)
  9. **Piecewise CUDA Graph**: Another path to the SGLang main warehouse - design comparison of monolithic vs piecewise, split points mechanism, three-stage execution model, relationship with S2-Pro solution and inspiration
  10. **Design review**: S2-Pro complete optimization stack and future direction (including piecewise’s outlook)

## Draft completion analysis

Draft path: `torch/cuda-graph/readme-2.md`

### Completed part
- Description of the dual-head architecture of S2-Pro (basic concept of slow head + fast head)
- CUDA Graph capture changes problem analysis and root cause location from seconds to minutes
- Cost comparison table of CUDA Graph vs torch.compile (eliminating launch overhead vs operator fusion)
- Effect analysis of Transformer (slow head) and Codebook Loop (fast head)
- Why SGLang LLM usually opens discussions on both
- Conclusion: Closing torch.compile and CUDA Graph are the main optimization methods
- Issue: Enable the overlay optimization exploration solution of torch.compile + CUDA Graph separately for the codebook loop### Completeness
- **Relative to the topic of the draft itself (CUDA Graph vs torch.compile comparison analysis)**: ~90%
- **Relative to the full objectives of this study plan**: ~15%

### Parts to be added (by priority)
1. **[High priority] Deep relationship between CUDA Graph and torch.compile** (Step 5): Five-layer overhead model of GPU execution pipeline, differences in graph management strategies of four compile modes, "max-autotune-no-cudagraphs" philosophy of "division of labor", conflict analysis between CUDAGraph Trees mechanism and SGLang CudaGraphRunner, `fullgraph=True` constraint in S2-Pro Specific challenges in - The draft comparison stops at a high-level description of "eliminating disparate overhead" and lacks an in-depth analysis of PyTorch's internal graph management mechanism
2. **[High priority] Framework-level engineering blueprint of Issue #172** (Step 7): three-phase implementation plan, `get_compile_targets()` protocol design, technical details and challenges of Phase 1/2/3, mapping of five design principles and lessons learned from PR #153, benchmark plan for Layered vs Unified coexistence strategy - not covered at all in the draft
3. **[High priority] Complete code implementation analysis of PR #153** (steps 1 and 2): buffer reading and writing protocols of `setup_vq_decode()`, persistent buffers, deferred capture, `_decode_codebooks()`, `_update_vq_buffers()/_build_outputs()`
4. **[High priority] Source code level in-depth analysis of S2-Pro model architecture** (Step 1): `forward_kvcached()` of `FishQwen3AudioDecoder`, static KV cache design, graph security mode of `input_pos.fill_()`, coexistence of two KV caches
5. **[Medium priority] SGLang CudaGraphRunner source code reading** (Step 4): multi-bs graph management mechanism, capture order and memory strategy, interaction with S2-Pro persistent buffers
6. **[Medium priority] PR iteration story** (Step 6): torch.compile addition → problem → removal process, Ratish1’s benchmark data in-depth interpretation (combined with five-layer overhead model), sdli1995’s mega cache suggestion
7. **[Medium priority] PyTorch CUDA Graph encapsulation layer** (step 8): memory pool sharing mechanism, transparency of inductor kernel to external graph capture in torch.compile `no-cudagraphs` mode