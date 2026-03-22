# Chunk parallel calculation of GAE in PPO (implementation based on slime)

> Corresponding to the original text on Zhihu: "Parallel computing of GAE in chunks in PPO" (https://zhuanlan.zhihu.com/p/1975237289425798560)
> Corresponding code: [`THUDM/slime` PR #850 — Chunk-Scan GAE](https://github.com/THUDM/slime/pull/850)

## 1. **TL;DR**

In this article, the author made a performance improvement around PPO + GAE in the slime framework:

**Background**: In the agentic RL scenario, when the sequence is extremely long, Slime’s original GAE calculation is to scan the samples serially in batches from the end to the beginning. And this will directly become a training bottleneck.

**Things to do**:

1. Slime first changes the traditional serial backward recursive method of calculating GAE to first divide GAE into multiple chunks according to time, and then in reverse chronological order, use the chunk at the current traversed time point and the `lastgaelam` calculated from the previous chunk to gradually derive the final GAE.
2. This article draws on the idea of ​​*linear attention* [**@sonta**](https://www.zhihu.com/people/buhezuobugaoxing). Through block prefix scanning, multiple calculated local GAEs are merged to calculate the final GAE. There is no dependence between the calculations of each Chunk.

**Effect**:

1. In slime, GAE calculation time is accelerated by **100×–300×**;
2. The degree of parallelism depends on `chunk_size`. Without OOM, the larger the `chunk_size`, the more obvious the acceleration.

## 2. Technical background: Why do we need GAE Chunk-Scan?

### 2.1 Why does GAE become a bottleneck now?

In RLHF/Agentic RL, PPO is still a very commonly used and stable algorithm. We need to calculate advantage on each token, the most common one is GAE (Generalized Advantage Estimation). The standard writing method of GAE is a recursive formula from back to front. For sequence length T, it is O(T) serial dependence.

In slime, the GAE algorithm is implemented as follows:

```python
lastgaelam = torch.zeros(B, device=device, dtype=dtype)
adv_rev = []

for t in reversed(range(max_len)):
    next_value = full_values[:, t + 1] if t < max_len - 1 else 0.0
    delta = full_rewards[:, t] + gamma * next_value - full_values[:, t]
    lastgaelam = delta + gamma * lambd * lastgaelam
    adv_rev.append(lastgaelam)

full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
```
In the initial implementation, slime pursued "support for variable-length sequences". The advantage is that when calculating the model, there is no need to padding to max_len of all sequences, so as to avoid wasting invalid calculations. Therefore, when calculating GAE, it is calculated sequence by sequence instead of batch calculation, which caused a performance bottleneck. We quickly changed it to the common writing method of "padding to max_len, and then calculating GAE in batches". But unfortunately, this is not enough to achieve the best possible performance. It is still serial in the time dimension, which makes it still very difficult in long sequence scenarios.

On this basis, the author combined the idea of ​​"chunk parallelism + lightweight recursion between chunks" mentioned by [**@sonta**](https://www.zhihu.com/people/buhezuobugaoxing) when talking about *linear attention*, and tried to transform GAE into a "prefix scan (scan)" problem that can be parallelized at the chunk level.

### 2.2 The problem of exploding video memory under complete matrix calculation

If you want to parallelize GAE, there is actually a very elegant solution, which is written directly as matrix multiplication:

- Write GAE as $A_t = \sum_{k=t}^{T-1} w^{k-t} \delta_k$ (where $w = \gamma \lambda$)

-Construct a T×T upper triangular weight matrix W, and then do $A = \delta W^\top$

This is completely parallelizable, but this directly results in both time complexity and space complexity being O(T²). Once T reaches the 64K or 128K level, OOM will occur directly.

> There is a solution in torchrl to [use conv1d to reduce the time complexity to O(T)](https://github.com/pytorch/rl/blob/8570c25a745da54ca647b8a70231112f063d1421/torchrl/objectives/value/utils.py#L13), but the space complexity is still O(T²), so there will still be the above OOM problem.

Therefore, we hope to find a GAE calculation method that can take into account parallelism and ensure controllable video memory.

## 3. Architecture design: from serial GAE to Chunk-Scan GAE

### 3.1 GAE

Before we begin, let's review standard GAE:

We write delta as
$$
\delta_t = r_t + \gamma V_{t+1} - V_t
$$
Then the advantage of GAE is
$$
A_t = \sum_{k=t}^{T-1} (\gamma \lambda)^{k-t} \delta_k
$$
It can also be written in the form of backward recursion:
$$
A_t = \delta_t + \gamma \lambda A_{t+1}, \quad t = T-1, T-2, \dots, 0
$$

### 3.2 Option 1: Serial solution

The current version of slime has actually given the answer:

```python
lastgaelam = torch.zeros(B, device=device, dtype=dtype)
adv_rev = []

for t in reversed(range(max_len)):
    next_value = full_values[:, t + 1] if t < max_len - 1 else 0.0
    delta = full_rewards[:, t] + gamma * next_value - full_values[:, t]
    lastgaelam = delta + gamma * lambd * lastgaelam
    adv_rev.append(lastgaelam)

full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
```
- Advantages: simple implementation, stable numerical value;

- Disadvantages: This version is completely serial in the time dimension, and the performance is not good in long sequences.

### 3.3 Option 2: Pure matrix solution

Use forward expansion:
$$
A_t = \sum_{k=t}^{T-1} w^{k-t} \delta_k,\quad w = \gamma \lambda
$$
We can construct a T×T weight matrix W:
$$
W_{t,k} =
\begin{cases}
w^{k-t}, & k \ge t \\
0, & k < t
\end{cases}
$$
So there is:
$$
A = \delta W^\top
$$

- Advantages: Matrix multiplication can be highly parallelized on GPU;
- Disadvantages: Very easy to OOM

### 3.4 Solution 3: Chunk-Scan

We can split the entire sequence into several chunks of length C:

```
First chunk: 0 - C-1
Second chunk: C - 2C-1
...
The cth chunk: cC - (cC + L_c - 1)
```Define GAE recursion on the reverse sequence:
$$
S_i = \widetilde{\delta}_i + w S_{i-1}, \quad w = \gamma \lambda, \quad S_{-1} = 0
$$
For the c-th chunk, define the "cross-chunk state":
$$
s_{\text{prev}} = S_{cC - 1}
$$
When c = 0, there is $s_{\text{prev}} = S_{-1} = 0$;

Now consider the t-th element inside chunk c (local index t = 0..L_c-1):

- Global index $i = cC + t$;
- Expand the recurrence relationship to get:

$$
\begin{aligned}
S_{cC + t}
&= \widetilde{\delta}_{cC + t}
 + w \widetilde{\delta}_{cC + t - 1}
 + \cdots
 + w^t \widetilde{\delta}_{cC}
 + w^{t+1} S_{cC - 1}
\end{aligned}
$$

Take out the part "in the current chunk" separately:
$$
s^{(c)}_t = \sum_{k=0}^{t} w^{t-k} \widetilde{\delta}_{cC + k}
$$
So the final formula can be written as:
$$
\boxed{
S_{cC + t} = s^{(c)}_t + w^{t+1} \, s_{\text{prev}}, \quad t = 0, \dots, L_c - 1
}
$$
This means:

- The local part $s^{(c)}_t$ can be calculated in parallel using matrix/conv within the chunk;
- Only one scalar state `s_prev` needs to be maintained across chunks, and serial recursion is enough.

Time complexity: O(T·C)
Space complexity: O(T + C²)

The above is a very rigorous formula derivation process. In fact, simply speaking, the core idea of Chunk-scan is:

1. Cut the long sequence into several small chunks
2. Let the GPU calculate the recursion in each chunk in parallel. The above formula obtains the part $s^{(c)}_t$ that can be calculated in parallel.
3. Then combine the results of these chunks

### 3.5 Pseudocode implemented by [`THUDM/slime` PR #850 — Chunk-Scan GAE](https://github.com/THUDM/slime/pull/850)

The following is a pseudocode that shows how to write Chunk-Scan GAE as a batch calculation function. The code prototype comes from: [`THUDM/slime` PR #850 — Chunk-Scan GAE](https://github.com/THUDM/slime/pull/850)

```pseudocode
function chunked_gae(rewards, values, gamma, lambda, chunk_size):

    w = gamma * lambda

    # 1. Calculate δ_t at each step
    deltas = compute_deltas(rewards, values) # δ_t = r_t + γV_{t+1} - V_t

    # 2. Reverse time sequence (recursion from back to front -> from left to right in reverse sequence)
    deltas_rev = reverse_time(deltas)

    # 3. Pad to an integer multiple of chunk_size and split into several chunks
    deltas_chunks = split_into_chunks(deltas_rev, chunk_size)

    # 4. Precompute a small core for scanning "inside each chunk":
    # Given a section Δ[0..C-1], calculate s_local[t] = Σ_{k≤t} w^(t-k) * Δ[k]
    kernel = build_chunk_kernel(chunk_size, w) # C×C upper triangular matrix
    pow_vec = build_power_vector(chunk_size, w) # [w^1, w^2, ..., w^C]

    # 5. All chunks perform partial scans in parallel internally.
    # local_scan[c, t] = s_local^(c)[t]
    local_scans = []
    for each chunk in deltas_chunks in parallel:
        s_local = chunk @ kernel # You can use any parallel implementation here.
        local_scans.append(s_local)

    # 6. Propagate "prefix status" serially between chunks s_prev
    s_prev = 0
    full_scan_rev = empty_like(deltas_rev)

    for c from 0 to num_chunks-1:
        s_local = local_scans[c] # The result inside the current chunk, length L_c

        #Inject cross-chunk status:
        # S_global[t] = s_local[t] + w^(t+1) * s_prev
        S_global = s_local + s_prev * pow_vec[0:L_c]

        write_into(full_scan_rev, chunk_index=c, values=S_global)

        #The starting state of the next chunk = the last position of the current chunk
        s_prev = S_global[L_c - 1]

    # 7. Remove padding and reverse to forward time
    advantages = reverse_time(remove_padding(full_scan_rev))

    # 8. returns are generally V_t + A_t
```

```pseudocode
    returns = values + advantages

    return advantages, returns

```
## 4. Realize the effect

According to the experimental results of the original article, the implementation effect is very impressive:

| No chunk | chunk size = 64 | chunk size = 128 | chunk size = 256 | |
| --------------- | --------------- | ---------------- | ---------------- | ------------------ |
| B=256, T=131072 | 5.935994s | 0.070122s | 0.034059s | 0.018390s ( x317 ) |
| B=128, T=65536 | 2.902570s | 0.232986s | 0.017645s | 0.009134s |

You can see:

1. When T=131072, `chunk_size=256`, the acceleration ratio is about **317×**;
2. When T=65536, `chunk_size=256`, the acceleration ratio is also very impressive;

As long as there is enough video memory to increase the chunk size, the degree of parallelism can be greatly increased, and the calculation time of GAE can also be reduced considerably.

## 5. Specific usage: How to use Chunk-Scan GAE in slime?

Chunk-Scan has been adopted as the default training behavior, so for user installation or migration, only the image needs to be updated.

### 5.1 Installation

1. Pull the latest version of the docker image and ensure that it contains the changes of [`THUDM/slime` PR #850 — Chunk-Scan GAE](https://github.com/THUDM/slime/pull/850). As of 11/24, the official docker image has not been updated.
2. Deploy services according to official guidelines: https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md
3. Use Chunk-Scan to train PPO using default parameters.

### 5.2 Migration

1. Upgrade to the latest version of the docker image and ensure that it contains the changes of [`THUDM/slime` PR #850 — Chunk-Scan GAE](https://github.com/THUDM/slime/pull/850). As of 11/24, the official docker image has not been updated.
2. Just resume training

## 6. Future plans

1. More systematic benchmark & visualization tools

   A one-click script is provided to facilitate users to evaluate whether their tasks are worth turning on Chunk-Scan.

2. Test the performance of the overall framework more comprehensively, measure the time-consuming situation of each part in a more fine-grained manner, and identify similar potential problems.

3. Check whether there are situations in other parts of the code where the concurrency can be improved by modifying the algorithm. If so, you need to explore the possibility of optimization.

## 7. Engineering Appendix: Pitfalls and Things Learned

1. The fact that GAE has become a bottleneck is an example of "using experimental results to correct engineering intuition." Before the experimental data actually ran out, it was difficult to imagine that GAE calculations would become the bottleneck of the PPO pipeline. Therefore, for a mature framework, when testing performance, the granularity of performance testing should be divided into fine enough parts to discover some problems that may have been overlooked at the beginning of the design.