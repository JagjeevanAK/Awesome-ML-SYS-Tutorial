# Zero-Overhead Batch Scheduler

> About the author: I am Wang Guanhua, a second-year undergraduate and postgraduate student in computer science at the University of Electronic Science and Technology of China (26 years old). I am currently looking for an internship in the direction of inference acceleration. Welcome to contact me! 281484683@qq.com
> Reference for this article [https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/lmsys_1st_meetup_sglang.pdf](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/lmsys_1st_meetup_sglang.pdf)

# Introduction

In traditional inference systems, CPU scheduling and GPU calculations are performed serially. The two need to wait for each other before they can continue execution, which results in a long unnecessary Bubble on the GPU, as shown in the following figure:

![image-20250320212055165](image/pipeline1.png)

An unoptimized engine may spend half of its time on CPU scheduling. The figure below is the model running overhead data of vLLM version 0.5.4. The Scheduler overhead is serious (of course VLLM has also updated overlap now).

![](image/breakdown.png)

We can overlap CPU scheduling with GPU computation. The scheduler runs a batch ahead of time and prepares all the metadata needed for the next batch. This keeps the GPU busy and hides the expensive overhead of scheduling, as shown in the figure.

![image-20250320212202164](image/pipeline2.png)

In this article, we will combine SGLang's code and analyze some key codes of SGLang's implementation of overlap. First, let’s logically analyze the steps of the pipeline.

# Pipeline logic explanation

## What are the steps in reasoning?

Before building the pipeline, we need to sort out the steps of SGLang's reasoning in order to better analyze which steps can overlap. The reasoning process of SGLang is mainly divided into the following four stages:

1. Pre schedule:

   - Collect incoming requests from the front end and put them into a waiting queue.
   - Scheduling from the waiting queue involves Radix Tree and Longest Prefix Matching algorithms.
   - Allocate the memory resources required by Token for each request.
2. Compute batch:

   - Send the batch to the GPU for one-step (i.e. one iter of Continue Batch) inference
3. Sample：

   - Sampling is performed based on the Logit output by the model to generate the Token for the next step.
4. Post schedule:

   - After one step of reasoning is completed, dynamically check whether the request satisfies the end condition (Check Finish Condition).
   - Remove completed requests from the batch and send them to Detokenizer for processing, and finally return the results to the front end.

That is, in the serial execution of the following table, each step is blocked synchronously. You must wait for one process to complete before proceeding to the next process.```python
Pre schedule -> Compute batch -> Sample -> Post schedule
```
## How do I overlap these steps?

We noticed that the two stages of Compute batch and Sample are GPU heavy, while the two stages of schedule are CPU heavy. When multiple batches are pipelined, we can use GPU Compute and Sample to overlap the post scheduler of the previous batch and the pre scheduler of the current batch.

In fact, if we want the GPU to execute code asynchronously, we must have a function on the CPU to launch the Kernel to the GPU; and when the kernel returns, we also need a function to help handle logits. Here, we divide the CPU into two parts:

1. CPU-S (Scheduler CPU): Responsible for scheduling tasks, including Pre-schedule and Post-schedule.
2. CPU-L (Launch CPU): Specifically responsible for the startup of the Kernel and the processing of the results returned by the Kernel.

After CPU-S passes the scheduled batch to CPU-L, it immediately moves to the next step, allowing the GPU to perform asynchronous calculations without waiting for the Kernel to start or the processing of the returned results.

A picture is worth a thousand words, and the picture below represents an overlapping assembly line. What we overlap is actually the dotted line as the dividing line, which overlaps **CPU-S** and **CPU-L + GPU** as a whole. In the figure, we assume that the first batch has been scheduled and run directly. The RunBatch stage means that we transfer the task from CPU-S to CPU-L. **Please pay careful attention to the batch numbers of CPU-S. **

![image-20250320212426343](image/real_pipeline1.png)

In a pipeline, the Scheduler can continue scheduling the next batch without waiting for the Compute stage to return results. However, it cannot schedule multiple batches continuously at once (for example, when Compute Batch1 is calculated, 5 batches are directly scheduled continuously).

My personal understanding is that this restriction is mainly based on the following two reasons:

1. Check Finish Condition: After each iteration (iter) is completed, Check Finish Condition needs to be performed on each request in the current batch. If a request has been completed, its results need to be returned to the user immediately.
2. Update constrained sampling: After each iteration is completed, vocab_mask also needs to be updated in order to apply constraints in the next round of sampling (more details below).In addition, after the Sample phase is completed and before entering the Post-schedule phase, there is an Event to force synchronization. You must wait for the Token generated by Sample to be transferred to the CPU before Post-schedule scheduling can be performed. This synchronization mechanism ensures data integrity and scheduling correctness.

The pipeline in [SGLang Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#zero-overhead-batch-scheduler) is drawn as shown below, considering the CPU as a whole. I feel that the pipeline diagram on the SGL blog is easier to understand the idea, and the pipeline diagram in this article is easier to understand the code details (you will see why the CPU-L is disassembled in the next section).

![](image/sgl_blog_pipeline.png)

## Overlap pseudocode

The pseudocode of the Overlap version of the event loop provided by SGLang is as follows. The run_batch function implements the functions of compute and sample. If you don’t understand this process, you can first read the [article without overlap version](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-scheduler/readme-CN.md) for explanation:```python
last_batch = None
while True:
    # pre schedule
    recv_reqs = recv_requests()
    process_input_requests(recv_reqs)
    batch = get_next_batch_to_run()
    # compute and sample
    result = run_batch(batch) # This is an asynchronous function, directly returns
    result_queue.put((batch, result))
    # post schedule
    if last_batch is not None:
        tmp_batch, tmp_result = result_queue.get()
        process_batch_result(tmp_batch, tmp_result)
    last_batch = batch
```
> This pseudocode does not talk about the overlap of constrained decoding. Together with the **Overlap constraint decoding** in the last section of this article, I want to complete the event_loop_overlap function of SGLang.

#Core code

## TpModelWorkerClient

When we enable overlap, sglang will change Scheduler's TpWorkerClass from TpModelWorker to TpModelWorkerClient, which means we are embarking on the path of **asynchronous**.

`TpModelWorkerClient` This class corresponds to the CPU-L entity of the pipeline, which is responsible for asynchronous Launch Kernel and processing the Logit results returned by the Kernel (i.e. Sample stage). After the Scheduler passes the scheduled batch to `TpModelWorkerClient.forward_batch_generation()`, it does not need to care about the startup of the Kernel or the processing of the return value, but continues to execute the next step, thereby achieving efficient pipelining.

In TpModelWorkerClient, batches are passed between pipeline stages through `input_queue` and `output_queue`:

- `input_queue`: The batch is passed from the run_batch phase of CPU-S to the launch_compute phase of CPU-L through this queue.
- `output_queue`: The return value of the batch is passed from the sample stage to the process batch result stage of CPU-S through this queue.

It should be noted that there is only one task in these two queues at the same time.

### forward_batch_generation

The run_batch function of schedule will call `TpModelWorkerClient.forward_batch_generation()` to forward the newly scheduled batch asynchronously to the GPU through CPU-L.

But it stands to reason that run_batch will definitely return a token list with completed inference as the output of the next inference (the autoregressive feature of LLM), but now I have to return it without inference. What should I do with this token list?

`TpModelWorkerClient` uses the following three data structures to construct a set of "**token list placeholders**" to stage a "civet cat for prince".1. future_token_ids_map: A less thorough **ring buffer** whose size is `5*max_running_requests`. We use the negative value of the serial number of the buffer to construct a placeholder and return it first. The actual inferred token will be asynchronously placed in the buffer. A replacement will be performed before the next inference to replace the placeholder with the actual token in the buffer.
2. future_token_ids_limit: buffer water level, size is `3*max_running_requests`.
3. future_token_ids_ct: This is the tail pointer of the current buffer. The next request will allocate the token position starting from the position of ct.

The process is as follows:

- If ct+batch_size is less than limit, we continue to linearly allocate **negative idx** as placeholders in the map buffer;
- If ct+batch_size is greater than limit, ct will be modulo and wrapped around the beginning of the map buffer, but the placeholder will exceed limit.

In the figure below, when ①, `max_running_requests=3`, `future_token_ids_limit=9`, `future_token_ids_ct initial=7`, when ②, there are three requests in the newly arrived batch, we assign new_future_next_token_ids to three positions [-8, -9, -10](each req and one iter infer a token, so allocate 3 token positions), that is, the yellow area, but ct wraps around the beginning of the buffer and continues to allocate from the position of idx=1.

![image-20250320212534333](image/placeholder1.png)

> There is something strange here. Obviously, new_future_next_token_ids can be directly allocated to [-8, 0, 1], so there is no need to overflow the buffer. And it stands to reason that a ring buffer of 2*`max_running_requests` is enough. Why set it to 5max and allow overflow after the ring buffer? And using two buffers with a size of `max_running_requests` as a double buffer cycle can already meet the requirements. I don't know what considerations were used to design this data structure and method to save tokens in asynchronous reasoning.

After allocating the new_future_next_token_ids placeholder, we return it to the scheduler. The scheduler first takes this "placeholder" for subsequent operations.Okay, now let's review this function in general

1. When the batch comes, pass the tuple `(input_batch, input_batch_start_ct)` to the forward_thread_func_ function through input_queue, and forward_thread_func_ performs asynchronous forwarding.
2. Based on the current future_token_ids_ct and batch_size, construct the new_future_next_token_ids placeholder, and return this set of token placeholders to the scheduler.

### forward_thread_func_

forward_batch_generation is used to perform inference asynchronously. Its core code is as follows```
input_ids = model_worker_batch.input_ids
resolve_future_token_ids(input_ids, _self_.future_token_ids_map)

_# run forward_
logits_output, next_token_ids = _self_.worker.forward_batch_generation(
    model_worker_batch, _self_.launch_done
)

_# Update the future token ids map_
bs = len(model_worker_batch.seq_lens)
_self_.future_token_ids_map[
    future_token_ids_ct + 1 : future_token_ids_ct + bs + 1
] = next_token_ids
```
#### resolve_future_token_ids

Assume that the current batch is batch2. Before batch2 starts decoding, its input_token is the placeholder returned by forward_batch_generation when batch1 is decoding. We need to replace the placeholder before batch2 forward, that is, the resolve_future_token_ids function.

> This is a bit confusing, you can take a look at the above code

The resolve_future_token_ids function takes as input `(placeholder,` _future_token_ids_map_ `)`, and then converts the placeholder into the corresponding token in `future_token_ids_map`. Actually it is

$$
\text{real\_token} = \text{future\_token\_ids\_map[-placeholder]}
$$

In the picture below, assuming our placeholder is [-8, -9, -10], we will replace it with the content of idx=8, 9, 10 in the map. The replaced input_token is [99, 76, 556]

![image-20250320212615309](image/placeholder2.png)

#### worker.forward_batch_generation

This function is a synchronous blocking function. We put the batch on the GPU to run, and then perform a Sample on the GPU after running.

#### Process newly generated tokens

The `worker.forward_batch_generation` function will return the newly generated token after sampling, that is, next_token_ids. We need to put next_token_ids into the corresponding position of future_token_ids_map, so that the next batch can hold the placeholder and exchange it for a real token before inference.

> There is actually no need for placeholder placeholders and ring buffers. We only need to use two buffers for double buffering, and assign the token of the previous buffer to input_ids before each docode.

### resolve_batch_result

The above functions can logically form a pipeline. But TpModelWorkerClient also has a resolve_batch_result function and output queue.This is because in forward_thread_func, the tensors of token, logprobs and hidden_states after the sample are all on the GPU, and these tensors need to be used on the CPU in the process_batch_result of the scheduler.

We need to copy these tensors to the CPU, so we registered a copy done cuda event in forward_thread_func. The post scheduler stage of the pipeline, that is, the process_batch_result function will call the resolve_batch_result function to wait for the copy done event to complete and make a dynamic removal request.

# Overlap constraint decoding

When we look at the code, we will see two additional details:

1. In `event_loop_overlap`, there is an empty batch with ForwardMode of `DUMMY_FIRST`
2. In the function link `TpModelWorker.forward_batch_generation() -> ModelRunner.sample() -> _preprocess_logits()`, there is a `sampling_info_done` event, and the comment says "_the function update_regex_vocab_mask was executed in process_batch_result of the last batch._"

Strange, the pipeline just now has obviously been established, why is there another pipeline that overlaps update_regex_vocab_mask? What function is this?

The story goes back to SGLang's constraint decoding. SGLang's constraint decoding ensures that the output of the model must conform to the given grammar format. When generating the next token, the model's logits (unnormalized probability) will be combined with `vocab_mask` to set the logits of invalid tokens to a very small value (such as `-inf`), so that they are excluded during sampling.

For example, if I want the output to conform to the format of grammar=json, then the first token of the output json format must be a `"{"`. I need to call the `apply_vocab_mask` function before sampling to set the softmax probability of other token positions to a small value to ensure that the first token sampled by the model must be `"{"`.The process of updating the mask is performed in the `update_regex_vocab_mask` function. Because it involves syntax parsing logic, it is a CPU-intensive operation. **As for applying mask to logit, it just needs to be processed before the sample of the `forward_batch_generation()` function.

After adding constraint decoding, our reasoning process becomes```python
··· -> forward -> constrained decoding(update_regex_vocab_mask) -> sample -> ···
```
If we do not do the overlap of `update_regex_vocab_mask`, then our decoding will be blocked by `update_regex_vocab_mask`, causing the GPU to get stuck before sample.

And if we have batch1 and batch2, we can update the `regex_vocab_mask` of batch2 in advance in the process_batch_result of batch1 before batch2 sample. In this way, we overlap this CPU-intensive operation after the GPU. The code is as follows:```python
# in process_batch_result_prefill and process_batch_result_decode
if batch.next_batch_sampling_info: # Note that this is next_batch! ! ! _
_ batch.next_batch_sampling_info.update_regex_vocab_mask()_
_ self.current_stream.synchronize()_
_ batch.next_batch_sampling_info.sampling_info_done.set()
```For the first batch, we need to calculate regex_vocab_mask before it runs, so a dummy_batch is needed to trigger update_regex_vocab_mask. This is why a dummy_batch must be constructed before the actual first batch is run (actually this batch is not run on the GPU, but is just used to trigger the update_regex_vocab_mask function call link).

![image-20250320212649949](image/real_pipeline2.png)

The red box in the above figure represents the pipeline after adding constraint decoding. The red dotted line represents that we wait for the sampling_info_done event to ensure that the mask must be updated before the sample.