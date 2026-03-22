# verl sglang multi-turn over sample

## Quick reproduction

1. Create a new docker (you can skip this if you are familiar with this installation):

You need to configure `WANDB_API_KEY` before use, refer to [this process](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914).

```bash
# If your system has not been configured with HF_TOKEN and WANDB_API_KEY, please configure it first
docker run -it --name h100_verl_{your_name} --gpus all \
    --shm-size 32g \
    -v {your_cache_path}:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```

After entering docker, you can view the mapped environment variables:

```bash
echo $HF_TOKEN
echo $WANDB_API_KEY
```
In the future, every time you exit from docker, you can use this command to restart:

```bash
docker start -i h100_verl_{your_name}
```
2. Install verl-sglang based on source code

Configure python environment:

```bash
mkdir -p /tmp
chmod 1777 /tmp
apt update
apt install -y python3.10 python3.10-venv
python3 -m ensurepip --upgrade
python3 -m venv ~/.python/verl-sglang
source ~/.python/verl-sglang/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade uv
```
3. Install verl-sglang:

```bash
cd ~
git clone -b over_sample https://github.com/zhaochenyang20/verl.git
cd verl

python -m uv pip install wheel setuptools
python3 -m uv pip install -e ".[sglang]" --prerelease=allow
python3 -m uv pip install -r ./requirements.txt --no-build-isolation
python3 -m uv pip install torch_memory_saver
```
4. Test gsm8k:

```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Pull and preprocess the gsm8k data set
python examples/data_preprocess/gsm8k_multiturn_w_tool.py

# Start 8-card training
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

5. Test dapo:
```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash examples/sglang_multiturn/run_qwen3_4b_dapo_multiturn.sh
```

## Design ideas and specific implementation

Based on this commit: [b979a73e358313afafab5db512cd5ae0009ccac0](https://github.com/zhaochenyang20/verl/tree/b979a73e358313afafab5db512cd5ae0009ccac0)

The design idea has been discussed many times. In order to solve the long tail problem, using over sample is a very common strategy. Compared with partial rollout, the strategy designed here is rougher. Uncompleted reqs will be discarded directly.

Specifically, it is implemented through three functions: `monitor_and_cancel`, `process_request_with_monitoring` and `run_with_cancellation`. `monitor_and_cancel` is responsible for monitoring the completion number, taking immediate action once the target is reached, canceling the remaining tasks, and sending an abort signal to the engine. `process_request_with_monitoring` is responsible for processing a single request and returning actual results or padding data based on completion. `run_with_cancellation` starts both `monitor_and_cancel` and `process_request_with_monitoring`.

- `process_request_with_monitoring`

```python
async def process_request_with_monitoring(req):
    nonlocal completed_count
    try:
        result = await self._async_rollout_a_request(req, do_sample, is_validate, **kwargs)

        async with completion_lock:
            if completed_count < target_completion:
                completed_count += 1
                print(f"✅ Request {req.request_id} completed ({completed_count}/{total_requests})")
                return result # Return the real result
            else:
                # If the target is exceeded, return padding
                logger.info(f"Request {req.request_id} finished after target met, creating padding")
                return self._create_padding_request(req)
```

1. Each request will independently start its own `process_request_with_monitoring` and execute `_async_rollout_a_request` in a blocking manner through `await`.
2. For those requests that were completed earlier, result gets the real result and the `completed_count` counter is incremented. Note that `completed_count` here is a global variable, and `completion_lock` needs to be used to ensure the atomicity of the counting operation and no conflict between reading and writing.
3. For those requests that are completed later, `monitor_and_cancel` detects that `completed_count` reaches `target_completion`, will cancel these tasks, and send `abort_requests` requests to the sglang engine.

- `monitor_and_cancel`

```python
async def monitor_and_cancel():
    nonlocal completed_count
    while completed_count < target_completion:
        await asyncio.sleep(0.1) # Check every 0.1 seconds

    print(f"🎯 Target reached: {completed_count}/{total_requests} completed!")
    print("🚫 Canceling remaining requests and sending abort to engine...")

    # Cancel remaining tasks
    canceled_count = 0
    for task in all_tasks:
        if not task.done():
            task.cancel()
            canceled_count += 1

    #Send abort signal to engine
    try:
        abort_result = await self._engine.abort_request(abort_all=True)
        print(f"✅ Abort signal sent to engine: {abort_result}")
    except Exception as e:
        print(f"❌ Failed to send abort signal to engine: {e}")
```

Continuously monitor the completion number, and once the target is reached, take immediate action, cancel the remaining tasks, and send the `abort_requests` signal to the sglang engine. Note that the engine abort here is actually written in the `AsyncEngine` class in `sglang_rollout.py`:


```python
    async def abort_request(self, rid: str = "", abort_all: bool = False):
        """Abort a specific request or all requests.

        Args:
            rid: The request ID to abort. If empty and abort_all is False, no action is taken.
            abort_all: If True, abort all running requests regardless of rid.
        """
        try:
            result = self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)
            print(f"🔍 Abort result: {result}")
            return result if result is not None else {"status": "aborted"}
        except Exception as e:
            logger.error(f"Failed to abort requests: {e}")
            raise
```

Here are a few points worth pondering:

1. In fact, verl's `AsyncEngine` inherits and overrides many methods of sglang Engine, such as `update_weights_from_tensor` and `resume_memory_occupation`. Logically speaking, sglang Engine does not implement these methods and does not affect verl, and of course it affects other frameworks. At first I thought I had to implement `abort_request` for Engine in sglang, because at first only server had it but engine did not. But considering that `AsyncEngine` rewrites `abort_request`, sglang Engine does not actually need to implement this function, and we do not need to release a version for this. After all, updating SGLang version on verl is really too painful.
2. [Now we just abort the engine, what is the impact of whether the tool aborts?]
3. [Unlike `update_weights_from_tensor`, `abort_request` cannot call `self.tokenizer_manager.abort_request` internally through await, it must be called directly. Thanks to jiajun and yuzhen for the reminder. Here you have to check the internal implementation of sglang tokenizer_manager. If a function is implemented asynchronously in tokenizer_manager, then external calls can only have await syntax. Frankly speaking, this is because I am not familiar with asynchronous syntax, and I do not understand why `resume_memory_occupation` and `abort_request` in tokenizer_manager, the former is asynchronous and the latter is synchronous. In addition, if we want to write only one line of await for an asynchronous function in an asynchronous function, does this mean that we are actually waiting for the execution of the internal asynchronous function to complete? What’s the point of writing this way? Specifically:]

```python
# sglang_rollout.py

    async def resume_memory_occupation(self, tags: Optional[list[str]] = None):
        """Resume GPU occupation."""
        # because __init__ is a sync method, it can not call the async release_memory_occupation
        # have to move release_memory_occupation from __init__ to here
        # For multi-stage awake, we run release weight and kv_cache when we resume weights for the first time.
        if self._need_reload:
            await self.release_memory_occupation()
            self._need_reload = False

        if tags is None:
            obj = ResumeMemoryOccupationReqInput()
        else:
            obj = ResumeMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.resume_memory_occupation(obj, None)
```

The inner `self.tokenizer_manager.resume_memory_occupation` is an asynchronous function, so the outer `resume_memory_occupation` function is waiting for the completion of the inner layer. If so, why can't the outer `resume_memory_occupation` function be synchronous? Currently, the outer function is asynchronous, so calling the outer function and waiting requires await. For example:

```python
# fsdp_sglang.py

async def release_memory(self):
    if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
        if self.multi_stage_wake_up:
            await self.inference_engine.release_memory_occupation(tags=["kv_cache", "weights"])
        else:
            await self.inference_engine.release_memory_occupation()
        log_gpu_memory_usage("After release memory occupation in sharding manager", logger=logger)
```

- `run_with_cancellation`

```python
async def run_with_cancellation():
    nonlocal all_tasks

    #Create all tasks
    all_tasks = [asyncio.create_task(process_request_with_monitoring(req)) for req in req_list]

    # Start monitoring task
    monitor_task = asyncio.create_task(monitor_and_cancel())

    try:
        # Wait for all tasks to complete (including canceled ones)
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Process the result and convert the exception into padding
        output_req_list = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert exceptions to padding
                logger.warning(f"Task {i} resulted in exception: {result}")
                output_req_list.append(self._create_padding_request(req_list[i]))
            else:
                output_req_list.append(result)

        return output_req_list
    finally:
        # Clean up monitoring tasks
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
```

With the understanding of the first two, the final `run_with_cancellation` is very clear. Note that `all_tasks` and the read-write lock `completion_lock` are global variables of these three functions. Here, `process_request_with_monitoring` of all reqs is started at the same time and `monitor_task` is created to monitor. Note that although `_async_rollout_a_request` of each req may not be completed, the `process_request_with_monitoring` of this function will definitely end, so `results = await asyncio.gather(*all_tasks, return_exceptions=True)` will definitely return and then be processed one by one. `results`, there are three situations at this time: `COMPLETED`, `Exception` and `PADDING`. After converting `Exception` to `PADDING`, return `output_req_list`.

After reading it as a whole, I think the design is quite clear, but the implementation may not be good and needs major changes.

Here are a few places I think must be checked:

1. [TODO: Is the implementation of directly typing padding correct? At least there must be no loss. In my ideal design, this reqs is discarded, so that the group size of GRPO is reduced, the request does not exist, and some training time is saved. You have to check carefully here to see if there is anything else to modify besides setting response_loss_mask to 0. I initially modified the `agg_loss` function, but then I asked claude that it might not be necessary and requires additional confirmation. In addition, I am not sure whether the reward function needs to be changed. For FSDP, it is this function `def _expand_to_token_level`. Finally, if we implement perfect discarding and the number of requests of different GRPO groups is inconsistent, it stands to reason that this will affect the variance of the GRPO group and may make the training more unstable. Is it possible to characterize this impact?] There is another design, which is directly marked as padding. In fact, the trajs obtained by partial rollout are directly lost. You can consider retaining these tracjs, but the loss mask is 0, and the reward is also 0. It may be better to listen to Teacher Long.
2. In the asynchronous part, I mentioned this problem:

> Frankly speaking, this is because I am not familiar with asynchronous syntax, and I do not understand why `resume_memory_occupation` and `abort_request` in tokenizer_manager, the former is asynchronous and the latter is synchronous. In addition, if we want to write only one line of await for an asynchronous function in an asynchronous function, does this mean that we are actually waiting for the execution of the internal asynchronous function to complete? What’s the point of writing this way? Specifically:3. Can the entire implementation be made clearer so that the verl team can agree to this simple feature? Currently I need to change the compute_data_metrics function of sglang_rollout.py and metric_utils.py. A lot has been said about the former, but the latter is actually very tricky. I currently exclude requests that have been aborted from the denominator when calculating the average reward. There are several questions: [First, we need to confirm that during validation, there are no aborted requests. Theoretically, the validation step will not be affected by `aborted_mask = (response_length == 0).bool() `. This needs to be verified experimentally, and the validation step will not have aborted reqs.] In addition, if these aborted requests are recorded in the metric, the reward of this part is all 0, so compared to the baseline without over sample, the reward of over sample will be much lower. If we do not consider these aborted requests, in fact, these aborted requests are often more difficult turns, for example, and the reward for this part is lower than those that have not been aborted. Therefore, ignoring this part of the reward will lead to an inflated reward. At present, I have no way to avoid this false high, but I think the reward in the validation step is accurate and there is no serious problem at the moment. [We directly set loss mask = 0 to avoid padding reqs affecting loss. However, I asked GPT. It seems that it is only effective for a specific agg loss mode. We need to study it here.]

The overall performance is in line with expectations. The reward of the training step is artificially high, the reward of the validation step can be aligned with the baseline, and the rollout time has been significantly improved.