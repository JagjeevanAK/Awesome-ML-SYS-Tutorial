# slime Rollout system detailed explanation

## Overview

The `slime/rollout` module is the core component in the slime framework and is responsible for handling sample generation, filtering and evaluation during the reinforcement learning training process. This module provides a complete pipeline to obtain hints from data sources, generate responses, apply reward models, and select high-quality samples for training through filters.

## System architecture diagram analysis
![slime overall workflow](overall_workflow.jpg)
### Training cycle process
```
train.py → ray/rollout.py → RolloutManager → RolloutController → Data generation → Model training
```
### SGLang distributed generation
```
Router → SGLang Server 1/2 → TP0/TP1/TP2/TP3 → Sample generation → Reward evaluation
```
## Module structure
```sh
slime/rollout/
├── __init__.py
├── sglang_rollout.py # Asynchronous sample generation based on SGLang
├── sft_rollout.py # SFT training sample processing
├── filter_hub/ # Sample filter
│ ├── dynamic_sampling_filters.py
│ └── over_sampling_filters.py
└── rm_hub/ # Reward model collection
    ├── __init__.py
    ├── deepscaler.py
    ├── f1.py
    ├── math_utils.py
    └── math_dapo_utils.py
```
## Detailed explanation of core components

### SGLang Rollout (`sglang_rollout.py`)

This is the main sample generation engine for efficient asynchronous text generation based on SGLang.

**Key Features:**
- **Asynchronous generation**: Use `asyncio` to achieve concurrent sample generation
- **State Management**: `GenerateState` singleton class manages global generation state
- **Interruptible Build**: Supports interrupting and resuming during the build process
- **Batch Processing**: Supports batch generation and reward model evaluation

**Core classes and functions:**



`GenerateState` is the global generation state manager.
- Manage the generation status of `Group: List[Sample]`
- Control submission of `generate_and_rm_group` tasks
- Maintain `semaphore`, `sampling_params`, `args`, etc.
<details>
<summary>GenerateState class</summary>

```python
class GenerateState(metaclass=SingletonMeta):
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.semaphore = asyncio.Semaphore(args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine)
        self.sampling_params = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )
        self.reset()

    def reset(self):
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]):
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)
```
</details>


`generate_rollout_async` This is the main function for asynchronous sample generation.
workflow:
* Step1. Define relevant filters, e.g. dynamic_filter&over_sampling_filter
* define target_data_size=
  1. If over_sampling_filter is not enabled, then target_data_size=rollout_batch_size
  2. If over_sampling_filter is turned on, then target_data_size=over_sampling_batch_size
* Step2. Determine the data size and take out the samples of over_sampling_batch_size from the dataset
* Step3. Wait until the first group in the batch of Step 2 ends, and sample the completed part (at the same time perform dynmaic filter)
* Step4. If the overall number of samples is not enough (the number of effective groups that have been obtained + the number of remaining groups being rolled out <target_data_size), repeat Step2&3
* Step5. Abort the job that is still pending. If it is a partial rollout, it will be recycled to the buffer.
* Step6. If over_sampling_filter is used, filter



<details>
<summary>generate_rollout_async function</summary>

```python
async def generate_rollout_async(args, rollout_id: int, data_source) -> list[list[Sample]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        list[list[Sample]]: a list of samples generated by the rollout, the length of the list is exactly the same as the `rollout_batch_size`
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    over_sampling_filter = (
        load_function(args.over_sampling_filter_path) if args.over_sampling_filter_path is not None else None
    )

    # target_data_size is the total number of valid samples to get
    target_data_size = args.over_sampling_batch_size if over_sampling_filter is not None else args.rollout_batch_size

    data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                print(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            if dynamic_filter is not None and not dynamic_filter(args, group):
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    print(
        f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, label: {data[-1][0].label}, reward: {data[-1][0].reward}",
        flush=True,
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    if over_sampling_filter is not None:
        data = over_sampling_filter(args, data)[: args.rollout_batch_size]

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0].index)

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    return data, aborted_samples
```
</details>

`generate_and_rm_group` performs generation and reward model evaluation on sample groups.
- Process each sample in `Group: List[Sample]`
- Perform `generate_and_rm` operation on each sample

<details>
<summary>generate_and_rm_group function</summary>

```python
async def generate_and_rm_group(args, group: list[Sample], sampling_params: dict, evaluation=False) -> list[Sample]:
    """Generation and reward model evaluation on sample groups"""
    state = GenerateState(args)

    if state.aborted:
        return group

    # Generate all samples concurrently
    group = await asyncio.gather(
        *[generate_and_rm(args, sample, sampling_params.copy(), evaluation=evaluation) for sample in group]
    )

    # For reward models that require the entire group, evaluate here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards):
            sample.reward = reward

    return group
```
</details>

`generate_and_rm` Single sample generation and reward model evaluation.
- Process single samples such as `sample1`, `sample2` etc.
- Perform generation and reward evaluation

<details>
<summary>generate_and_rm function</summary>

```python
async def generate_and_rm(args, sample: Sample, sampling_params: dict, evaluation=False) -> Sample:
    """Single sample generation and reward model evaluation"""
    # For samples that have responded, check whether they are completed
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        if args.custom_generate_function_path is not None:
            custom_generate_func = load_function(args.custom_generate_function_path)
            sample = await custom_generate_func(args, sample, sampling_params)
        else:
            sample = await generate(args, sample, sampling_params)

    if sample.status == Sample.Status.ABORTED:
        return sample

    # For reward models that require the entire group, they are not evaluated here
    if args.group_rm:
        return sample

    # Evaluation rewards
    sample.reward = await async_rm(args, sample)
    return sample
```
</details>

`abort` aborts the build process and collects partially completed samples.
- post abort_all to sglang_router
- If partial_rollout, put `aborted_samples` into data buffer
<details>
<summary>abort function</summary>

```python
async def abort(args, rollout_id: int):
    """Interrupt the build process"""
    aborted_samples = []
    state = GenerateState(args)
    state.aborted = True
    
    # Interrupt all requests
    response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
    for url in response["urls"]:
        await post(f"{url}/abort_request", {"abort_all": True})

    # Collect partially completed samples
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group = task.result()
            aborted_samples.append(group)

    return aborted_samples
```
</details>

**Detailed explanation of the generation process:**

1. **Initialization**: Set generation parameters and concurrency control
2. **Data acquisition**: Obtain prompt samples from the data source
3. **Task Submission**: Submit the generation task to the SGLang server
4. **Dynamic Filtering**: Apply dynamic sampling filter
5. **Oversampling Filter**: Apply an oversampling filter to select the final sample
6. **Cleanup**: Interrupt unfinished tasks and collect results

**filter logic**
In the system architecture diagram, the rollout part shows the logic of not enabling the filter. If the filter is enabled, the specific rollout flow is:

* The system first starts over_sampling_batch_size=6 concurrent generate_and_rm_group tasks. target_data_size=over_sampling_batch_size=6
* When a group is completed, the reward standard deviation will be checked through Dynamic filter (groups with Std=0 are discarded).
* Since target_data_size=6 valid groups are required, when it is detected that the number of valid groups that have been obtained + the number of groups being rolled out <target_data_size, a new batch will be submitted to obtain enough samples.
* After finally collecting target_data_size=6 valid groups, interrupt the unfinished tasks through the finish & abort operation, and then apply the Over Sampling filter to sort the 6 completed sample groups by reward standard deviation, and select the 4 highest quality ones as the final Completed Samples.

As follows:
![slime sampling flow](sampling_flow.jpg)

### SFT Rollout (`sft_rollout.py`)

A dedicated sample processing module for supervised fine-tuning (SFT).

**Core features:**
- **Word Segmentation**: Use tokenizer to segment samples
- **Loss Mask Generation**: Generate loss masks for training
- **Response length calculation**: Calculate the length of the response part

**Implementation example:**

<details>

```python
def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    # Get samples
    samples = data_buffer.get_samples(args.rollout_batch_size)
    
    for sample in samples:
        # Generate loss mask
        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages)
        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
        
        #Set sample properties
        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]
    
    return samples
```
</details>

### Filter system (`filter_hub/`)

The filter system is reflected in the architecture diagram as a dynamic filtering and oversampling filtering mechanism to ensure sample quality.

**Dynamic Sampling Filters (`dynamic_sampling_filters.py`)**

<details>
<summary>Detailed explanation of dynamic sampling filter</summary>

**Function**: Filter out sample groups with a reward standard deviation of 0 (delete all 0/1 sample groups)
```python
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    """
    Check whether the reward standard deviation of the sample group is greater than 0
    
    Args:
        args: global parameters
        samples: sample list
        **kwargs: additional parameters
    
    Returns:
        bool: Returns True if the standard deviation is greater than 0, otherwise returns False
    """
    rewards = [sample.get_reward_value(args) for sample in samples]
    return torch.tensor(rewards, dtype=torch.float).std() > 0.0
```

**Function**:
- Ensure that the sample group selected has sufficient diversity
- Avoid selecting groups where all sample rewards are the same
- Improve the quality of training data

**Role in system architecture diagram**:
- applied in real time during the build process
- Filter out unqualified sample groups
- Ensure sample groups have bonus diversity
</details>

**Oversampling Filters (`over_sampling_filters.py`)**

<details>
<summary>Detailed explanation of oversampling filter</summary>

**Function**: Sort sample groups by reward standard deviation, giving priority to sample groups with large variances

```python
def sort_by_reward_std(args, samples: list[list[Sample]], **kwargs) -> list[list[Sample]]:
    """
    Sort sample groups by reward standard deviation
    
    Args:
        args: global parameters
        samples: list of sample groups
        **kwargs: additional parameters
    
    Returns:
        list[list[Sample]]: group of samples sorted by standard deviation in descending order
    """
    samples_with_std = []
    for group in samples:
        rewards = [item.reward for item in group]
        std = torch.tensor(rewards, dtype=torch.float).std()
        samples_with_std.append((group, std))
    
    # Sort by standard deviation in descending order (python sort is stable)
    samples_with_std.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in samples_with_std]
```
**Function**:
- Prioritize sample groups with more diverse reward distributions
- These sample groups usually contain more valuable training signals
- Improve the efficiency of model learning

**Role in system architecture diagram**:
-Apply after all candidate samples are generated
- Select the optimal subset from candidate samples
- Ensure that the final samples have high-quality training signals
</details>

### Reward model collection (`rm_hub/`)

The reward model set is reflected in the architecture diagram as an evaluation mechanism for generated samples, supporting multiple evaluation methods.

**Supported reward model types:**

1. **DeepScaler**: Rule-based reward model
2. **DAPO**: Mathematics Problem Assessment Model
3. **Math**: Mathematical answer verification model
4. **F1**: F1 score calculation model
5. **Remote RM**: Remote reward model interface

**Core functions:**

<details>
<summary>Detailed explanation of async_rm function</summary>

Evaluate a single sample based on the configured reward model type.

```python
async def async_rm(args, sample: Sample, **kwargs):
    """
    Asynchronously evaluate rewards for individual samples
    
    Args:
        args: global parameters
        sample: sample to be evaluated
        **kwargs: additional parameters
    
    Returns:
        float: reward value
    """
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    rm_type = args.rm_type
    response = sample.response
    label = sample.label
    
    # Handle special prefixes
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_"):]

    #Select reward model based on type
    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    elif rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "dapo":
        return compute_score_dapo(response, label)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    else:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
```

**Role in system architecture diagram**:
- called in `generate_and_rm` function
- Evaluate the reward value of a single sample
-Supports multiple assessment strategies
</details>

<details>
<summary>Detailed explanation of batched_async_rm function</summary>

Batch evaluation of rewards for multiple samples to improve evaluation efficiency.

```python
async def batched_async_rm(args, samples: list[Sample], **kwargs) -> list[Union[int, float]]:
    """
    Batch asynchronous evaluation of rewards for multiple samples
    
    Args:
        args: global parameters
        samples: sample list
        **kwargs: additional parameters
    
    Returns:
        list[Union[int, float]]: list of reward values
    """
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
```

**Role in system architecture diagram**:
- Called in `generate_and_rm_group` function
- Support group-level reward model evaluation
- Improve batch evaluation efficiency
</details>

## Detailed explanation of workflow

### Complete system workflow

According to the architecture diagram, the workflow of the entire system is as follows:

```
Training loop (train.py)
    ↓
RolloutManager (ray/rollout.py)
    ↓
RolloutController (ray/buffer.py)
    ↓
RolloutDataSourceWithBuffer (ray/rollout_data_source.py)
    ↓
generate_rollout (rollout/sglang_rollout.py)
    ↓
SGLang Router → SGLang Servers → TP Ranks
    ↓
Sample generation → reward evaluation → filter selection
    ↓
completed_samples → training loop
```

### Detailed process of training cycle

<details>
<summary>Detailed explanation of training cycle steps</summary>

1. **rollout samples**:
   - Call `RolloutManager.async_generate(rollout_id)`
   - Trigger `RolloutController.generate(rollout_id)`
   - Execute sample generation process

2. **offload sglang**:
   - Call `RolloutManager.async_offload()`
   - Release SGLang related memory to make space for training

3. **model training**:
   - Use generated samples for model training
   - Update model parameters

4. **offload megatron**:
   - Release Megatron related memory
   - Prepare for SGLang recovery

5. **resume sglang weight**:
   - Call `RolloutManager.async_onload()`
   - Restore SGLang weights

6. **weight sync**:
   - Synchronize model weights
   - Ensure that the status of each component is consistent

7. **resume sglang kv cache**:
   - Restore SGLang KV cache
   - Prepare for the next round of generation

8. **Back to rollout samples**:
   - Start the next round of sample generation
   - Form a complete training cycle
</details>

### SGLang distributed generation process

<details>
<summary>Detailed explanation of SGLang generation process</summary>

1. **Router routing**:
   - Central Router receives generation request
   - Distributed to different SGLang Servers according to load balancing strategy

2. **SGLang Server processing**:
   - Each SGLang Server handles assigned requests
   -Support multiple servers for parallel processing

3. **Tensor Parallelism (TP) parallel**:
   - Each Server uses TP0-TP3 internally for tensor parallelism
   - Improve the inference efficiency of large models

4. **Sample Generation and Evaluation**:
   - Execute `generate_and_rm_group` operation
   - Generate samples and perform reward evaluation
   - Supports "start" and "abort" control points

5. **Result returned**:
   - Completed samples are returned as `completed_samples`
   - Aborted samples are returned as `aborted_samples`
</details>

### Data flow and buffering mechanism

<details>
<summary>Detailed explanation of data flow</summary>

1. **Data acquisition process**:
```
   RolloutDataSourceWithBuffer.get_samples()
   ├── First try to get the sample from buffer
   ├── If the buffer is not enough, call the parent class get_samples()
   └── Return enough sample groups
```
2. **Sample generation process**:
```
   generate_rollout_async()
   ├── Submit the generation task to SGLang
   ├── Wait for the generation to complete
   ├── Apply dynamic filters
   └── Apply oversampling filter
```
3. **Result processing process**:
```
   Generate results
   ├── completed_samples → return to training loop
   └── aborted_samples → added to RolloutDataSourceWithBuffer.buffer
```
4. **Buffer Management**:
```
   RolloutDataSourceWithBuffer
   ├── buffer: stores interrupted samples
   ├── add_samples(): Add samples to the buffer
   └── get_samples(): Get samples from the buffer
```
</details>

### Detailed explanation of configuration parameters

<details>
<summary>Key configuration parameters</summary>

| Parameters | Description | Reflection in the architecture diagram | Impact |
|------|------|----------------|------|
| `rollout_batch_size` | Number of samples generated in each batch | Number of samples finally returned | Control generation efficiency |
| `over_sampling_batch_size` | Oversampling batch size | Number of samples during generation | Control sample selection range |
| `n_samples_per_prompt` | Number of samples generated per prompt | Number of samples in Group | Control diversity |
| `dynamic_sampling_filter_path` | Dynamic filter path | Dynamic filtering mechanism | Real-time filtering of unqualified samples |
| `over_sampling_filter_path` | Oversampling filter path | Oversampling filtering mechanism | Select the optimal sample subset |
| `rollout_num_gpus_per_engine` | Number of GPUs per engine | SGLang Server configuration | Controlling parallelism |
| `rollout_num_gpus` | Total number of GPUs | System scale | Impact on overall performance |
| `sglang_server_concurrency` | SGLang server concurrency number | Concurrency control | Affects generation speed |
</details>
