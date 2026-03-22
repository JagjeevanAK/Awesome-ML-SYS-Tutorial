#RolloutSystemWalkthrough

## File overview

The rollout system is the core component responsible for data generation in slime. It mainly consists of two files:
- `slime/ray/rollout.py`: `class RolloutManager` manages the life cycle of rollout engine and router;
- `slime/ray/buffer.py`: `class RolloutController` handles rollout data generation and conversion

![slime rollout workflow](rollout_parts.png)

## Detailed explanation of core components

### RolloutManager - Coordinator

**Function**
RolloutManager is the main controller of the rollout system and is responsible for coordinating the interaction between Router, Controller and Engines.

**Initialization process**
<details>
<summary>RolloutManager initialization</summary>

```python
class RolloutManager:
    def __init__(self, args, pg, wandb_run_id):
        self.args = args
        
        # 1. Start Router
        _start_router(args)
        
        # 2. Create Controller
        self.controller = RolloutController.options(
            num_cpus=1,
            num_gpus=0,
        ).remote(args, wandb_run_id=wandb_run_id)

        # 3. Create Engine pool
        self.all_rollout_engines = create_rollout_engines(args, pg)
        
        # 4. Multi-node configuration: only send requests to node-0 of each engine
        nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // args.rollout_num_gpus_per_node)
        self.rollout_engines = self.all_rollout_engines[::nodes_per_engine]
        
        # 5. Create lock
        self.rollout_engine_lock = Lock.options(
            num_cpus=1,
            num_gpus=0,
        ).remote()
```
</details>

**Key Methods**

**A. Data Generation**
<details>
<summary>async_generate method</summary>

```python
def async_generate(self, rollout_id):
    return self.controller.generate.remote(rollout_id)
```
</details>

**B. Assessment**
<details>
<summary>async_eval method</summary>

```python
def async_eval(self, rollout_id):
    return self.controller.eval.remote(rollout_id)
```
</details>

**C. Memory management onload/offload**
<details>
<summary>onload/offload</summary>

```python
def async_offload(self):
    return [engine.release_memory_occupation.remote() for engine in self.rollout_engines]

def async_onload(self, tags: List[str] = None):
    return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.rollout_engines]
```
</details>

### create_rollout_engines - Engine creation

**Function**
Create an SGLang engine pool to be responsible for model inference services.

**Core Logic**
<details>
<summary>create_rollout_engines implementation</summary>

```python
def create_rollout_engines(args, pg):
    if args.debug_train_only:
        return []

    # Calculation engine configuration
    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.rollout_num_gpus_per_node)
    num_engines = args.rollout_num_gpus // num_gpu_per_engine

    #Create Ray Actor
    RolloutRayActor = ray.remote(SGLangEngine)
    
    rollout_engines = []
    for i in range(num_engines):
        num_gpus = 0.2 # Each engine uses 0.2 GPUs
        num_cpus = num_gpus

        # Set scheduling policy
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        #Create engine
        rollout_engines.append(
            RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={"env_vars": {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}},
            ).remote(args, rank=i)
        )

    # Port allocation and initialization
    # ... port allocation logic ...
    
    #Initialize all engines
    init_handles = [engine.init.remote(**ports) for engine, ports in zip(rollout_engines, addr_and_ports)]
    ray.get(init_handles)

    return rollout_engines
```
</details>

**Key Features**:
- **Resource Allocation**: Each engine uses 0.2 GPUs by default
- **Multi-node support**: Supports cross-node deployment
- **Port Management**: Automatically assign server ports, NCCL ports, etc.
- **Initialization synchronization**: Wait for all engine initialization to complete

### _start_router - Router startup

**Function**
Start the SGLang router to provide load balancing services.

**Implementation details**
<details>
<summary>_start_router implementation</summary>

```python
def _start_router(args):
    if args.sglang_router_ip is not None:
        return # Use external Router

    from sglang_router.launch_router import RouterArgs

    # Automatically assign IP and port
    args.sglang_router_ip = get_host_info()[1]
    args.sglang_router_port = find_available_port(random.randint(3000, 4000))

    # Configure Router parameters
    router_args = RouterArgs(
        host=args.sglang_router_ip,
        port=args.sglang_router_port,
        balance_abs_threshold=0,
    )

    # Set log level and timeout
    if hasattr(router_args, "log_level"):
        router_args.log_level = "warn"
    if hasattr(router_args, "request_timeout_secs"):
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    # Start Router process
    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True
    process.start()
    
    # Wait for startup to complete
    time.sleep(3)
    assert process.is_alive()
```
</details>

### RolloutController - Executor

**Function**
RolloutController is the core executor of the rollout system and is responsible for data generation, conversion and management.

**Initialization**
<details>
<summary>RolloutController initialization</summary>

```python
@ray.remote
class RolloutController:
    def __init__(self, args, wandb_run_id):
        self.args = args
        init_wandb_secondary(args, wandb_run_id)

        #Create data source
        self.data_source = RolloutDataSourceWithBuffer(args)

        # Dynamically load rollout function
        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        
        print(f"import {self.args.rollout_function_path} as generate_rollout function.")
        print(f"import {self.args.eval_function_path} as eval_generate_rollout function.")
```
</details>

**Key Features**:
- **Dynamic function loading**: Support custom rollout function
- **SFT support**: You can switch to SFT mode through `--rollout-function-path`
- **Data Source Management**: Use buffered data sources

**generate method - core generation process**

<details>
<summary>Generate method implementation</summary>

```python
def generate(self, rollout_id):
    self.rollout_id = rollout_id

    # 1. Debug mode: load data from disk
    if self.args.load_debug_rollout_data:
        data = torch.load(
            open(self.args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
        )["samples"]
        data = [Sample.from_dict(sample) for sample in data]
    else:
        # 2. Normal mode: call the rollout function to generate data
        data = self.generate_rollout(self.args, rollout_id, self.data_source, evaluation=False)
        
        # 3. Flatten data (if it is a nested list)
        if isinstance(data[0], list):
            data = sum(data, [])

    # 4. Optional: Save debugging data
    if (path_template := self.args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=self.rollout_id))
        print(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=self.rollout_id,
                samples=[sample.to_dict() for sample in data],
            ),
            path,
        )
    
    # 5. Convert to training data format
    data = self._convert_samples_to_train_data(data)
    
    # 6. Wrap and return
    return Box(ray.put(data))
```
</details>

**Generation process**:
1. **Storage rollout ID**: Record the current rollout ID
2. **Data acquisition**: Obtain data from the debug file or rollout function
3. **Data flattening**: Handling nested data structures
4. **Debug Save**: Optional save debug data
5. **Format conversion**: Convert to training data format
6. **Ray Storage**: Wrapping into Ray Object Storage

**eval method - evaluation process**

<details>
<summary>eval method implementation</summary>

```python
def eval(self, rollout_id):
    if self.args.debug_train_only:
        return # Debug mode does not generate evaluation data

    # Call the evaluation rollout function
    data = self.eval_generate_rollout(self.args, rollout_id, self.data_source, evaluation=True)
    
    # Record evaluation data
    log_eval_data(rollout_id, self.args, data)
```
</details>

### _convert_samples_to_train_data - Data conversion

**Function**
Convert the generated Sample object into the dictionary format required for training.

**Conversion logic**
<details>
<summary>_convert_samples_to_train_data implementation</summary>

```python
def _convert_samples_to_train_data(self, samples: Union[list[Sample], list[list[Sample]]]):
    """
    Convert inference generated samples to training data.
    """
    #Basic training data
    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "rewards": [sample.get_reward_value(self.args) for sample in samples],
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
    }

    # Process loss mask
    loss_masks = []
    for sample in samples:
        # If loss_mask is not provided, create a default one
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        
        # Verify loss_mask length
        assert(
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks

    # Process raw reward
    if samples[0].metadata and "raw_reward" in samples[0].metadata:
        train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

    # Process round_number (for rollout buffer)
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]
    
    return train_data
```
</details>

**Conversion content**:
- **tokens**: token sequence of prompt + response
- **response_lengths**: response token length
- **rewards**: reward value
- **truncated**: Flag whether to be truncated
- **sample_indices**: sample index
- **loss_masks**: Loss masks
- **raw_reward**: Raw reward (optional)
- **round_number**: round number (optional)

### log_eval_data - Evaluation log

**Function**
Log evaluation data to wandb and console.

<details>
<summary>log_eval_data implementation</summary>

```python
def log_eval_data(rollout_id, args, data):
    log_dict = {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)

    print(f"eval {rollout_id}: {log_dict}")
    
    if args.use_wandb:
        log_dict["eval/step"] = (
            rollout_id
            if not args.wandb_always_use_train_step
            else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
        )
        wandb.log(log_dict)
```
</details>

## Detailed explanation of component relationships

### **1. Overview of component relationships**

![slime rollout component relationship](rollout_parts.png)

The above figure shows the relationship between the components in the slime rollout system. The entire system adopts a layered architecture to achieve separation of responsibilities and efficient collaboration.

### **2. Data flow**

**A. Generate request flow**
The training process initiates a generation request, and after a complete process of Manager coordination, Controller execution, and Engine inference, the training data is finally returned.

**B. Management operation flow**
- **Memory Management**: Manager directly calls Engine's offload/onload method
- **State Management**: Controller manages the state saving and loading of data sources
- **Evaluation**: Controller calls the evaluation function and records logs

### **3. Interaction with Dataset**

Refer to [Dataset Walkthrough](./dataset_code_walkthrough.md), data source interaction process:

1. **Controller** has a `RolloutDataSourceWithBuffer` instance
2. **When generating**: Call `data_source.get_samples()` to obtain prompt samples
3. **Buffer Management**: Supports partial rollout and over-sampling data reuse
4. **State Persistence**: Supports recovery from training interruptions

## Custom Rollout support

### **1. Function path configuration**

```bash
# RL mode (default)
--rollout-function-path slime.rollout.sglang_rollout.generate_rollout

# SFT mode
--rollout-function-path slime.rollout.sft_rollout.generate_rollout

# Custom mode
--rollout-function-path path.to.custom.generate_rollout
```

### **2. Function signature requirements**

```python
def generate_rollout(args, rollout_id, data_source, evaluation=False) -> list[list[Sample]]:
    """
    Args:
        args: global parameters
        rollout_id: rollout identification
        data_source: data source
        evaluation: whether it is evaluation mode
    
    Returns:
        list[list[Sample]]: generated sample group
    """
    # Implement logic
    return samples
```

### **3. SFT mode features**

SFT mode is implemented through a custom rollout function:
- **Data Reading**: Read pre-generated samples from a file
- **Format Conversion**: Convert to training data format
- **Reuse Architecture**: Completely reuse the RL architecture and process

## Key configuration parameters

| Parameters | Description | Default value |
|------|------|--------|
| `rollout_num_gpus_per_engine` | Number of GPUs used by each engine | 0.2 |
| `rollout_num_gpus` | Total number of GPUs | - |
| `rollout_function_path` | rollout function path | slime.rollout.sglang_rollout.generate_rollout |
| `eval_function_path` | Evaluate function path | - |
| `sglang_router_ip` | Router IP address | None (automatically assigned) |
| `sglang_router_port` | Router port | None (automatically assigned) |

## Summary of design features

1. **Layered architecture**: Manager coordination, Controller execution, Engine reasoning
2. **Asynchronous Design**: All major operations are asynchronous
3. **Scalability**: Supports multi-engine load balancing and multi-node deployment
4. **Flexibility**: Supports custom rollout function and SFT mode
5. **Fault Tolerance**: Supports training interruption recovery and state persistence
6. **Resource Management**: Precise GPU allocation and memory management

This architecture makes the rollout system efficient and flexible, and can support various complex reinforcement learning and supervised learning training scenarios!