# Data Source with/without Buffer

## 1. Introduction
slime/ray/rollout_data_source.py is the data source management module of the rollout system, responsible for providing training data for the rollout engine. This file defines two core classes: RolloutDataSource (basic data source) and RolloutDataSourceWithBuffer (buffered data source).

![DataSource](./datasource.svg)

## 2. Core Class and Function
### RolloutDataSource Class

**Function**
The basic data source class is responsible for loading data from the original data set and supports global data set management and state persistence.

**Key Attributes**

```python
class RolloutDataSource:
    def __init__(self, args):
        self.args = args
        self.epoch_id = 0 # Current epoch ID
        self.sample_index = 0 # Global sample index
        self.sample_offset = 0 # Offset in the current epoch
        self.metadata = {} # Metadata storage
        self.dataset = None #Dataset object
```

**Initialization logic**
<details>
<summary>Initialization logic</summary>

```python
if args.rollout_global_dataset:
    #Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    
    #Create dataset
    self.dataset = Dataset(
        args.prompt_data, # Data file path
        tokenizer=tokenizer, # tokenizer
        max_length=args.rollout_max_prompt_len, # Maximum prompt length
        prompt_key=args.input_key, # prompt field name
        label_key=args.label_key, # label field name
        metadata_key=args.metadata_key, # Metadata field name
        tool_key=args.tool_key, # Tool field name
        apply_chat_template=args.apply_chat_template, # Whether to apply the chat template
        seed=args.rollout_seed, # Random seed
    )
    
    # If shuffle is needed, perform shuffle
    if self.args.rollout_shuffle:
        self.dataset.shuffle(self.epoch_id)
else:
    # Do not use the global dataset, dataset is None
    self.dataset = None
```
</details>

**Key Points**:
- Only load the real dataset if `rollout_global_dataset=True`
- Otherwise `dataset=None`, used for testing or special scenarios

**get_samples() method**

**Function**: Obtain the specified number of sample groups from the data set.

**Core logic**:
<details>
<summary>get_samples method</summary>

```python
def get_samples(self, num_samples):
    samples = []
    
    if self.dataset is not None:
        # Branch 1: Use real data set
        if self.sample_offset + num_samples <= len(self.dataset):
            # Case 1: There is still enough data in the current epoch
            prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
            self.sample_offset += num_samples
        else:
            # Situation 2: The current epoch data is insufficient and you need to enter the next epoch.
            prompt_samples = self.dataset.samples[self.sample_offset :] #Finish the remaining data of the current epoch
            num_samples -= len(prompt_samples)
            self.epoch_id += 1 # Enter the next epoch
            
            #Reshuffle the data set
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
            
            # Get the remaining required data from the new epoch
            prompt_samples += self.dataset.samples[:num_samples]
            self.sample_offset = num_samples
        
        #Create multiple samples for each prompt (n_samples_per_prompt)
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample) # Deep copy avoids modifying the original data
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            samples.append(group)
    else:
        # Branch 2: Create empty samples without using real data sets
        for _ in range(num_samples):
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = Sample(index=self.sample_index)
                self.sample_index += 1
                group.append(sample)
            samples.append(group)
    
    return samples
```
</details>

**Key Features**:
1. **Epoch Management**: Automatically handle epoch boundaries and support data set re-shuffle
2. **Multi-sample generation**: Each prompt generates `n_samples_per_prompt` samples
3. **Status Maintenance**: Maintain `sample_offset`, `epoch_id`, `sample_index`
4. **Data Integrity**: Use deep copies to avoid data pollution
5. **The format of the extracted samples is list[list[Sample]]**, where Sample is defined in slime/utils/types.py.
<details>
<summary>Sample class</summary>

```python
class Sample:
    """The sample generated"""

    index: Optional[int] = None
    # prompt
    prompt: Union[str, list[dict[str, str]]] = ""
    tokens: list[int] = field(default_factory=list)
    # response
    response: str = ""
    response_length: int = 0
    label: Optional[str] = None
    reward: Optional[Union[float, dict[str, Any]]] = None
    loss_mask: Optional[list[int]] = None

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        return value

    @staticmethod
    def from_dict(data: dict):
        data["status"] = Sample.Status(data["status"])
        return Sample(**data)

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]
```
</details>

**add_samples() method**

**Function**: Add samples to the data source (not supported by basic classes).

<details>
<summary>add_samples method</summary>

```python
def add_samples(self, samples: list[list[Sample]]):
    raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")
```
</details>

**Design principle**: The basic data source is read-only and does not support dynamic addition of data.

**save() method**

**Function**: Save the data source status to a file.

<details>
<summary>save method</summary>

```python
def save(self, rollout_id):
    if not self.args.rollout_global_dataset:
        return # No need to save when not using the global data set
    
    state_dict = {
        "sample_offset": self.sample_offset, # Offset in the current epoch
        "epoch_id": self.epoch_id, # Current epoch ID
        "sample_index": self.sample_index, # Global sample index
        "metadata": self.metadata, # Metadata
    }
    
    # Save to the specified path
    path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
```
</details>

**Purpose**: Support recovery after training interruption and ensure data sequence consistency.

**load() method**

**Function**: Load data source status from file.

<details>
<summary>load method</summary>

```python
def load(self, rollout_id=None):
    if not self.args.rollout_global_dataset:
        return # No need to load when the global data set is not used
    
    if self.args.load is None:
        return # No loading path specified
    
    path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
    if not os.path.exists(path):
        print(f"Checkpoint {path} does not exist.")
        return
    
    # Loading status
    state_dict = torch.load(path)
    self.sample_offset = state_dict.get("sample_offset", 0)
    self.epoch_id = state_dict.get("epoch_id", 0)
    self.sample_index = state_dict.get("sample_index", 0)
    self.metadata = state_dict.get("metadata", {})
    
    # Reshuffle the data set (if necessary)
    if self.args.rollout_global_dataset and self.args.rollout_shuffle:
        self.dataset.shuffle(self.epoch_id)
```
</details>

### RolloutDataSourceWithBuffer class

**Function**
The buffered data source class inherits from `RolloutDataSource`, adds data buffering function, and supports data reuse and partial rollout.

**Key Attributes**
<details>
<summary>RolloutDataSourceWithBuffer initialization</summary>

```python
class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = [] # Data buffer
        
        # Set buffer filter
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first #Default: first in, first out
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path) # Custom filter
```
</details>

**get_samples() method**

**Function**: Get data from the buffer first, and supplement it from the original data set when the buffer is insufficient.

<details>
<summary>get_samples method</summary>

```python
def get_samples(self, num_samples: int) -> list[list[Sample]]:
    # 1. First get the sample from the buffer
    samples = self._get_samples_from_buffer(num_samples)
    num_samples -= len(samples)
    
    # 2. If the buffer is not enough, get the remaining samples from the original data set
    if num_samples > 0:
        samples += super().get_samples(num_samples=num_samples)
    
    return samples
```
</details>

**Data acquisition priority**:
1. **Buffer first**: Get data from the buffer first
2. **Data set supplement**: Obtain from the original data set when the buffer is insufficient
3. **Seamless integration**: Mixed use of buffer and dataset data

**_get_samples_from_buffer() method**

**Function**: Get the specified number of sample groups from the buffer.

<details>
<summary>_get_samples_from_buffer method</summary>

```python
def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
    if len(self.buffer) == 0 or num_samples == 0:
        return [] # buffer is empty or no sample is needed
    
    # Use buffer filter to get samples
    samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
    return samples
```
</details>

**Key Points**:
- Use the `buffer_filter` function to decide how to select samples from the buffer
- The `pop_first` function is used by default (first in, first out)

**add_samples() method**

**Function**: Add a sample group to the buffer.

<details>
<summary>add_samples method</summary>

```python
def add_samples(self, samples: list[list[Sample]]):
    if not samples:
        return
    
    # Verify input format
    assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
    assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
    
    # Verify the size of each group
    for i in range(0, len(samples)):
        assert(
            len(samples[i]) == self.args.n_samples_per_prompt
        ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
        group = samples[i]
        self.buffer.append(group) # Add to buffer
```
</details>

**Verification Mechanism**:
1. **Format Verification**: Make sure the input is in the format of `list[list[Sample]]`
2. **Size verification**: Ensure that each group contains the correct number of samples
3. **Data Integrity**: Ensure that the data format in the buffer is consistent

**Helper Method**

<details>
<summary>Helper methods</summary>

```python
def update_metadata(self, metadata: dict):
    """Update metadata"""
    self.metadata.update(metadata)

def get_metadata(self):
    """Get metadata"""
    return self.metadata

def get_buffer_length(self):
    """Get buffer length"""
    return len(self.buffer)
```
</details>

### pop_first() function

**Function**
The default buffer filter implements a first-in-first-out (FIFO) data acquisition strategy.

<details>
<summary>pop_first function</summary>

```python
def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples) # Take the smaller value of buffer length and demand
    samples = buffer[:num_to_pop] # Get the first num_to_pop samples
    del buffer[:num_to_pop] # Delete these samples from the buffer
    return samples
```
</details>

**Features**:
- **FIFO strategy**: Data that enters the buffer first is taken out first
- **Safe access**: Will not exceed the actual length of the buffer
- **Memory Management**: Delete from the buffer immediately after taking it out

## Data flow and calling relationship

### Call chain

```sh
RolloutController.generate()
    ↓
RolloutDataSourceWithBuffer.get_samples()
    ↓
_get_samples_from_buffer() + super().get_samples()
    ↓
Return list[list[Sample]]
```

### Buffer usage scenarios

**A. Partial Rollout**
<details>
<summary>Partial Rollout example</summary>

```python
# In sglang_rollout.py, aborted samples will be written back to the buffer
if hasattr(data_source, 'add_samples') and len(filtered_data) > args.rollout_batch_size:
    rejected_samples = filtered_data[args.rollout_batch_size:]
    data_source.add_samples(rejected_samples)
```
</details>

### Status management

**A. Training Recovery**
<details>
<summary>Training recovery example</summary>

```python
# in train.py
if args.rollout_global_dataset:
    ray.get(rollout_manager.controller.load.remote(args.start_rollout_id - 1))
```
</details>

**B. Checkpoint Save**
<details>
<summary>Checkpoint saving example</summary>

```python
# in train.py
if args.rollout_global_dataset:
    ray.get(rollout_manager.controller.save.remote(rollout_id))
```
</details>

## Summary of design features

1. **Layered design**: basic data source + buffer extension
2. **State Persistence**: Supports recovery from training interruptions
3. **Data reuse**: Improve data utilization through the buffer mechanism
4. **Flexible filtering**: Supports custom buffer selection strategy
5. **Data Integrity**: Strict format verification and status management
6. **Epoch Management**: Automatically handle data set boundaries and reshuffle

## Key configuration parameters

| Parameters | Description | Default value |
|------|------|--------|
| `rollout_global_dataset` | Whether to use the global data set | False |
| `rollout_shuffle` | Whether to shuffle the data set | False |
| `n_samples_per_prompt` | Number of samples generated per prompt | 8 |
| `buffer_filter_path` | Custom buffer filter path | None |
| `rollout_max_prompt_len` | Maximum prompt length | - |
| `input_key` | Input field name | - |
| `label_key` | Label field name | - |

## Usage example

### **Basic usage**
<details>
<summary>Basic usage examples</summary>

```python
#Create data source
data_source = RolloutDataSourceWithBuffer(args)

# Get samples
samples = data_source.get_samples(32) # Get 32 prompt groups

#Add sample to buffer
data_source.add_samples(rejected_samples)
```
</details>

### **Custom Buffer Filter**
<details>
<summary>Example of custom Buffer filter</summary>

```python
#Define custom filter
def custom_buffer_filter(args, rollout_id, buffer, num_samples):
    # Sort by reward and take the sample with the highest reward
    sorted_buffer = sorted(buffer, key=lambda x: x[0].reward, reverse=True)
    return sorted_buffer[:num_samples]

# Set in args
args.buffer_filter_path = "path.to.custom_buffer_filter"
```
</details>

This design enables the rollout system to efficiently manage training data and support complex training scenarios such as partial rollout and over-sampling.