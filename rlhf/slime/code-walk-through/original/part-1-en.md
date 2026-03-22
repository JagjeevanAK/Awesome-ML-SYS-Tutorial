# A Brief Code Walkthrough of slime


## Introduction

**slime** is an LLM post-training framework designed for large-scale reinforcement learning training.



### Core Competencies

1. **High-performance training**: Provides distributed training capabilities through Megatron-LM, supporting Dense and MoE models
2. **Flexible data generation**: Through the SGLang engine and custom interfaces, any complex data generation process can be realized
3. **Asynchronous training**: Supports asynchronous execution of training and inference, significantly improving GPU utilization.

### Project link

- **Project address**: [https://github.com/THUDM/slime/tree/main/slime](https://github.com/THUDM/slime)
- **Documentation**: [slime/docs/](https://github.com/THUDM/slime/tree/main/docs)
- **Docker image**: `zhuzilin/slime:latest`

## Core architecture

slime adopts a decoupled architecture and decomposes the RLHF training process into three independent and collaborative modules:

- **Training (Megatron)**: Responsible for the main training process and supports multiple parallel strategies
  - *Code location*: [`slime/backends/megatron_utils/`](https://github.com/THUDM/slime/tree/main/slime/backends/megatron_utils/)
  
- **Rollout (SGLang)**: Generate new data (including reward/verifier), optimize reasoning based on SGLang
  - *Code location*: [`slime/ray/rollout.py`](https://github.com/THUDM/slime/tree/main/slime/ray/rollout.py)
  
- **Data Buffer**: Bridge module, manages data flow and custom generation logic
  - *Code location*: [`slime/ray/buffer.py`](https://github.com/THUDM/slime/tree/main/slime/ray/buffer.py)

### Overall workflow

![slime overall workflow](overall_workflow.jpg)

The above figure shows the core workflow of slime, including the complete interaction process of training loop, RolloutController, RolloutDataSourceWithBuffer and SGLang distributed reasoning system.## Key Features

### Distributed resource management

Resource scheduling based on Ray framework:
- **Placement Groups**: Resource isolation and allocation
- **Multiple parallel strategies**: data/tensor/pipeline/expert parallelism
- **Dynamic expansion and contraction**: Training and inference resources can be adjusted independently

*Core implementation*: [`slime/ray/placement_group.py`](https://github.com/THUDM/slime/tree/main/slime/ray/placement_group.py)

### Asynchronous training optimization

slime provides two training modes:

- **Synchronous training** ([`train.py`](https://github.com/THUDM/slime/tree/main/train.py)): traditional sequential execution mode
- **Asynchronous training** ([`train_async.py`](https://github.com/THUDM/slime/tree/main/train_async.py)), under the disaggregated architecture, use `rollout_manager.async_generate` and `actor_model.async_train` to separate training and push asynchronous training, in which rollout is always one step ahead of train, using the off-policy strategy

### Flexible data generation

Supports user-defined complex data generation logic:
- Multiple rounds of dialogue ([example](https://github.com/THUDM/slime/tree/main/examples/search-r1))
- Tool call
- Reward model integration
- Custom validator

*Extended interface*: [`slime_plugins/rollout_buffer/`](https://github.com/THUDM/slime/tree/main/slime_plugins/rollout_buffer/)


## Usage scenarios

### Supported model types

- **Dense Model**: GLM-4-9B, Qwen3-4B, etc.
  - *Configuration example*: [`slime/scripts/run-qwen3-4B.sh`](https://github.com/THUDM/slime/tree/main/scripts/run-qwen3-4B.sh)
  
- **MoE Model**: Qwen3-30B-A3B, DeepSeek-R1, etc.
  - *Configuration example*: [`slime/scripts/run-deepseek-r1.sh`](https://github.com/THUDM/slime/tree/main/scripts/run-deepseek-r1.sh)### Training task type

- **Reinforcement Learning**: PPO, GRPO, DPO and other algorithms
- **Supervised Fine-tuning**: SFT training support

### Deployment mode

- **Single machine with multiple cards**: Suitable for small and medium-sized models
- **Multiple machines and multiple cards**: Support large-scale distributed training (such as 128×H100)
- **Hybrid deployment**: Separate deployment of training and inference resources

## Code structure

```sh
slime/
├── slime/ # Core framework code
│ ├── ray/ # Ray distributed component
│ │ ├── actor_group.py # Training Actor Management
│ │ ├── rollout.py # Reasoning Actor Management
│ │ ├── buffer.py # Data buffer
│ │ └── placement_group.py # Resource allocation
│ ├── backends/ # Backend engine integration
│ │ ├── megatron_utils/ # Megatron training backend
│ │ └── sglang_utils/ # SGLang inference backend
│ └── utils/ # Utility function
├── slime_plugins/ # Plugins and extensions
│ ├── rollout_buffer/ # Custom generated plug-in
│ └── models/ # Model adaptation
├── scripts/ # Startup script
│ └── models/ # Configuration of each model
├── examples/ # Usage examples
├── docs/ # Detailed documentation
├── train.py # Synchronous training entrance
└── train_async.py # Asynchronous training entrance
```
---

*Reference architecture design: [SGLang Code Walk-through](https://github.com/maocheng23/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme-CN.md)*

### Purpose and serial relationship of each directory

- `scripts/`: startup script and model configuration
  - Used to start the Ray cluster and submit training jobs; the sample script will choose `train.py` or `train_async.py`
  - For example: `slime/scripts/run-qwen3-4B.sh`, `slime/scripts/run-deepseek-r1.sh`

- `train.py` / `train_async.py`: training entrance
  - Create `PlacementGroup` to allocate GPU → Create `actor_group` (training) and `rollout_manager` (inference) → Enter the training loop
  - Synchronous mode executes step by step; asynchronous mode is interleaved with `rollout_manager.async_generate()` and `ray.get()` to parallelize

- `slime/ray/`: distributed orchestration and resource management
  - `placement_group.py`: GPU resource allocation and packaging based on Ray Placement Group
  - `actor_group.py`: training Actor group management, exposing `async_init/async_train/async_update_weights` and other interfaces
  - `rollout.py`: Rollout Actor (SGLang engine container), inference service routing, weight reception
  - `buffer.py`: data buffering, sample batch organization, and an intermediate bridge with Rollout/Training

- `slime/backends/`: backend engine adaptation
  - `megatron_utils/`: training backend (optimizer, weight update, integration with distributed communication)
  - `sglang_utils/`: inference backend (wrapper SGLang, batch generation, engine lifecycle management)

- `slime_plugins/`: pluggable extension
  - `rollout_buffer/`: Custom trajectory generator system through external linkage such as HTTP/OpenAI interface
  - `models/`: small adaptation layers for different model families

- `examples/`: Minimal runnable example
  - For example, `examples/search-r1/` shows the generation and training series of multiple rounds of dialogue + tool calls- `docs/`: documentation and usage guide
  - Contains model usage, SFT, AMD and other platform adaptation and tuning manuals

### Series relationship (from script to training and generation)

1) Script layer (`scripts/`)
- Start Ray → Submit job → Select `train.py` or `train_async.py` and pass in the parameters

2) Entry layer (`train*.py`)
- `create_placement_groups(args)` allocate/map GPU
- `create_actor_group(args, pgs["actor"])` Build training Actor group
- `create_rollout_manager(args, pgs["rollout"])` Build inference and data generation manager

3) Execution layer (`ray/` + `backends/`)
- Training: `actor_group.async_train(...)` → Megatron optimization/gradient calculation
- Generate: `rollout_manager.async_generate(...)` → SGLang batch inference
- Synchronization: `actor_group.async_update_weights()` → Push training weights to the inference engine

4) Data flow (`buffer.py` + plug-in)
- `Buffer` is responsible for sampling/batching/calling custom generation (`slime_plugins/rollout_buffer/`) → returns available samples for training

Through the above links, slime naturally strings together *script → entry → distributed execution → data/weight flow* to achieve efficient and scalable RL post-training.