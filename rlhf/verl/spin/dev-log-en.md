# Online DPO / SPIN implementation log

## Main transformation content

### 1. Create new DPO Ray Trainer and main_dpo

- **main_spin**
  Created a new `main_dpo` module as the entry script for online DPO. It is responsible for loading the configuration, initializing the Ray cluster and constructing the DPO trainer, and starting the training process. This module has been appropriately modified based on the original PPO entrance to adapt to the online DPO process.

- **RayDPOTrainer**
  The new `RayDPOTrainer` reuses PPO's resource pool management, worker grouping and data loading logic, while calling the new DPO update interface during the training update phase. This interface utilizes the contrastive loss calculation (sigmoid or IPO) implemented in the core algorithm (core_algos) to implement direct updates of the policy model.

### 2. Core algorithm module (core_algos) for online dpo

- **Loss Calculation**
  In core_algos, we have added two new functions:
  - `compute_online_dpo_loss` (sigmoid version): Calculate a sigmoid based loss using the difference in log probability ratios.
  - Another version is the IPO loss (version based on squared differences).
  
  Both functions use the hyperparameter `beta` to control the magnitude of model updates and output the mean loss on return.

### 3. Patch upgrade PPO Worker

- **New DPO update interface**
  In the original DataParallelPPOActor, the `update_policy_dpo` method was added. This method is similar to the traditional PPO update step, but it receives the chosen and rejected responses merged by union, extracts `"chosen_mask"` from meta_info, and then calls the DPO loss function in the core algorithm module to calculate the loss and perform backpropagation and gradient update.
  In the original ActorRolloutRefWorker(Worker), the `update_actor_dpo` method was added. Provides an interface for `update_policy_dpo`.

### 4. Some pitfall records

- **trainer.ref_update_freq setting**
  During the previous implementation test, the setting of this parameter was always too large, causing the model to be easily rewarded and hacked, entering the local max, causing the training to crash. After re-selecting, it has been able to stabilize the rising point and converge.