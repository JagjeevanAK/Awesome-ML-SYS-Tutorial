# DAPO Dynamic Filtering implementation and Batch Size analysis


The relevant logic of `dynamic filtering` has been well implemented in the code. We now want to explore whether a higher degree of parallelism can be achieved by filling the prompt to a smaller batch size. For example, padding to `mini_batch_size`. This requires us to do a more comprehensive analysis of the subsequent processing of the completed batch. Moreover, more batches means more check->back fill. Whether it will affect performance also needs to be considered.


## 1 Filter Groups configuration entry

`FilterGroupsConfig` defines three parameters: whether to enable, filter indicators, and dynamic supplementary acquisition upper limit.

```py
46:67:verl/trainer/config/algorithm.py
@dataclass(frozen=True)
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy)."""

    enable: bool = False # switch
    metric: Optional[str] = None # acc / score / seq_reward / seq_final_reward
    max_num_gen_batches: int = 0 # ≤0 means no upper limit
```

## 2 Batch Size related parameters

| Field | Function | Settings in `test_dapo_7b.sh` |
|------|------|----------|
| `data.train_batch_size` | **Prompt number** used for one parameter update after filtering | **512** |
| `data.gen_batch_size` | The number of initial prompts generated in each round | **1536** |
| `rollout.n` | Number of Response samples per Prompt | **16** |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | Actually also the batch size of a single GPU | **32** |
| `ppo_micro_batch_size_per_gpu` | Barch_size of a single GPU TODO: relationship with mini size | None |


## 3 Core implementation (`dapo_ray_trainer.py`)

The following code snippet shows a **dynamic filtering loop** within one iteration:

```py
178:230:recipe/dapo/dapo_ray_trainer.py
else: # NOTE: If the number of Prompts is less than train_batch_size after filtering, it will jump to the next round of generation.
        metric_name = self.config.algorithm.filter_groups.metric
        # Prepare sequence-level metrics for diversity checks
        if metric_name == "seq_final_reward":
            # To facilitate calculation of std, convert tensor to numpy
            new_batch.non_tensor_batch["seq_final_reward"] = (
                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
            )
        elif metric_name == "seq_reward":
            new_batch.non_tensor_batch["seq_reward"] = (
                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
            )

        # Collect indicator values by Prompt UID
        prompt_uid2metric_vals = defaultdict(list)
        for uid, metric_val in zip(
            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]
        ):
            prompt_uid2metric_vals[uid].append(metric_val)

        # Calculate the standard deviation within the group and determine the retained prompt
        prompt_uid2metric_std = {}
        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

        kept_prompt_uids = [
            uid
            for uid, std in prompt_uid2metric_std.items()
            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1 # Keep if diverse/single sampling
        ]
        num_prompt_in_batch += len(kept_prompt_uids)

        #Map to track index based on retained UID
        kept_traj_idxs = []
        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
            if traj_from_prompt_uid in kept_prompt_uids:
                kept_traj_idxs.append(idx)

        # Keep only filtered trajectories
        new_batch = new_batch[kept_traj_idxs]
        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

        # Determine whether to continue generating or aligning the batch size
        prompt_bsz = self.config.data.train_batch_size
        if num_prompt_in_batch < prompt_bsz:
            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                print(f"{num_gen_batches=}. Keep generating...")
                progress_bar.update(1)
                continue
            else:
                raise ValueError(
                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                )
        else:
            # Number of trajectories aligned to train_batch_size × rollout.n
            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
            batch = batch[:traj_bsz]
```

Process points:
1. The variance calculation method is determined by `metric_value`.
2. If the standard deviation of the indicator within the group is 0 and there is more than one prompt in the group, the entire group of prompts is discarded. 
3. The effective prompt after discarding is less than `train_batch_size`, and collection continues until the quantity is met or exceeds `max_num_gen_batches`.
4. Once the target Prompt number is reached, it still needs to be cropped to an integer multiple of `train_batch_size × rollout.n` for subsequent alignment.

## 4 mini-batch related logic

In `RayPPOTrainer._validate_config`, there is verification logic for batch size:

```py
# 430:462:verl/trainer/ppo/ray_trainer.py
real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
assert real_train_batch_size % minimal_bsz == 0 # Can be evenly divided by DP

# Actor check when dynamic_bsz is not enabled
assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sequence_parallel_size >= n_gpus
```
TO BE CONTINUE...
