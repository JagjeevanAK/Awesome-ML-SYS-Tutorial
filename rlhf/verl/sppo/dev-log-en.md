# Dev-log

Compared with ppo implemented by Verl itself, the main difference between sppo is not using critic and modifying loss (passing in rewards).

plan:

Since the monkey patch logic of ray may be different from the stand-alone version, the first version first intrusively modified the actor of ppo to verify the correctness of the algorithm. See the increase in val_score. If the loss is implemented correctly, then consider monkey patch or implement workers and actors by yourself (if monkey patch cannot be implemented).

Current step:

1. Create new main_SPPO and SPPO Ray Trainer
   
main_sppo
- Modifications have been made based on the original PPO entrance to adapt to the SPPO process.

RaySPPOTrainer
- Reuse PPO Trainer logic to the greatest extent, and implement SPPO update logic in fit.

Algorithm correctness verification (val_score 0.78 -> 0.92) has been implemented, and the next step is to make the code structure more reasonable.

```
main_sppo -> override trainer fit()
          -> fsdp_workers.ActorRolloutRefWorker override init_model() 
	  -> DataParallelPPOActor override update_policy -
 	  -> update_policy
                 
	 
	 -> megatron_workers.ActorRolloutRefWorker needs support ?
```
Confusing points during the implementation process:

There are two paths:

1. Implement it yourself according to the SPPO data flow method according to Tutorial(https://verl.readthedocs.io/en/latest/advance/dpo_extension.html).
2. Learn the PPO source code, and based on the difference between SPPO and PPO (without using critic + modifying the loss function), copy the ppo implementation logic + minimize the modification differences.

Finally, I chose step 2 because I don’t know much about what logic is necessary for verl on the implementation path of PPO (resulting from not being familiar enough with RL algorithms and Verl)