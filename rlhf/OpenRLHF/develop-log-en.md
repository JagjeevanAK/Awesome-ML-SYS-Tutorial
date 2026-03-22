# Extend the inference engine of OpenRLHF

As we all know, OpenRLHF has used vllm as the main inference engine for a long time, and I hope to be able to connect SGLang to it, so this log mainly records this development process. Although I have been doing this for several weeks, there are really big pitfalls along the way. The pitfalls I have encountered under SGLang have been explained in detail. Here is a ref:

- [Latency optimization for weight updates](./sglang/latency-accelerate-for-weight-updates/readme.md): A debugging process for efficiency, also published in [An optimization of SGLang weight update latency](https://zhuanlan.zhihu.com/p/9908228168).

## Quick Start

OpenRLHF's documentation defaults to users who understand the RLHF process, so many places written in it are not introductory. It is more painful for people like me who don't understand RLHF very well. Just running it will encounter a lot of pitfalls.

### Configuration environment

I misjudged the dependency complexity of OpenRLHF at first and guessed it should be very high, so I chose docker. Later I found out that all I needed was deepspeed vllm and openrlhf together. However, here I will share the docker instructions I use myself:```bash
docker run --runtime=nvidia -it --shm-size="40g" --cap-add=SYS_ADMIN   -v /opt/dlami/nvme/chenyang:/var/lib/docker   
nvcr.io/nvidia/pytorch:24.07-py3 bash
```
I removed `--rm` in [Original Document Instructions](https://openrlhf.readthedocs.io/en/latest/quick_start.html#installation). I don’t understand why this parameter is added, which causes the docker container to be automatically deleted after exiting.

After entering docker, first uninstall some libraries in the environment to avoid dependency conflicts with OpenRLHF.```bash
pip uninstall xgboost transformer_engine flash_attn -y
```
Then, install OpenRLHF with vllm dependency.```bash
 pip install openrlhf[vllm]
```
This release may occasionally be canceled. You can also directly install the latest specified version of openrlhf and vllm. The former version does not matter. The latter has to find the supported version from the dependencies of OpenRLHF. The latest vllm may not support it. I used 0.6.4.post1.

If you use docker, you can save the docker commit, use `docker ps -a` to search for `<container_id>`, and then `docker commit <container_id> openrlhf_chenyang`. Next time you can directly enter docker by `docker run --gpus all -it openrlhf_chenyang`.

Finally, I configured `wandb`. To be honest, I haven’t touched this thing for almost two years. I increasingly feel that it has little meaning other than monitoring the training curve. OpenRLHF can be used based on ray, and ray has its own set of prometheus monitoring. You can directly use ray dashboard to view the log. Of course, it is not troublesome to configure `wandb`, just use `wandb init`.

### Quick Check Out

Since I mainly use a single machine with multiple cards for SGLang and vllm shooting, I do not use multi-machine mode. Here are two simple instructions:```bash
ray start --head --node-ip-address 127.0.0.1 --num-gpus 3 --port 4567 --temp-dir="/opt/dlami/nvme/chenyang/.cache/ray"
```
This is to start the head node of ray on three cards of a single machine. You may encounter various startup failures, such as the port is occupied or the card is not allocated enough, so you will continue to run `ray stop` and `ray start` until it succeeds. In addition, ray is a very powerful resource scheduler. If 6 cards are opened here, the remaining 3 cards can be assigned to other tasks.

<details>
<summary>Output of ray start</summary>

```bash
ray start --head --node-ip-address 127.0.0.1 --num-gpus 3 --port 4567

Usage stats collection is enabled. To disable this, add `--disable-usage-stats` to the command that starts the cluster, or run the following command: `ray disable-usage-stats` before starting the cluster. See https://docs.ray.io/en/master/cluster/usage-stats.html for more details.

Local node IP: 172.31.54.252

--------------------
Ray runtime started.
--------------------

Next steps
  To add another node to this Ray cluster, run
    ray start --address='172.31.54.252:4567'

  To connect to this Ray cluster:
    import ray
    ray.init(_node_ip_address='172.31.59.18')

  To submit a Ray job using the Ray Jobs CLI:
    RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python my_script.py

  See https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html
  for more information on submitting Ray jobs to the Ray cluster.

  To terminate the Ray runtime, run
    ray stop

  To view the status of the cluster, use
    ray status

  To monitor and debug Ray, view the dashboard at
    127.0.0.1:8265

  If connection to the dashboard fails, check your firewall settings and network configuration.
```
</details>

The start address of ray is given here, that is, `ray start --address='172.31.59.18:4567'`. Note that this address must be used in OpenRLHF instructions later. Then the address of ray dashboard is also given, which is `127.0.0.1:8265`. You can view very detailed monitoring information by logging in.

Next, submit a test job. This is a script that I ran through on three H100s. You can refer to it.

<details>
<summary>Test Job</summary>

```bash
#Adjust url ray start address, working_dir and save_path according to needs

ray job submit --address="172.31.59.18:4567" \
   --runtime-env-json='{"working_dir": "/opt/dlami/nvme/chenyang/rlhf-ckpt"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /opt/dlami/nvme/chenyang/rlhf-ckpt/examples/checkpoint/llama3-8b-rlhf \
   --save_steps 100 \
   --micro_train_batch_size 16 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 128 \
   --max_samples 512 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing
```
</details>

Any framework has to trade off between ease of use and performance. My instructions above can almost complete the OpenRLHF process test the fastest. Pay attention to these parameters:

1. `colocate_critic_reward` and `colocate_actor_ref`: Put critic/reward and actor/ref on the same card, which significantly saves video memory, but there are some empty caches in the middle, which will slow down the training speed. If not enabled, each card will occupy one card and the video memory usage will double.
2. `adam_offload`: Offload the adam optimizer to the CPU, which significantly saves video memory, but will slow down the training speed. If not enabled, OOM will occur on the 80G H100.
3. `max_samples` is the maximum number of samples sampled from `prompt_data`, which must be greater than `rollout_batch_size`, otherwise it is not enough for one rollout and an error will be reported.

Finally, let me add how to stop the openrlhf process. It is actually very violent:```bash
pkill -9 -f train_ppo_ray
```
## Analyze the use of Ray in OpenRLHF

This is mainly based on this Zhihu article: [Illustration of Ray-based distributed training process in OpenRLHF](https://zhuanlan.zhihu.com/p/12871616401). The original text is very clear. Here is some further elaboration. You can read it together with the original text for more clarity.

### Some core concepts of Ray

The original article mentioned some concepts of Ray, but I personally feel it is a little vague, so I will add more.

1.Placement Group

There is a variable `pg` in OpenRLHF, which most of the time refers to the Placement Group, not the process group in torch communication. Placement Group can be understood as a set of resource allocation plans that allow users to precisely control resource allocation and task scheduling. For example here:```python
import ray

#Create Placement Group
pg = ray.util.placement_group(
    bundles=[{"CPU": 2, "GPU": 1}, {"CPU": 4, "GPU": 2}],
    strategy="PACK"
)

# Use Placement Group to specify the execution location of the task
@ray.remote(placement_group=pg)
def train_model():
    # Code for training the model
    pass
```2. Driver

Ray The control node of the program, usually the starting point of the program. It usually runs on a separate node and is responsible for starting the Ray cluster, submitting tasks, and scheduling execution. The driver side does not perform calculation work, but allocates calculation tasks through remote calls.

3. Worker

Worker is a computing node in the Ray cluster, responsible for executing tasks submitted by Driver. Multiple Worker processes run on each Worker node, and these processes handle tasks from Drivers or other Workers.

4. Task

Ray Task is the most basic computing unit, usually representing a function or operation that needs to be executed, and is the smallest unit of parallel execution. Each task is a function call that is assigned to a Worker in the Ray cluster for execution. **The task is stateless. It will not save any state after executing the task. Each execution is independent. **

5. Actor and Actor Handle

Unlike Tasks, Actors are stateful computing units in Ray that retain internal state during their lifetime. When created, Ray allocates an independent execution instance to it and returns its reference Actor Handle. When the Actor method is called through the Actor Handle, the Driver will send the request to the appropriate Worker node through the Ray scheduling system.```python
import ray

# Initialize Ray cluster
ray.init()

# Define a simple Actor class
@ray.remote
class Counter:
    def __init__(self, value=0):
        self.value = value

    def increment(self):
        self.value += 1
        return self.value

#Create an Actor instance and return an Actor Handle
counter_handle = Counter.remote()

# Call the increment method through Actor Handle
result = ray.get(counter_handle.increment.remote())
print(result) # Output 1

# Call the increment method again
result = ray.get(counter_handle.increment.remote())
print(result) # Output 2
```What’s more troublesome is that Actor in the Ray system and Actor in RLHF are two different concepts, and the two will be specially distinguished later. In OpenRLHF, `PPORayActorGroup` represents the Actor group of the Ray system, while `ActorModelRayActor` represents the Actor in Ray-based RLHF.

### colocate’s resource allocation strategy

OpenRLHF implements the colocate strategy of Actor/Reference and Value/Reward, that is, Actor and Reference will share the same computing resources. Intuitively, I saved almost half of the video memory, and can be turned on directly through `--colocate_actor_ref`. What’s more interesting is that after colocate is turned on, the resources are not actually divided in half, but:```python
actor_model = PPORayActorGroup(
    args.actor_num_nodes,
    args.actor_num_gpus_per_node,
    ActorModelRayActor,
    pg=pg,
    num_gpus_per_actor=0.75 if pg else 1,
)

ref_model = PPORayActorGroup(
    args.ref_num_nodes,
    args.ref_num_gpus_per_node,
    ReferenceModelRayActor,
    pg=pg,
    num_gpus_per_actor=0.25 if pg else 1,
)
```
Here is a trick. The general idea is that according to the current startup logic, assuming that the actor model requires data parallelism to occupy two cards, and set `num_gpus_per_actor=0.5`, Ray will first start the first actor model on the first card with 0.5 video memory, and then allocate the second actor model occupying 0.5 video memory. Ray will continue to allocate the second actor model to the first card and use the saved memory. 0.5 instead of the second card. Therefore, when colocating, the strategy of `num_gpus_per_actor=0.75, 0.25` was adopted. Graphics cards are not actually split in half, and this strategy has no impact when using only one card.

## Extend OpenRLHF’s inference engine

After finishing these preliminary tasks, let’s get down to business. As you all know, one of my big jobs is to support the SGLang backend in the OpenRLHF system, with two specific requirements:

1. Support SGLang’s inference to ensure accuracy and speed can be matched.
2. Abstract the current vllm engine into an inference Engine Backend class, and then this backend supports huggingface, SGLang and vllm

Based on my long-term development experience, I will first review all vllm uses in OpenRLHF to achieve a unified backend.

### `openrlhf/cli/batch_inference.py`

This file implements three functions, using vllm and transformers for generation and using transformers for inference to obtain rewards. This approach is not necessarily rigorous, because strictly speaking, the inference engine can only be used for generation in RLHF, and the generated log probs, logits, embedding and reward are all inaccurate:

> There is a big gap between the kernel fusion of the inference engine and the training engine. When the batch sizes are different, the inference requests are dispatched to different kernels, and then the numerical errors are accumulated layer by layer. When the log probs layer is reached, it reaches a level that cannot be ignored. This problem has existed since the Bert era. The accuracy difference between the training engine and the inference engine cannot be avoided, and it may not be repaired within a month or two if we work hard.
>
> So now the inference engine in RLHF is more about accelerating sampling, and reward and embedding have to be calculated using training scripts. It may take several months to study this issue in half a year.

These three functions are still very simple. As I have described, we need to create a unified backend, so the general idea of ​​modifying this file is to open a new class GenerationBackend, make a branch in GenerationBackend, and implement the inference of SGLang, vllm and transformers.

After writing this, I discovered a surprising thing, OpenRLHF does not have a single test. I will first test the usability of this system. Refer to this `examples/scripts/train_rejection_sampling_llama.sh` and write a one-sided shot:

<details>
<summary>Single shot test</summary>

```bash
# For vllm
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p ./checkpoint/llama-3-8b-rejection
GENERATE_OUTPUT=./checkpoint/llama-3-8b-rejection/generate.jsonl
RM_OUTPUT=./checkpoint/llama-3-8b-rejection/rm.jsonl
ITER_LOG_PATH=./checkpoint/llama-3-8b-rejection/iter.log
MODEL_OUTPUT_PATH=./checkpoint/llama-3-8b-rejection

TRAINING_ITERS=10
ROLLOUT_BATCH_SIZE=10240

POLICY_MODEL_PATH=OpenRLHF/Llama-3-8b-sft-mixture

iter=0

python batch_inference.py \
   --eval_task generate_vllm \
   --pretrain $POLICY_MODEL_PATH \
   --bf16 \
   --max_new_tokens 2048 \
   --prompt_max_len 2048 \
   --dataset OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --temperature 0.9 \
   --zero_stage 0 \
   --best_of_n 4 \
   --enable_prefix_caching \
   --tp_size 2 \
   --max_num_seqs 64 \
   --iter $iter \
   --rollout_batch_size $ROLLOUT_BATCH_SIZE \
   --output_path $GENERATE_OUTPUT \
   --max_samples 200
```

```bash
# For SGLang

mkdir -p ./checkpoint/llama-3-8b-rejection
GENERATE_OUTPUT=./checkpoint/llama-3-8b-rejection/generate.jsonl
RM_OUTPUT=./checkpoint/llama-3-8b-rejection/rm.jsonl
ITER_LOG_PATH=./checkpoint/llama-3-8b-rejection/iter.log
MODEL_OUTPUT_PATH=./checkpoint/llama-3-8b-rejection

TRAINING_ITERS=10
ROLLOUT_BATCH_SIZE=10240

POLICY_MODEL_PATH=OpenRLHF/Llama-3-8b-sft-mixture

iter=0

python batch_inference.py \
   --eval_task generate_sglang \
   --pretrain $POLICY_MODEL_PATH \
   --bf16 \
   --max_new_tokens 2048 \
   --prompt_max_len 2048 \
   --dataset OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --temperature 0.9 \
   --zero_stage 0 \
   --best_of_n 4 \
   --enable_prefix_caching \
   --tp_size 2 \
   --max_num_seqs 64 \
   --iter $iter \
   --rollout_batch_size $ROLLOUT_BATCH_SIZE \
   --output_path $GENERATE_OUTPUT \
   --max_samples 200
```
</details>


After writing, I discovered that sglang vllm and openrlhf have irreconcilable conflicts. The torch dependencies of sglang and vllm are different, and it cannot be repaired at present. I have tried many vllm versions but cannot solve this problem. You can only start diverge out of two environments here. The reason for using two environments instead of two dockers is that I am not used to docker mapping and do not want to reset system variables.

Installing the environment is very simple now that I think about it. When I first installed it, I did it myself, but now I can refer to [SGLang developer mode environment configuration document](https://sgl-project.github.io/references/contribution_guide.html#setting-up-building-from-source). As for openrlhf and vllm, as described at the beginning, install openrlhf first, then install ray, and finally install vllm. After doing this for a long time, I found that the nvidia link of the new cluster torch kept failing. A few years ago, I encountered [this problem](https://github.com/pytorch/pytorch/issues/111469) when I was still playing in the mud at SenseTime. However, three years later, my solution ideas have matured a lot, and now there is GPT to ask. Here is a list of the solution process for this bug:

1. I found that there has been a problem with the torch link. I changed `LD_LIBRARY_PATH` according to [this issue](https://github.com/pytorch/pytorch/issues/111469) and a series of other methods. I tried various environments such as docker and conda, but there is still a problem with the torch 2.5.1 link:```bash
ImportError: /usr/local/lib/python3.10/dist-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```
2. I found that the ray submit job has this problem, but there is no problem in ipython. Narrowing the scope further, I found that `python -c` was fine, so what was the problem?

3. I asked claude, it’s a problem with the ray environment:

<details>
<summary>claude's answer </summary>

> No, why the hell can python -c import, but ray submit cannot?

Ah, this is a very typical environment variable inheritance problem.

When you execute `python -c` in your local terminal, it inherits all environment variables you set in `.zshrc` (including `LD_LIBRARY_PATH`). But when submitting tasks through ray submit, these environment variables may not be passed correctly to the ray worker process.

It is recommended to try the following solutions:

1. Explicitly set environment variables in the ray task:```python
import os
os.environ["LD_LIBRARY_PATH"] = "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages/nvidia/cusparse/lib"
```
2. Or pass runtime_env when ray.init():```python
ray.init(runtime_env={
    "env_vars": {
        "LD_LIBRARY_PATH": "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages/nvidia/cusparse/lib"
    }
})
```
3. If you use ray submit, you can add:```bash
export RAY_RUNTIME_ENV='{"env_vars": {"LD_LIBRARY_PATH": "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages/nvidia/cusparse/lib"}}'
ray submit script.py
```
This should ensure that environment variables are passed correctly to the ray worker.


</details>

The problem was solved. When I submitted the job, I had to write it like this:```bash
export RAY_RUNTIME_ENV='{"env_vars": {"LD_LIBRARY_PATH": "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages/nvidia/cusparse/lib"}}'
ray submit script.py
```
<!-- It's very annoying, you see me and claude almost swearing when talking to each other. It seems that I have accomplished a lot today, but in fact I always feel like I am standing still:

1. It is found that the latest dependency conflicts between vllm, sglang and openrlhf cannot be resolved at present. It is not only a problem with outlines, but also a deeper problem with torch. I tried using vllm 0.6.3.post1 and vllm 0.6.4.post1 to see if they were compatible. In the end, only vllm 0.6.5's `update_weights_from_distritbued` was successful in the environment at that time. Other versions did not work, and it took another hour. 
2. I had no choice but to try the diverge environment, only to find that the first cluster crashed while writing. God knows if it was me, but I never write to `/home`.
3. Switch clusters, modify several configurations, and finally set up another H100. Then I discovered the angel’s torch link error. I tried various methods for 2 hours. I first used conda to open a new environment, and then used docker to try to bypass the torch link. I found no results and was very desperate.
4. Report the problem to everyone in the group, and try `python -c` to see if there are any errors. I didn't find it, so I finally asked claude and found out about the ray environment variable problem. Without modern LLM, this bug would really have made me autistic two years ago, and I remembered the pain I experienced configuring deepspeed on SenseTime’s cluster on Zier 308 during the epidemic, and I ended up in this situation again.
5. In fact, I still encountered some problems. Generally speaking, I was impatient. For example, I observed that the openrlhf process was stuck on DeepSpeedEngine compile, so I would stop and restart it. Later I discovered that it actually took a long time to wait for the first time. Guo Lei After a while, my training got stuck again, this time on vllm broadcast weights. To be honest, I'm a little devastated because I know this broadcast won't take that long. Adjusting to 0.6.5 worked before, but now it doesn't. I reinstalled the environment again because the exact same problem was solved in this way.
6. Still not right, I asked the OpenRLHF author and he said that the vllm update crashed again and the weights update bug appeared again. Only then did I realize that everyone was very anxious. This is the normal state of mlsys... He suggested that I use the stable version of openrlhf instead of main. I switched to 0.5.4 and it still crashed.

Not to mention, I finally got a stable development environment. Tomorrow I will review the PR my friend gave me, and then explain the current situation in the previous PR to OpenRLHF. **Today I downloaded lolm which I had deleted before. Damn it, lolm started. This lolm has llm in it, what a godsend. ** -->

Finally, after solving these problems, I encountered a server explosion while running, and I couldn't even connect to it via ssh. It turns out that ray's logging will default to `tmp/ray`, and this log is so big that it bursts `tmp`. SSH also needs to write things into `tmp`, so it directly destroyed an H100. Thanks to NV who worked overtime on Christmas and replaced it with a new one. In short, the solution to these two bugs together is as follows:```bash
# Specify temp dir when ray starts
ray start --head --node-ip-address 127.0.0.1 --num-gpus 6 --port 1234 --temp-dir="/root/.cache/ray"

# Specify env var when submitting job
export RAY_RUNTIME_ENV='{"env_vars": {"LD_LIBRARY_PATH": "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages/nvidia/cusparse/lib"}}'

ray submit script.py
```### `openrlhf/cli/train_ppo_ray.py`

PPO may be the most important training script of openrlhf in my opinion, and it is also where I mainly tested it before. I record my environment first to avoid environment conflicts in the future:

- openrlhf[vllm]: openrlhf on main, vllm 0.6.5, torch 2.5.1, outlines 0.1.11, ray 2.12.0
- openrlhf[sglang]: openrlhf on main, sglang 0.4.1, torch 2.5.1+cu121, vllm 0.6.4.post1, outlines 0.0.46, ray 2.12.0

It's sad to say that nearly two weeks have passed between the two times I wrote this part of the document. Looking at the benefits, I successfully connected sglang to openrlhf, but the bad news is that the two are far from a stable replacement. On our H100, nccl will often hang on a certain step of deepspeed and fall into a deadlock after a day or two of stable training. This is very anti-human, because the first few epochs will not hang on that step. For example, the backward process will get stuck after reaching 91%. I can't figure it out. In the following document, I first record how to additionally support the sglang engine in addition to the vllm engine in PPO. Then, I will give the derivation step by step and analyze the reasons why I think it may hang. Of course, recently we also asked the core developers of deepspeed to debug nccl hang with us.

Back to PPO, let’s discuss the file changes of my PR here. As for the `train_ppo_ray.py` file itself, the change is actually very small. This file is to change all the variables called `vllm_engines` to the common name of `inference_engines`, and then add the `--backend` parameter.

### `openrlhf/trainer/ppo_utils/experience_maker.py`

Essentially, in RLHF, the inference engine is used to make experience, so this change is quite big.

- **`llm.generate`**

【TODO】

First, the original:```python
llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
```
I changed it to:```python
llm.generate.remote(
    sampling_params=sampling_params, prompt_token_ids=prompt_token_ids, all_prompts=all_prompts
)
```
This is actually the difference between vllm and sglang. In the source code of openrlhf, `prompt_token_ids` is directly passed to the vllm engine. This is probably `input_ids`, and I first implemented the generate prompts passed in to sglang. As I have said, both vllm training engine and sglang training engine will experience unstable efficiency and freeze. I do not believe this is a problem introduced by me. But I do suspect that some subtle differences have a big impact. For example, will there be a difference between the token ids passed in by the vllm engine and the prompts passed by the sglang engine? Will there be some strange tokens added? In addition, the sglang engine needs to be tokenized again, which brings a non-negligible overhead. So here I have three TODOs, implement them all:

1. Directly pass token ids to sglang and do not tokenize prompts again.
2. Print out the beginning and end of tokens to check whether there is any difference in how vllm and sglang handle special tokens.
3. Print out the size of the tokens matrix passed to experience making. Will the difference in matrix size between the two (for example, the longest string is particularly long, resulting in a particularly large difference after padding) have a significant impact?

- **token collection**

This change is not interesting anymore. I manually collected all `input_ids` and `output_ids` to avoid the following inelegant for loops.

<details>
<summary> Changes to list-based reasoning </summary>

Originally:```python

max_input_len, max_output_len = 0, 0
for output in outputs:
    max_input_len = max(max_input_len, len(output.prompt_token_ids))
    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))
```
My changes:```python
input_token_id_list = None
output_token_id_list = None
if backend == "vllm":
    input_token_id_list = [list(output.prompt_token_ids) for output in outputs]
    output_token_id_list = [list(output.outputs[0].token_ids) for output in outputs]
else:
    input_token_id_list = [list(output["input_ids"]) for output in outputs]
    output_token_id_list = [list(output["output_ids"]) for output in outputs]

max_input_len = max(len(input_id) for input_id in input_token_id_list)
max_output_len = max(len(output_id) for output_id in output_token_id_list)
```
</details>

I have to say that the original operation of traversing the list to get the maximum value is indeed not very scientific. Using list derivation is a very basic pythonic operation. I am convinced that this change is completely equivalent. After all, if this equivalent replacement cannot be done well, the four-year code that I studied as an undergraduate will not be able to graduate long ago... However, what puzzles me is that although such a change seems to be equivalent to me, why does nccl hang occur? I haven't tested main. Whether it is main itself is problematic. So, I have another TODO:

- Will the main test also get stuck?

### `openrlhf/trainer/ray/ppo_actor.py`

I have made almost no changes to this file other than naming it. One thing worth noting is that I changed the backend of `init_process_group` from `gloo` to `nccl` because a process group creation error occurred once, but `nccl` is stable as the backend. Could this be the reason for the instability:

- Is the test a backend issue?

### `openrlhf/trainer/ray/vllm_engine.py`

This is the biggest change. The reason is very simple. The rough way to change it is to add branch under this file and select different engines according to the backend. I haven't had time to change the file name yet, so it should be changed to `inference_engine.py`. But these problems can be solved later...

There is no need to change the vllm part, just move it below if. However, a lot of changes are needed in the sglang part. The main reason is that the `__init__` of `LLMRayActor` passes in `*args, **kwargs`, which directly faces the server args of vllm when starting. If I pass it directly to `sglang.Engine`, an error will be reported because the positional parameters do not match. Therefore, I have to find the corresponding parameters of sglang and vllm, but this matter has been done in [batch_inference.py](#openrlhfclibatch_inferencepy), and I suspect there may be a mistake. Here's what I did:

<details>
<summary> Server args mapping from vllm to sglang </summary>

Here are the server parameters for vllm:```python
# Pretrain is model path, the name is strangely abstract
pretrain,
noset_visible_devices=noset_visible_devices,
trust_remote_code=True,
tensor_parallel_size=tensor_parallel_size,
dtype="bfloat16",
seed=seed + i,
enable_prefix_caching=enable_prefix_caching,
enforce_eager=enforce_eager,
max_model_len=max_model_len,
backend=backend,
```This is my mapping in sglang:```python
#! TODO chenyang check engine params
sglang_params = {
    "model_path": args[0],  # pretrain path
    "trust_remote_code": kwargs.get("trust_remote_code", True),
    "dtype": kwargs.get("dtype", "auto"),
    "tp_size": kwargs.get("tensor_parallel_size", 1),
    "device": "cuda",
    "disable_radix_cache": not kwargs.get("enable_prefix_caching", False),
    "random_seed": kwargs.get("seed", 42),
    "disable_cuda_graph": not kwargs.get("enforce_eager", False),
    "disable_cuda_graph_padding": not kwargs.get("enable_prefix_caching", False),
    "context_length": kwargs.get("max_model_len", None),
    "log_level": "info",
    "return_token_ids": True,
}
self.llm = sglang.Engine(**sglang_params)
```
</details>

To be honest, I am quite confident, but I have to check it out. Note that `return_token_ids` is a new feature written specifically for openrlhf. I have to thank [Shuai Shi](https://github.com/shuaills) for this [PR](https://github.com/sgl-project/sglang/pull/2636). This is also my first SGLang PR written by a mentor. It feels very fulfilling, but I actually don’t have many PRs myself. 🤣🤣🤣

Going back to these parameters, there is also `log_level = "info"` which I added because of compassion to see if the inference engine is fully ultized. At present, I have looked at `token usage = 0.61`, which seems to be okay, but Mercy said that we can look at `cache hit rate` and check this later. Here are three more TODOs:

1. Check whether the parameter mapping from vllm to sglang is correct?
2. Similarly, is the sampling params detected correct?
3. Check the cache hit rate. There should be room for improvement in performance.

As mentioned in 2, of course I also have to map the sampling params. In my image, sglang should be completely close to the sampling params written by the openai api, but I still have to check the parameter mapping.

<details>
<summary> Sampling params mapping from vllm to sglang</summary>

```python
if self.backend == "vllm":
    outputs = self.llm.generate(
        sampling_params=kwargs["sampling_params"], prompt_token_ids=kwargs["prompt_token_ids"]
    )
elif self.backend == "sglang":
    # Note that sglang sampling params are different from vllm
    sampling_params = kwargs["sampling_params"]
    all_prompts = kwargs["all_prompts"]

    # min_tokens, include_stop_str_in_output is not used in sglang

    sampling_params = dict(
        max_new_tokens=sampling_params.max_tokens,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        temperature=sampling_params.temperature,
        repetition_penalty=sampling_params.repetition_penalty,
        skip_special_tokens=sampling_params.skip_special_tokens,
    )
    outputs = self.llm.generate(all_prompts, sampling_params)
```
Of course, the sampling params passed in from the front end are as follows:```python
sampling_params = SamplingParams(
    temperature=kwargs.get("temperature", 1.0),
    top_p=kwargs.get("top_p", 1.0),
    top_k=kwargs.get("top_k", -1),
    max_tokens=kwargs.get("max_new_tokens", 1024),
    min_tokens=kwargs.get("min_new_tokens", 1),
    skip_special_tokens=kwargs.get("skip_special_tokens", False),
    include_stop_str_in_output=True,
)
```
</details>

After that, there are `init_process_group` and `update_weight`, which I am really familiar with. Because these two interfaces were written by me, it seems that openrlhf is currently using their own Wrapper, which is not the official code of vllm? It doesn’t matter, here is the code for switching to sglang that I am very familiar with:

<details>
<summary>Relevant code for parameter update </summary>

`init_process_group`:```python
if self.backend == "vllm":
    if self.use_gpu_executor:
        return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
            master_address, master_port, rank_offset, world_size, group_name, backend
        )
    else:
        return self.llm.llm_engine.model_executor._run_workers(
            "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
        )
elif self.backend == "sglang":
    return self.llm.init_weights_update_group(
        master_address, master_port, rank_offset, world_size, group_name, backend="nccl"
    )
```

`update_weight`：

```python
if self.backend == "vllm":
    self.stop_remote_worker_execution_loop()

    if self.use_gpu_executor:
        return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
    else:
        return self.llm.llm_engine.model_executor._run_workers(
            "update_weight", name, dtype, shape, empty_cache
        )
elif self.backend == "sglang":
    return self.llm.update_weights_from_distributed(name, dtype, shape)
```
</details>

In fact, I was confused here, because the training pipeline of sglang would OOM at the beginning. I compared the Wrapper written by openrlhf for vllm, and saw that they would have `del weights` after updating the parameters, but I did not see it in sglang. I thought it was because of the memory leak of sglang. Actually no, Python itself will do this kind of memory recycling within functions. In fact, OOM comes from the deepspeed engine. If I reduce the training batch size, there will be no OOM. This is actually the same conjecture mentioned above. Is it because the token ids matrix given by sglang is different in size, which directly leads to OOM?

### NCCL Hang’s conjecture

As I said before, in my opinion, my modifications are all completely equivalent, and if the sglang engine and the vllm engine work functionally equivalent, there shouldn't be any difference. However, I firmly believe that both frameworks are very stable products after being used by countless users. The difference most likely comes from unequal mappings that I did not notice, especially the mapping of serving params and sampling params. Here is a summary of all my conjectures and TODOs:

1. Directly pass token ids to sglang and do not tokenize prompts again.
2. Print out the beginning and end of tokens and check whether there is any difference in how vllm and sglang handle special tokens.
3. Print out the size of the tokens matrix passed to experience making. Will the difference in matrix size between the two (for example, the longest string is particularly long, resulting in a particularly large difference after padding) have a significant impact?
4. Will the main test also get stuck?
5. Is the test a backend problem?
6. Check whether the parameter mapping from vllm to sglang is correct?
7. Similarly, is the sampling params detected correct?
8. Check the cache hit rate. There should be room for improvement in performance.
9. Test whether it is an environmental problem, or even try another device.
10. The difference between all_prompt_tokens and input token ids in engine outputs.
11. Print the input tensor size and time of each training step, and check why some places are stuck for an hour.

There are so many conjectures, but print can actually verify many of them, so I made a very detailed log and printed directly into the gap between my nails.

## Shooting instructions

### Start ray cluster

<details>
<summary>launch ray</summary>

```bash
al 6

ray stop

ray start --head --node-ip-address 127.0.0.1 --num-gpus 6 --port 1234 --temp-dir=$RAY_TEMP_DIR

pkill -9 -f train_ppo_ray

rm -rf $RLHF_CKPT_DIR/*
```
</details>

### NV 01 100k

<details>
<summary>Using 100k samples for comparison on docker of NV 01</summary>

```bash
# conda activate rlhf-sglang

rlhf-sglang

TIME=$(now)

echo $TIME

ray job submit --address="172.17.0.2:1234" \
--runtime-env-json="{
  \"working_dir\": \"${RLHF_CKPT_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/miniconda3/envs/rlhf-sglang/lib/python3.11/site-packages\"
  }
}" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend sglang \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path ${RLHF_CKPT_DIR}/examples/checkpoint-sglang-$(now)/llama3-8b-rlhf \
   --save_steps 5 \
   --micro_train_batch_size 8 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name sglang-$TIME \
   --wandb_project openrlhf >> ~/log/sglang-$TIME.log
```

```bash
# conda activate rlhf-vllm

rlhf-vllm

TIME=$(now)

echo $TIME

ray job submit --address="172.17.0.3:1234" \
   --runtime-env-json='{
     "working_dir": "/root/rlhf-ckpt",
     "env_vars": {
       "PYTHONPATH": "/root/miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages"
     }
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend vllm \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /root/rlhf-ckpt/examples/checkpoint-vllm-$(now)/llama3-8b-rlhf \
   --save_steps 5 \
   --micro_train_batch_size 8 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_project openrlhf \
   --wandb_run_name vllm-$TIME >> ~/log/vllm-$TIME.log
```
</details>

### NV 02 100k

<details>
<summary>Use 100k samples directly on NV 02 for comparison</summary>

```bash
# conda activate rlhf-sglang

rlhf-sglang

TIME=$(now)

echo $TIME

ray job submit --address="172.31.59.18:1234" \
   --runtime-env-json='{
     "working_dir": "/opt/dlami/nvme/chenyang/rlhf-ckpt",
     "env_vars": {
       "PYTHONPATH": "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-sglang/lib/python3.11/site-packages"
     }
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend sglang \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /opt/dlami/nvme/chenyang/rlhf-ckpt/examples/checkpoint-sglang-$(now)/llama3-8b-rlhf \
   --save_steps 5 \
   --micro_train_batch_size 8 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name sglang-$TIME \
   --wandb_project openrlhf >> ~/log/sglang-$TIME.log
```

```bash
# conda activate rlhf-vllm

rlhf-vllm

TIME=$(now)

echo $TIME

ray job submit --address="172.31.59.18:1234" \
   --runtime-env-json='{
     "working_dir": "/opt/dlami/nvme/chenyang/rlhf-ckpt",
     "env_vars": {
       "PYTHONPATH": "/opt/dlami/nvme/chenyang/.miniconda3/envs/rlhf-vllm/lib/python3.11/site-packages"
     }
   }' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend vllm \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /opt/dlami/nvme/chenyang/rlhf-ckpt/examples/checkpoint-vllm-$(now)/llama3-8b-rlhf \
   --save_steps 5 \
   --micro_train_batch_size 8 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_project openrlhf \
   --wandb_run_name vllm-$TIME >> ~/log/vllm-$TIME.log
```
</details>

### NV 01 512

<details>
<summary>Single test using 512 samples on docker of NV 01</summary>

```bash
rlhf-sglang

TIME=$(now)

echo $TIME

ray job submit --address="172.17.0.2:1234" \
--runtime-env-json="{
  \"working_dir\": \"${RLHF_CKPT_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/miniconda3/envs/rlhf-sglang/lib/python3.11/site-packages\"
  }
}" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend sglang \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path ${RLHF_CKPT_DIR}/examples/checkpoint-sglang-$(now)/llama3-8b-rlhf \
   --save_steps 5 \
   --micro_train_batch_size 8 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 128 \
   --max_samples 512 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name sglang-$TIME \
   --wandb_project openrlhf >> ~/log/sglang-$TIME.log
```
</details>

### Hyperbolic 100K

<details>
<summary>Testing of Hyperbolic 100K</summary>

```bash
rlhf-sglang

TIME=$(now)

echo $TIME

ray job submit --address="172.27.13.23:1234" \
--runtime-env-json="{
  \"working_dir\": \"${RLHF_CKPT_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"/data/chayenne/miniconda3/envs/rlhf-sglang/lib/python3.11/site-packages\"
  }
}" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend sglang \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path ${RLHF_CKPT_DIR}/examples/checkpoint-sglang-hyperbolic-$(now)/llama3-8b-rlhf \
   --save_steps 5 \
   --micro_train_batch_size 8 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name sglang-hyperbolic-$TIME \
   --wandb_project openrlhf >> ~/log/sglang-hyperbolic-$TIME.log
```

```bash
rlhf-sglang

TIME=$(now)

echo $TIME

ray job submit --address="172.27.13.23:1234" \
--runtime-env-json="{
  \"working_dir\": \"${RLHF_CKPT_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"/data/chayenne/miniconda3/envs/rlhf-sglang/lib/python3.11/site-packages\"
  }
}" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend vllm \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path ${RLHF_CKPT_DIR}/examples/checkpoint-vllm-hyperbolic-$(now)/llama3-8b-rlhf \
   --save_steps 5 \
   --micro_train_batch_size 8 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name vllm-hyperbolic-$TIME \
   --wandb_project openrlhf >> ~/log/vllm-hyperbolic-$TIME.log
```
</details>

### Hyperbolic 100K default parameters

<details>
<summary> Hyperbolic 100K default parameters </summary>

Default parameters on main```bash
rlhf-sglang

TIME=$(now)

echo $TIME

ray job submit --address="172.27.13.23:1234" \
--runtime-env-json="{
  \"working_dir\": \"${RLHF_CKPT_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"/data/chayenne/miniconda3/envs/rlhf-sglang/lib/python3.11/site-packages\"
  }
}" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path ${RLHF_CKPT_DIR}/examples/checkpoint-vllm-$(now)/llama3-8b-rlhf \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_project ppo-dev \
   --wandb_run_name vllm-$TIME >> ~/log/vllm-$TIME.log
```
Default parameters on dev pr```bash
rlhf-sglang

TIME=$(now)

echo $TIME

ray job submit --address="172.27.13.23:1234" \
--runtime-env-json="{
  \"working_dir\": \"${RLHF_CKPT_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"/data/chayenne/miniconda3/envs/rlhf-sglang/lib/python3.11/site-packages\"
  }
}" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --backend sglang \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path ${RLHF_CKPT_DIR}/examples/checkpoint-sglang-$(now)/llama3-8b-rlhf \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_project ppo-dev \
   --wandb_run_name sglang-$TIME >> ~/log/sglang-$TIME.log
```
</details>

## Debug NCCL Hang
