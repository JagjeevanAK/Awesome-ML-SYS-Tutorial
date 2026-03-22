# Enable verl's agent loop feature

<!-- In our earliest release of [multi-turn RL](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/release_log/verl-multiturn-rollout-Release_ZH.md), our tool call state management was managed inside SGLang rollout. Although it has been recognized by a large number of community users, due to the integration of rollout and tool call management, it is not completely easy to maintain in the long run. In addition, in our original design, each step of multi-turn will be called once [`_preprocess_prompt_to_async_rollout_requests`](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme-2.md#_preprocess_prompt_to_async_rollout_requests) to do preprocessing, and this part of preprocessing is actually the same as step is irrelevant. The agent loop feature separates the management of tool calls from the rollout engine. The rollout engine only provides an interface for token in token out. The specific code analysis will be in the second half of this article; the first half will introduce how to enable the agent loop function. -->


## Quick Start

To put it simply, you only need to modify two configurations to enable the agent loop feature:

1. Add `actor_rollout_ref.rollout.mode=async` to the bash script that starts training and ensure `actor_rollout_ref.rollout.multi_turn.enable=true`;
2. Add a new column `agent_name` to the data set in the data set processing script and add it in `map_fn`.

We next provide a step-by-step reproduction process: this will rely on the latest verl and the latest version of sglang. Note that although verl still relies on sglang 0.4.6.post5 in [`setup.py`](https://github.com/volcengine/verl/blob/main/setup.py), this is because the transformers dependency in verl is blocked by the bug block of qwen2.5 vl in the new version of flash-attn. Verl itself can already enable a more advanced version of sglang, and you can also use the powerful feature of [multi-turn wake up](https://hebiao064.github.io/rl-memory-management).### Create a new docker

If you can run verl stably locally, then there is a high probability that you do not need a new docker. Using docker is only convenient for us to ensure that the experiment can be strictly reproduced.

You need to configure `WANDB_API_KEY` before use, refer to [this process](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914).

```bash
# If your system has not been configured with HF_TOKEN and WANDB_API_KEY, please configure it first
docker run -it --name verl_{your_name} --gpus all \
    --shm-size 32g \
    -v {your_cache_path}:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```
After entering docker, check the mapped environment variables:

```bash
echo $HF_TOKEN
echo $WANDB_API_KEY
```
In the future, every time you exit from docker, you can use this command to restart:

```bash
docker start -i verl_{your_name}
```
### Install SGLang based on source code

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
Install veRL first, then SGLang:

```bash
cd ~
git clone https://github.com/volcengine/verl.git
cd verl
python3 -m uv pip install -e ".[sglang,geo]"
python3 -m uv pip install -r ./requirements.txt
```
If you encounter this error:

```bash
Using Python 3.10.12 environment at: /root/.python/verl-sglang
  × No solution found when resolving dependencies:
  ╰─▶ Because there is no version of flashinfer-python==0.2.9rc2 and sglang[srt]==0.4.9.post6 depends on flashinfer-python==0.2.9rc2, we can conclude that sglang[srt]==0.4.9.post6 cannot be used.
      And because verl[sglang]==0.5.0.dev0 depends on sglang[srt]==0.4.9.post6, we can conclude that verl[sglang]==0.5.0.dev0 cannot be used.
      And because only verl[sglang]==0.5.0.dev0 is available and you require verl[sglang], we can conclude that your requirements are unsatisfiable.
```
Click this cmd fix:

```bash
python3 -m uv pip install --prerelease=allow -e ".[sglang,geo]"
```
During this process, the flash-attn installation will encounter this error:

```bash
Resolved 130 packages in 1.96s
  × Failed to build `flash-attn==2.8.1`
  ├─▶ The build backend returned an error
  ╰─▶ Call to `setuptools.build_meta:__legacy__.build_wheel` failed (exit status: 1)
```
Just follow the steps below to fix it:

```bash
python3 -m uv pip install wheel
python3 -m uv pip install -r ./requirements.txt --no-build-isolation
```
Then install SGLang upstream:

```bash
cd ~
git clone https://github.com/sgl-project/sglang.git
cd sglang
python3 -m uv pip install --upgrade pip
python3 -m uv pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python
```
Additional installation of vllm and weave dependencies for visualization:

```bash
python3 -m uv pip install vllm==0.9.1
python3 -m uv pip install weave
```
### Modify and run

We can make simple modifications to the existing script, enable `multi_turn` and `async rollout` in the running script, and add an `agent_name` column in `def make_map_fn(split)` in the dataset processing script.

Open the `~/verl/examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh` file in your docker, remove the `$@` at the end of the line, and change the following parameters:

```bash
    # Note to remove the $@ at the end of the original total_epochs line
    # Do not write these two lines of comments, otherwise an error will be reported
    trainer.total_epochs=15 \
    actor_rollout_ref.rollout.trace.backend=weave \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=true
```

Append `"agent_name": "tool_agent"` in `~/verl/examples/data_preprocess/gsm8k_multiturn_w_tool.py`

```python
def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                # new column for weave trace
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        #...
                    }
                ]
            }
            return data

        return process_fn
```
Just test it next:

```bash
cd ~/verl
python3 -m uv pip install .
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Pull and preprocess the gsm8k data set
python examples/data_preprocess/gsm8k_multiturn_w_tool.py
```Just start the 8-card workout.```bash
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```
### Debug

- If you get this error after starting bash:

```bash
raise ValueError(f"Feature type '{_type}' not found. Available feature types: {list(_FEATURE_TYPES.keys())}")
ValueError: Feature type 'List' not found. Available feature types: ['Value', 'ClassLabel', 'Translation', 'TranslationVariableLanguages', 'LargeList', 'Sequence', 'Array2D', 'Array3D', 'Array4D', 'Array5D', 'Audio', 'Image', 'Video', 'Pdf' ]
```
In fact, this is not an actual error report. This error report puzzled me for a very, very long time. I looked at the log carefully and found the problem. In fact, I can look up a few lines of the error report. At the very beginning of the error stack, the python environment where the error was reported is `/root/.python/verl-sglang/lib/python3.10`, and at the bottom of the stack it becomes `/usr/local/lib/python3.10`. There is no doubt that the python environment is misplaced;

1. The main process uses a virtual environment: `/root/.python/verl-sglang/lib/python3.10/site-packages/`
2. The Ray worker process uses system Python: `/usr/local/lib/python3.10/dist-packages/`

The final solution to this problem is to modify the `verl/trainer/constants_ppo.py` file directly to:

```python
import os
importsys

# Get the current Python interpreter path and virtual environment path
python_executable = sys.executable
virtual_env = os.environ.get("VIRTUAL_ENV", "")
python_path = os.environ.get("PYTHONPATH", "")

# If you are currently in a virtual environment, make sure to include the virtual environment's site-packages
if virtual_env:
    site_packages = os.path.join(virtual_env, "lib", "python3.10", "site-packages")
    if site_packages not in python_path:
        python_path = f"{site_packages}:{python_path}" if python_path else site_packages

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        #Add Python environment configuration
        "PYTHONPATH": python_path,
        "VIRTUAL_ENV": virtual_env,
    },
    #Specify Python interpreter
    "python": python_executable,
}
```
- If you encounter the following error:
```bash
File "/root/.python/verl-sglang/lib/python3.12/site-packages/triton/runtime/driver.py", line 8, in _create _driverraise RuntimeError(f"flen(actives)) active drivers ( factives,). There should only be one."RuntimeError: 0 active drivers ([]). There should only be one(MorkerDict pid-319609) MARMING 07-25 04:31:15 (en override.py:17) WCCL CUMEM EMABLE is set to 0, skipping override. This may increase menory overhead with cudagraph+allreduce: https://github.con/WVIDIA/nccl/issues/1234 (repeated 5x across cluster)
```
Please downgrade the triton version ([Reference link](https://github.com/triton-inference-server/server/issues/8007)):

```bash
uv pip install triton==3.1.0
```
## Code-Walk-Through

