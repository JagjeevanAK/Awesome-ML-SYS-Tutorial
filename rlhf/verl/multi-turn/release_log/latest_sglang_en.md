# Quickly set up the latest verl-sglang

This document documents how to quickly install the latest verl-sglang.

1. Create a new docker (you can skip this if you are familiar with this installation):

You need to configure `WANDB_API_KEY` before use, refer to [this process](https://community.wandb.ai/t where-can-i-find-the-api-token-for-my-project/7914).

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
mkdir ~/.python
python3 -m venv ~/.python/verl-sglang
source ~/.python/verl-sglang/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade uv
```
3. Install verl-sglang:
 
```bash
cd ~
git clone https://github.com/volcengine/verl.git
cd verl

python3 -m uv pip install -e ".[sglang]" --prerelease=allow
python3 -m uv pip install -r ./requirements.txt --no-build-isolation
python3 -m uv pip install torch_memory_saver
# to avoid vllm registration error with transformers 4.54.0, install it manually
python3 -m uv pip install vllm==0.10.0 --no-build-isolation
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
5. Test geo3k:
```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m uv pip install qwen-vl-utils
python3 -m uv pip install mathruler

# Pull and preprocess geo3k dataset
python examples/data_preprocess/geo3k.py --local_dir ~/data/geo3k

# Start 8-card training
bash examples/grpo_trainer/run_qwen2_5_vl-7b-sglang.sh
```
6. Test dapo:

Note that dapo is different from the previous settings because dapo requires another docker to start sandboxfusion, so you need to go back to the host and start the tool server separately:

```bash
#Start sandbox fusion (dapo tool call requirement)
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
```
In addition, in order to allow the dapo training script to be executed in a separate docker and to be able to access the 8080 port of sandboxfusion on the host, we need to add an additional `--network=host` when starting docker, that is, the startup command in step 1 needs to be changed to:

```bash
docker run -it --name h100_verl_{your_name} --gpus all \
    --shm-size 32g \
    -v {your_cache_path}:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --network=host \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```

After that, follow the installation process of steps 2 and 3. After the installation is completed, start training:
```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Start 8-card training
bash examples/sglang_multiturn/run_qwen3_4b_dapo_multiturn.sh
```## Debug

If you get this error after starting bash:```bash
raise ValueError(f"Feature type '{_type}' not found. Available feature types: {list(_FEATURE_TYPES.keys())}")
ValueError: Feature type 'List' not found. Available feature types: ['Value', 'ClassLabel', 'Translation', 'TranslationVariableLanguages', 'LargeList', 'Sequence', 'Array2D', 'Array3D', 'Array4D', 'Array5D', 'Audio', 'Image', 'Video', 'Pdf']
```

In fact, this is not an actual error report. This error report puzzled me for a very, very long time. I looked at the log carefully and found the problem. In fact, I can look up a few lines of the error report. At the very beginning of the error stack, the python environment where the error was reported is `/root/.python/verl-sglang/lib/python3.10`, and at the bottom of the stack it becomes `/usr/local/lib/python3.10`. There is no doubt that the python environment is misplaced;

1. The main process uses a virtual environment: `/root/.python/verl-sglang/lib/python3.10/site-packages/`
2. The Ray worker process uses system Python: `/usr/local/lib/python3.10/dist-packages/`

The final solution to this problem is that it is recommended not to use a virtual environment, ray...