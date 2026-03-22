# Qwen2.5VL GRPO with SGLang

## Environment configuration

### Create a new docker

You need to configure `WANDB_API_KEY` before use, refer to [this process](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914).

```bash
# If your system has not been configured with HF_TOKEN and WANDB_API_KEY, please configure it first
# The cache mapping path here is on the atlas cluster. If you need to use your own path, please modify it yourself.
docker run -it --name h100_verl_{your_name} --gpus all \
    --shm-size 32g \
    -v /.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```After entering docker, you can view the mapped environment variables:```bash
echo $HF_TOKEN
echo $WANDB_API_KEY
```
In the future, every time you exit from docker, you can use this command to restart:

```bash
docker start -i h100_verl_{your_name}
```
### Install SGLang based on source code

Configure python environment

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
Install veRL first, then SGLang.
```bash
cd ~
git clone https://github.com/volcengine/verl.git
cd verl
python3 -m uv pip install -e ".[sglang,geo]"
python3 -m uv pip install -r ./requirements.txt
```
> If you encounter this error:
```sh
ModuleNotFoundError: No module named 'torch'

hint: This error likely indicates that `flash-attn@2.7.4.post1` depends on `torch`, but doesn't declare it as a build dependency. If
`flash-attn` is a first-party package, consider adding `torch` to its `build-system.requires`. Otherwise, `uv pip install torch` into the
environment and re-run with `--no-build-isolation`.
```
> Follow the steps below to fix
```sh
python3 -m uv pip install wheel
python3 -m uv pip install -r ./requirements.txt --no-build-isolation
```
After installing SGLang, in order to align the torch version.
```bash
cd ~
git clone https://github.com/sgl-project/sglang.git
cd sglang
python3 -m uv pip install --upgrade pip
python3 -m uv pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python
```
Additional installation of `qwen-vl` dependencies:

```sh
uv pip install qwen_vl_utils
```
## 8 card starts the Qwen2.5VL GRPO training script and uses SGLang as the rollout engine

```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Pull and preprocess geo3k dataset
python examples/data_preprocess/geo3k.py --local_dir ~/data/geo3k
```Open the `~/verl/examples/grpo_trainer/run_qwen2_5_vl-7b.sh` file in your docker and remove the `$@` at the end of examples/grpo_trainer/run_qwen2_5_vl-7b.sh

After the modification is completed, start the 8-card training```bash
bash examples/grpo_trainer/run_qwen2_5_vl-7b.sh sglang
```
