# veRL-SGLang：Train Multimodal Model with Multi-Turn RL to Reason and Call Tool

Hello everyone, members of the SGLang community, Amazon SF AGI Lab and PolyU have added multi-modal support to the previously open source [multi-turn RL](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/release_log/verl-multiturn-rollout-Release_ZH.md). Everyone is welcome to experience it and develop together. Specifically, we implemented the following functions:

- The SGLang community has previously implemented tool calling, allowing the model to call specific tools during Actor rollout and seamlessly integrate the returned results into the training process.
- Now we have further added multi-modal inputs to Multi-Turn RL, allowing the model to handle multi-modal data during the actor rollout stage.
- We are adding support for tool-generated images, so stay tuned

[PR: volcengine/verl#2014](https://github.com/volcengine/verl/pull/2014)

[Training curve wandb](tbd)

Project Member:

- Nan Jiang, Congkai Xie (Author)
- Chenyang Zhao (PM)
- Xiang Long (Reviewer, PM)

Thanks for the contribution!

## Quick reproduction

### Create a new Docker container
```bash
docker run \
    -it \
    --shm-size 32g \
    --gpus all \
    -v {Huggingface-Cache-Path}:/root/.cache \
    --ipc=host \
    --network=host \
    --privileged \
    --name sglang_{your-name} \
    lmsysorg/sglang:dev \
    /bin/zsh
```
If you need to restart the container after exiting it:
```bash 
docker start -i sglang_{your-name}
```
### Update Python and configure virtual environment using uv
```bash
apt update
apt install -y python3.10 python3.10-venv

# Create virtual environment
python3 -m venv ~/.python/veRL-multiturn-rollout

# Activate virtual environment
source ~/.python/veRL-multiturn-rollout/bin/activate

# Install uv
python3 -m pip install uv
```
### Install veRL upstream

```bash
cd ~
git clone -b feat/add-multimodal-multiturn-sglang https://github.com/nanjiangwill/verl.git
cd verl

# install verl
python3 -m uv pip install .
python3 -m uv pip install -r ./requirements_sglang.txt

# Manually install flash-attn
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
```

### Set WANDB_API_KEY

If you don’t understand how to obtain the API Key, please refer to [here](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914).

```bash
export WANDB_API_KEY={YOUR_WANDB_API_KEY}

#Define timestamp function
function now() {
    date '+%Y-%m-%d-%H-%M'
}
```
### **Preprocessing Dataset**

Note that the following data processing and training commands need to be performed in the veRL-multiturn-rollout execution environment.

```bash 
python3 examples/data_preprocess/geo3k_multiturn_w_tool.py
```
### Tested on 8 X H20

```bash
# Make sure the now() function is defined
#Create log directory
mkdir -p logs

# Set up GPU and run, using appropriate log path
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup bash examples/sglang_multiturn/geo3k/run_qwen2.5-3b_geo3k_multiturn.sh trainer.experiment_name=qwen2.5-3b-it_rm-geo3k-sgl-multiturn-$(now) > logs/geo3k$(now).log 2>&1 &
```
## Notes