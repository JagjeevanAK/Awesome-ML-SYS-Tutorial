# FSDP Setup Guide

This document records how to test FSDP on slime, including H card and B card, and two placement methods: Colocate and Disaggregated. The following operations are completed on the H card:

## Quick Start

### Pull and start the Docker container

```shell
# Pull the latest image
# The latest image is for B card and H card
docker pull slimerl/slime:latest

# Start container
docker run -d --gpus all --ipc=host --shm-size=16g \
  --name slime_wren_fsdp \
  -it slimerl/slime:latest /bin/bash
```

### Install slime

Once inside the Docker container, follow these steps to clone the slime repository and install it:

```bash
# The path can be adjusted according to the actual situation
cd /root/
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .
```

### Model and data set download

The required models and datasets can be downloaded from platforms such as Hugging Face, ModelScope, etc. Here is the command to download the sample resources using `huggingface_hub`:

```bash

pip install -U huggingface_hub

# Download model weights (Qwen3-0.6B)
hf download Qwen/Qwen3-0.6B --local-dir /root/Qwen3-0.6B

# Download training data set (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# Download evaluation dataset (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

### Load the configuration file of the target model

First, load the configuration file of the target model. The `slime/scripts/models` directory contains configuration files that support models. A script corresponding to the model is required to load the configuration parameters into the current environment. Here we take the Qwen3-0.6B model as an example. It is similar for Qwen3-4B and Qwen3-30B-A3B.

```bash
cd /root/slime
source scripts/models/qwen3-0.6B.sh 
```
### Overview of training scripts and parameters

After completing the above preparations, you can run the training script.

```bash
cd /root/slime
bash tests/test_qwen3-0.6B_fsdp_colocated_2xGPU.sh # 2GPU collaborative training test
```

## Feature introduction

### Colocated Actor and Rollout

Under the default configuration, the resources for training (Actor) and inference (Rollout) are specified separately. `actor_num_nodes * actor_num_gpus_per_node` GPUs are allocated to the training part through `ray`, and `actor_num_nodes * rollout_num_gpus` GPUs are allocated to the reasoning part, that is, training and push separation.

**Standard (detached) configuration**:

```bash
ray job submit ... \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ...
```
In the above configuration, `Actor` uses 4 cards, `Rollout` also uses 4 cards, and both run in parallel.


> When training and pushing are separated, the training and inference GPUs always wait for each other. In order to avoid such idle resources, we can enable asynchronous training. The way to enable it is to change `train.py` in the startup script to `train_async.py`. In this way, slime will generate data for the next rollout while training the current rollout.

> ⚠️ During asynchronous training, sglang's performance testing logs and training logs may be mixed together and difficult to distinguish. You can reduce sglang's logs through `--sglang-log-level`.

**Colocated configuration**:

To deploy training and inference on the same set of GPUs, add the `--colocate` parameter. When enabled, `--rollout-num-gpus` will be ignored to make the number of cards for training and inference equal.

```bash
ray job submit ... \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ...
```
At this point, training and inference share all 8 GPUs.

### FSDP activation mechanism

```bash
#Key: Specify the backend as FSDP
"SLIME_BACKEND": "fsdp"
```

GPU sharding settings:
```bash
export CUDA_VISIBLE_DEVICES=1,2 # Use GPU 1,2
--actor-num-gpus-per-node 2 # 2 GPUs for model sharding
```
FSDP mode selection:

```bash
--fsdp-full-params # Enable FULL_STATE_DICT mode
# Comment out to use the default SHARDED_STATE_DICT mode
```

## Blackwell GPU Settings

### Start Docker

```shell
# Pull the latest image. The latest image is for B card and H card.
docker pull slimerl/slime:latest

# Start container
# The GPU related parameters here are exactly the same, the main difference is the image version and mounting directory (these are environment configurations, not hardware differences)
docker run \
      -itd\
      --shm-size 32g \
      --gpus all \
      --ipc=host \
      --network=host \
      --privileged \
      -v {your_cache_path}:/root/.cache \
      --name slime_fsdp_{your_name} \
      slimerl/slime:latest \
      /bin/bash
```
The remaining steps are exactly the same as the H card operation steps.

> If you encounter an error with `nccl`, you can specify a port when starting `ray`:
```shell
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --port 9987
```
