## How to give verl bump a new version of sglang
> This article mainly talks about my process and experience in bumping sglang to verl version from 0.4.9 to 0.4.10post2. The problems encountered and solved in each version update are not exactly the same, for reference

Previous PR: [bump sglang to 0.4.9](https://github.com/volcengine/verl/pull/2720)

### Operation process
1. First check the torch version corresponding to the new version of sglang [link](https://github.com/sgl-project/sglang/blob/v0.4.10.post2/python/pyproject.toml#L58)
2. Create a new docker file in the corresponding folder
For example `docker/verl0.5-cu126-torch2.7-fa2.7.4/Dockerfile.app.sglang0.4.10.post2.mcore0.12` Pay attention to the transformer version, torch version and sglang version

```bash
# Start from the verl base image
#Dockerfile.base
FROM verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.1-fa2.7.4

# Define environments
ENV MAX_JOBS=8
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

#Install sglang-0.4.10
#Install FlashInfer Python package
RUN pip install --upgrade pip setuptools packaging
RUN pip install --resume-retries 999 --no-cache-dir --no-build-isolation flashinfer-python==0.2.9rc1
RUN pip install --resume-retries 999 --no-cache-dir --no-build-isolation "sglang[all]==0.4.10.post2"

# Fix packages
RUN pip install --no-cache-dir "tensordict==0.6.2" "transformers[hf_xet]==4.54.1" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=19.0.1" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler blobfile xgrammar \
    pytest py-spy pyext pre-commit ruff

RUN pip uninstall -y pynvml nvidia-ml-py && \
    pip install --resume-retries 999 --no-cache-dir --upgrade "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

RUN pip install --resume-retries 999 --no-cache-dir nvidia-cudnn-cu12==9.8.0.87

#Install TransformerEngine
RUN export NVTE_FRAMEWORK=pytorch && pip3 install --resume-retries 999 --no-deps --no-cache-dir --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@v2.2.1

# Install Megatron-LM
RUN pip3 install --no-deps --no-cache-dir --no-build-isolation git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2

# Install mbridge
RUN pip3 install --no-cache-dir mbridge
```
3. Use this docker file build image
Note that it cannot be built in a docker container. Docker related operations are completed in ubuntu (bare metal):
3.1 First log in to your docker
```bash
docker login -u [your_username]
```
3.2 Build your own image again
```bash
docker build -t popsodazhp/verl:app-verl0.5-sglang0.4.10.post2-mcore0.12.2-te2.2 -f docker/verl0.5-cu126-torch2.7-fa2.7.4/Dockerfile.app.sglang0.4.10.post2.mcore0.12 .
```

4. Next try to push it up
```bash
docker push popsodazhp/verl:app-verl0.5-sglang0.4.10.post2-mcore0.12.2-te2.2
```
5. Modify CI configuration file
Such as this [commit](https://github.com/volcengine/verl/pull/3183/commits/8ebbea9a0206ce4e258b1cde2eb025b99139b13b)
Modify the image in CI to the one built above, and observe the CI pass status on github. If it basically passes, it means that there is basically no problem with the image built by this new version.
6. Discuss and negotiate with the verl people, let them build the official Byte image, and discuss the details~