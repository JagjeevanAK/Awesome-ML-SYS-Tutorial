# How to configure a refreshing development machine

The day before yesterday, a good friend of mine criticized me: "You are like 98% of men in the world. Whenever you see beautiful things, you want to keep them for yourself instead of appreciating the beauty."

I think what she said is right, but I just want to ask, how many men can handle it after seeing this?

<img src="./H200.png" width="70%" alt="H200">


Anyway, I couldn’t hold it any longer, and finally I was lucky enough to touch the H200 made of gold. This note simply records my environment configuration experience after countless pitfalls over the past few months in order to develop SGLang for RLHF. The world has been suffering from conda for a long time. I tried to manage the virtual environment completely based on native python, combined with uv package management. I hope it can help readers get started with zero frames and configure a refreshing development machine.

## Configure bash/zsh

When starting any cluster, I would recommend configuring bash/zsh first. The biggest advantage is that before any installation, we can determine all data paths to avoid data being written to `/root` or `/home` and other directories shared by the cluster. As far as I know, the data paths of most development clusters are not `/home` or `/root`. **If you write a large amount of content to these two paths and occupy all the disks, it will cause `/root/tmp` or `/home/tmp` to be unable to be written. SSH login to the cluster requires writing data to these two key `tmp` directories, so writing data to these two directories will cause ssh login failure and the cluster must be returned to the factory for repair. **

Here I would like to share a set of configurations that I like, for reference:

<details>
<summary>I like to use .bashrc files, the same goes for zsh</summary>


## Git related

```bash
#Create new branch
alias gcb="git checkout -b"

# Submit commit
alias gcm="git commit --no-verify -m"

# switch branch
alias gc="git checkout"

# Push the newly created local branch to the remote end
alias gpso='git push --set-upstream origin "$(git symbolic-ref --short HEAD)"'

# Push the local branch to the remote end
alias gp="git push"

# View local branch
alias gb="git branch"

# Pull the remote branch
alias gpl="git pull --no-ff --no-edit"

#Add all files
alias ga="git add -A"

# Set up remote branch
alias gbst="git branch --set-upstream-to=origin/"

# View commit tree
alias glg="git log --graph --oneline --decorate --abbrev-commit --all"

# View commit table
alias gl="git log"
````

```bash
## python related

# run python
alias py="python"

# Run pip. Note that this pip uses the forced alignment of the current python environment, which can avoid many pitfalls.
alias pip="python -m pip"

# run ipython
alias ipy="ipython --TerminalInteractiveShell.shortcuts '{\"command\":\"IPython:auto_suggest.resume_hinting\", \"new_keys\": []}'"
```

```bash
## Devlop Requirements

# Run pre-commit
alias pre="pre-commit run --show-diff-on-failure --color=always --all-files"

# View the disk space usage under the current path in a human-readable format
alias duh="du -hs"

# Check disk space usage

alias dfh="df -h"

#Install packages in the current directory
alias pi="pip install ."

# View files in the current path
alias le="less"

# View historical commands
alias his="history"


# View the file tree under the current path
alias tr="tree -FLCN 2"

# View the folders under the current path
alias trd="tree -FLCNd 2"

# Read the disk space usage under the current path every 0.1 seconds, especially useful when downloading models
alias wd="watch -n 0.1 du -hs"

# To read the tail content of the file in a streaming manner, you can submit the task in tmux and then exit tmux. Use this command on the command line to view the log of the task.
alias tf="tail -f"


#Set file permissions to fully transparent
alias c7="chmod 777 -R"

#Open the current path
alias op="open ."

# Use cursor to open a file, use vscode for the same reason
alias cur="cursor"

# Open a file with vscode
alias cod="code"

#Open configuration file
alias ope="cursor /home/chenyang/.zshrc"

# Use cursor to open the file in the current path
alias co="cursor ."

# Quickly check GPU usage
alias nvi="nvidia-smi"

# To check GPU usage every 1 second, you need to pip install gpustat first
alias gpu="watch -n 1 gpustat"

#Create tmux session
alias tns="tmux new -s"

# List tmux sessions
alias tls="tmux ls"

# Log back in to a tmux session
alias tat="tmux attach -t"

# Reload zsh configuration
alias sz="source ~/.zshrc"

# Reload bash configuration
alias zb="source ~/.bashrc"

# Kill process
alias k9="kill -9"

# Format code
alias bp="black *.py && black *.ipynb"

# Kill all python processes under your own name, use with caution
alias kp="ps aux | grep '[p]ython' | awk '{print \$2}' | xargs -r kill -9"

# Delete the output of ipynb file
alias nbs='find . -name "*.ipynb" -exec nbstripout {} \;'

# Use soft mode to reset git commit
alias grs="git reset --soft"

# Set the token of huggingface

export HF_TOKEN="*******************"

# Set the cache path of huggingface. Please configure it properly to avoid a cluster from downloading a certain model multiple times.

export HF_DATASETS_CACHE="/data/.cache/huggingface/datasets"
export HF_HOME="/data/.cache/huggingface"

#Set a personal default path. I usually put all data together under /data.
export HOME="/data/chenyang"

# Set the cache path of ray. If you don’t use ray, you don’t need to worry about it.
export RAY_ROOT_DIR="/data/.cache/ray"

# Set the api key of wandb
export WANDB_API_KEY="************************"

# Set LD_LIBRARY_PATH. This is a pitfall when configuring flash attention. If you encounter problems, you can refer to it.
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Function used to allocate graphics cards, al k can allocate k free cards,

function al() {
    local num_gpus=$1
    
    echo "Looking for $num_gpus free GPUs..."
    echo "Checking GPU memory usage:"
    nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits
    
    local gpu_ids=$(nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits |
                   awk -F, '{
                       gsub(/ /, "", $1);
                       gsub(/ /, "", $2);
                       if ($1 + 0 < 100) print $2
                   }' |
                   head -n $num_gpus)
    
    local found_count=$(echo "$gpu_ids" | wc -l)
    

    if [ $found_count -eq $num_gpus ]; then
        gpu_ids=$(echo "$gpu_ids" | paste -sd "," -)
        export CUDA_VISIBLE_DEVICES=$gpu_ids
        echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
        return 0
    else
        echo "Error: Requested $num_gpus GPUs but only found $found_count free GPUs"
        return 1
    fi
}

## Activate virtual environment

function ca() {
    env_name="$1"
    #Construct the path to the activation script based on the environment name
    activation_script="$HOME/.python/${env_name}/bin/activate"
    
    # Check if activation script exists
    if [ ! -f "$activation_script" ]; then
        echo "Error: Activation script '$activation_script' does not exist"
        return 1
    fi

    # Activate virtual environment
    # Note: After activation using source, the current shell environment will be modified.
    source "$activation_script"
    
    ceiling="===== Activated Env: ${env_name} ====="
``````bash
    echo "$ceiling"
    
    # Output the current python path and version
    python_path=$(which python)
    echo "Python path: $python_path"
    python --version

    # If you want to check whether the environment switch is successful, you can consider checking the environment variables
    # For example, the VIRTUAL_ENV variable is usually set in a virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "===== NO!! Environment switch failed ====="
        return 1
    else
        echo "====== YES!! Environment switched to: $VIRTUAL_ENV ====="
    fi
}


# Read the timestamp, used to mark the log

function now() {
    date '+%Y-%m-%d-%H-%M'
}

#My personal work path

alias sgl="cd /data/chenyang/sglang/python"
alias rlhf="cd /data/chenyang/OpenRLHF-SGLang/openrlhf"
alias vllm="cd /data/chenyang/vllm/"
alias docs="cd /data/chenyang/sglang/docs"
alias test="cd /data/chenyang/sglang/test"
alias awe="cd /data/chenyang/Awesome-ML-SYS-Tutorial"

#uv related

# View the current virtual environment
alias uvv="uv venv"

## Allocate 1 GPU
al 1

## Activate sglang environment
ca sglang
sleep 1
clear
```
</details>

## Install uv

uv is a more modern python package manager that can completely replace conda. Lightweight, convenient, fast and powerful, it is the future trend.

First, log in to a new cluster account. At this moment, you will use the python environment shared by the entire cluster. This environment cannot be modified for relatively secure cluster management, so we need to create our own virtual environment to install uv.```bash
# Create virtual environment
python3 -m venv ~/.python/sglang

# Activate virtual environment
source ~/.python/sglang/bin/activate

# Install uv
```bash
pip install uv
```
```bash
## Configure ssh
ssh-keygen
```
Just throw the public key to github. Then configure git config locally:

```bash
git config --global user.name "zhaochenyang20"
git config --global user.email "zhaochenyang20@gmail.com"
```
## Configure oh-my-zsh

In fact, I usually don’t change the shell myself, but if the default is zsh, I still like to use oh-my-zsh:

**Note that oh-my-zsh will overwrite all previous .zshrc files, so you need to back them up first! ! ! **

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```