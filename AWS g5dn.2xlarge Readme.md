# GRPO on agentic LLM

## AWS Ubuntu Guide
Pick the Ubuntu Image with pytorch and cuda supported

### CUDA Docker Container Support
Following the link if CUDA docker not installed:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

To test if you have the correct VM os, run:
```shell
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

### Prepare Dataset Download Qwen2.5-1.5B-Instruct
First download the dataset from a cheaper instant, upload the documents to this instant.
```shell
python data_prep.py
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir $(pwd)/Qwen2.5-1.5B-Instruct
python create_lora_model.py
```

### Tmux
```shell
tmux new -s grpo
python grpo_train.py
```
Detach from the tmux session: Ctrl+b followed by d
Reattach to tumx session:
```shell
tmux attach -t grpo
```
To Kill the session
```shell
tmux kill-session -t grpo
```