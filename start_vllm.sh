docker run -it --name vllm-server-0 --rm --gpus='"device=0"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace -p 8000:8000 \
                 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 \
                 vllm/vllm-openai:v0.7.2 \
                 --model /workspace/vllm-models/vllm-11 \
                 --tool-call-parser hermes \
                 --enable-auto-tool-choice \
                 --quantization bitsandbytes \
                 --load-format bitsandbytes \
                 --gpu-memory-utilization 0.8 \
                 --port 8000 \
                 --max_model_len 2048
