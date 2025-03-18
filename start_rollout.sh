docker run -it --name rollout-0 --rm --gpus '"device=0"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
            --network host \
            -v $(pwd):/workspace \
            grpo:dev \
            python /workspace/grpo_rollout.py \
            --dataset /workspace/Qwen2.5-3B-Instruct-qlora/temp_dataset_0_device_0 \
            --device cuda \
            --model grpo \
            --num_rollouts 8 \
            --vllm_port 8000