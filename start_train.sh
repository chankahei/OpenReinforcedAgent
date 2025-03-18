docker run -it --name training --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace \
                 grpo:dev \
                 accelerate launch --config_file "deepspeed_config.yaml" grpo_trainer.py \
                 --checkpoint-path /workspace/Qwen2.5-3B-Instruct-qlora \
                 --dataset-paths /workspace/Qwen2.5-3B-Instruct-qlora/temp_dataset_0_device_0_tokenized_ref_logprobs /workspace/Qwen2.5-3B-Instruct-qlora/temp_dataset_0_device_1_tokenized_ref_logprobs \
                 --max-steps 100