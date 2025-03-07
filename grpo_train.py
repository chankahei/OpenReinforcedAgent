import os
import torch
import subprocess
import signal
from transformers import get_scheduler, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from torch.optim import AdamW
from grpo_trainer import GRPOTrainer
from datasets import load_from_disk, Dataset
import psutil
import logging
import json
import time
import pickle
from grpo_rollout import run_llm_rollout
import requests
import shutil
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class TrainingManager:
    def __init__(
        self,
        base_model_path: str,
        base_model_path_quantized: str,
        lora_model_path: str,
        dataset_path: str,
        output_dir: str,
        tools: list = None,
        max_steps: int = 1000,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        batch_size: int = 4,
        num_rollouts: int = 32,
        beta: float = 0.1,
        pad_token_id: int = 0,
        num_devices: int = 1,
        eval_steps: int = 50,
        save_steps: int = 50,
        max_length: int = 2048,
    ):
        self.base_model_path = base_model_path
        self.base_model_path_quantized = base_model_path_quantized
        self.lora_model_path = lora_model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.batch_size = batch_size
        self.num_rollouts = num_rollouts
        self.beta = beta
        self.tools = tools
        self.num_devices = num_devices
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(lora_model_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def kill_docker_container(self, container_name="vllm-server"):
        try:
            cmd = f"docker ps -q --filter name={container_name}"
            container_id = subprocess.check_output(cmd, shell=True).decode().strip()
            if container_id:
                subprocess.run(f"docker stop {container_id}", shell=True)
                logging.info(f"Successfully stopped container {container_id}")
        except Exception as e:
            logging.error(f"Error stopping docker container: {e}")

    #Needed for FSDP / DS3 only. FSDP / DS3 requires bnb_quant_storage_type=torch.bfloat16, but vllm does not support it. Hence, the conversion.
    def spawn_convert_container(self, base_model_path, lora_model_path, output_path):
        """Convert a trained lora model into vllm model"""
        cmd = f"""docker run -d --name convert --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace \
                 grpo:dev \
                 python merge_lora.py \
                 --base-model /workspace/{base_model_path} \
                 --lora-model /workspace/{lora_model_path} \
                 --output-path /workspace/{output_path}"""
        
        try:
            # Capture the container ID from the output
            result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, text=True)
            container_id = result.stdout.strip()
            logging.info(f"Successfully started convert container with ID: {container_id}")
            return container_id
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to start convert container: {e}")
            raise

    def spawn_vllm_server(self, device, steps, base_model_path, lora_model_path, port=8000):
        # Kill any existing container
        self.kill_docker_container(f"vllm-server-{device}")
        
        # Start new container with vllm model and LoRA adapter
        # cmd = f"""docker run -d --name vllm-server-{device} --rm --gpus='"device={device}"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        #          -v $(pwd):/workspace -p {port}:{port} \
        #          -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 \
        #          vllm/vllm-openai:v0.7.2 \
        #          --model /workspace/{base_model_path} \
        #          --enable-lora \
        #          --lora-modules grpo=/workspace/{lora_model_path} \
        #          --tool-call-parser hermes \
        #          --enable-auto-tool-choice \
        #          --quantization bitsandbytes \
        #          --load-format bitsandbytes \
        #          --gpu-memory-utilization 0.8 \
        #          --port {port} \
        #          --max_model_len 8192"""
        
        # lora merge is inplace
        cmd = f"""docker run -d --name vllm-server-{device} --rm --gpus='"device={device}"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace -p {port}:{port} \
                 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 \
                 vllm/vllm-openai:v0.7.2 \
                 --model /workspace/{lora_model_path} \
                 --tool-call-parser hermes \
                 --enable-auto-tool-choice \
                 --quantization bitsandbytes \
                 --load-format bitsandbytes \
                 --gpu-memory-utilization 0.8 \
                 --port {port} \
                 --max_model_len 8192"""

        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True)
            container_id = result.stdout.strip()
            logging.info(f"Successfully started vLLM server {device} container with ID: {container_id}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to start vLLM server: {e}")
            raise

    def spawn_rollout_container(self, dataset_path, device, num_rollouts, steps, vllm_port=8000):
        """Start rollout container with specified parameters"""
        cmd = f"""docker run -d --name rollout-{device} --rm --gpus '"device={device}"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
            --network host \
            -v $(pwd):/workspace \
            grpo:dev \
            python /workspace/grpo_rollout.py \
            --dataset /workspace/{dataset_path} \
            --device cuda \
            --model /workspace/vllm-models/vllm-{steps} \
            --num_rollouts {num_rollouts} \
            --vllm_port {vllm_port}"""
        try:
            subprocess.run(cmd, shell=True, check=True)
            logging.info(f"Successfully started rollout container on device {device}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to start rollout container: {e}")
            raise

    def spawn_ref_inference_container(self, base_model_path, dataset_path, device):
        """Run distributed inference with reference model to get logprobs"""
        
        # Kill any existing container
        self.kill_docker_container(f"ref-inference-{device}")
        
        # Run distributed inference
        cmd = f"""docker run -d --name ref-inference-{device} --rm --gpus '"device={device}"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace \
                 grpo:dev \
                 python grpo_ref_logprob.py \
                 --model-path /workspace/{base_model_path} \
                 --dataset-path /workspace/{dataset_path}
                """
        
        try:
            result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, text=True)
            container_id = result.stdout.strip()
            logging.info(f"Successfully started ref-inference container with ID: {container_id}")
            return container_id
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed ref-inference container {device}: {e}")
            raise

    def spawn_training_container(self, checkpoint_path, dataset_paths, max_steps):
        """Start training container with specified parameters"""
        dataset_paths = ['/workspace/' + x for x in dataset_paths]
        cmd = f"""docker run -d --name training --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace \
                 grpo:dev \
                 accelerate launch --config_file "deepspeed_config.yaml" grpo_trainer.py \
                 --checkpoint-path /workspace/{checkpoint_path} \
                 --dataset-paths {' '.join(dataset_paths)} \
                 --max-steps {max_steps}"""
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            logging.info("Successfully started training container")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to start training container: {e}")
            raise

    def check_container_exists(self, container_id=None, container_name=None):
        """Check if a container exists with the given ID or name"""
        try:
            if container_id:
                # Check by container ID
                cmd = f"docker ps -a -q --filter id={container_id}"
                result = subprocess.check_output(cmd, shell=True).decode().strip()
                return bool(result)
            else:
                # Check by container name
                cmd = f"docker ps -a -q --filter name={container_name}"
                result = subprocess.check_output(cmd, shell=True).decode().strip()
                return bool(result)
        except Exception as e:
            logging.error(f"Error checking if container exists: {e}")
            return False

    def process_rollouts(self, rollout):
        """Process rollout results into training inputs"""
        input_ids = []
        attention_mask = []
        labels = []
        advantages = []
        
        for advantage, _messages in zip(rollout['advantages'], rollout['messages']):
            templated_text = self.tokenizer.apply_chat_template(
                _messages,
                tokenize=False,
                tools=self.tools,
            )
            
            parts = [x + '<|im_end|>' for x in templated_text.split('<|im_end|>') if x != '\n']

            _input_ids = []
            _completion_masks = []

            for part in parts:
                tokens = self.tokenizer.encode(part, add_special_tokens=False)
                _input_ids.extend(tokens)
                if '<|im_start|>system' in part or '<|im_start|>user' in part:
                    _completion_masks.extend([0]*len(tokens))
                else:
                    _completion_masks.extend([1]*len(tokens))
            
            _labels = _input_ids[1:] + [-100]
            for index, mask in enumerate(_completion_masks):
                if mask == 0:
                    _labels[index] = -100

            _attention_mask = [1] * len(_input_ids)
            _advantages = [advantage] * (len(_input_ids) - 1) + [0.0]

            # left paddings and right truncations
            if len(_input_ids) >= self.max_length:
                _input_ids = _input_ids[:self.max_length]
                _attention_mask = _attention_mask[:self.max_length]
                _labels = _labels[:self.max_length]
                _advantages = _advantages[:self.max_length]
            else:
                _input_ids = [self.tokenizer.eos_token_id] * (self.max_length - len(_input_ids)) + _input_ids
                _attention_mask = [0] * (self.max_length - len(_attention_mask)) + _attention_mask
                _labels = [-100] * (self.max_length - len(_labels)) + _labels
                _advantages = [0.0] * (self.max_length - len(_advantages)) + _advantages

            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            labels.append(_labels)
            advantages.append(_advantages)
        
        return Dataset.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantages": advantages
        })

    def train(self, resume_from_step=0):
        lora_model_path = self.lora_model_path
        steps = resume_from_step
        # 1. Load dataset
        dataset = load_from_disk(self.dataset_path)
        train_dataset = dataset['train'].shuffle()
        eval_dataset = dataset['test']
        logging.info("Dataset loaded successfully")

        while steps < self.max_steps:
            logging.info(f"Starting step {steps + 1}/{self.max_steps}")
            
            try:
                _dataset = Dataset.from_dict(train_dataset[steps*self.batch_size:(steps+1)*self.batch_size])

                checkpoint_path = self.lora_model_path
                if steps > 0:
                    checkpoint_path = checkpoint_path + f"/checkpoint-{steps}"
                # Conversion if needed for FSDP / DS3
                # Delete the last converted model to conserve space, except for save steps
                if not ((steps % self.save_steps == 0 and steps != 0) or steps == self.max_steps - 1):
                    subprocess.run(f'docker run -v $(pwd):/workspace python bash -c "rm -rf /workspace/vllm-models/vllm-{steps-1} || true"', shell=True, check=True)
                
                # Convert the current model into vLLM model
                self.spawn_convert_container(
                    base_model_path=self.base_model_path,
                    lora_model_path=checkpoint_path,
                    output_path=os.path.join('vllm-models', f"vllm-{steps}")
                )
                
                while True:
                    if not self.check_container_exists(container_name="convert"):
                        break
                    time.sleep(10)

                # Launch vLLM servers and rollout containers on each device
                for device in range(self.num_devices):
                    port = 8000 + device
                    self.spawn_vllm_server(
                        device=device,
                        steps=steps, 
                        base_model_path=self.base_model_path, 
                        #lora_model_path=checkpoint_path,
                        lora_model_path=os.path.join('vllm-models', f"vllm-{steps}"), 
                        port=port
                    )

                while True:
                    try:
                        responses = []
                        for device in range(self.num_devices):
                            response = requests.get(f"http://localhost:{8000+device}/v1/models")
                            responses.append(response)
                        if all([response.status_code == 200 for response in responses]):
                            break
                    except requests.exceptions.RequestException as e:
                        logging.error(f"Failed to connect to vLLM server {device}: {e}")
                        time.sleep(10)

                # Perform evaluation on the eval dataset
                if steps % self.eval_steps == 0 and steps != 0:
                    for device in range(self.num_devices):
                        port = 8000 + device
                        device_eval_dataset_path = os.path.join(self.output_dir, f"temp_dataset_{steps}_device_{device}_eval")
                        device_eval_dataset = eval_dataset.select(range(device*(len(eval_dataset)//self.num_devices), (device+1)*(len(eval_dataset)//self.num_devices)))
                        device_eval_dataset.save_to_disk(device_eval_dataset_path)

                        self.spawn_rollout_container(
                            dataset_path=device_eval_dataset_path,
                            num_rollouts=self.num_rollouts,
                            device=device,
                            steps=steps,
                            vllm_port=port
                        )
                    
                    # Wait for all rollouts to complete
                    while True:
                        all_completed = [not self.check_container_exists(container_name=f"rollout-{device}") for device in range(self.num_devices)]
                        if all(all_completed):
                            break
                        time.sleep(10)

                    # Load and combine rollout results from all devices
                    rollout_results = {
                        "messages":[],
                        "advantages":[],
                        "metrics":[]
                    }

                    for device in range(self.num_devices):
                        device_eval_dataset_path = os.path.join(self.output_dir, f"temp_dataset_{steps}_device_{device}_eval")
                        rollout_results_path = os.path.join(device_eval_dataset_path, "rollout_results.pkl")
                        with open(rollout_results_path, 'rb') as f:
                            result = pickle.load(f)
                            rollout_results['messages'].extend(result['messages'])
                            rollout_results['advantages'].extend(result['advantages'])
                            rollout_results['metrics'].extend(result['metrics'])
                    
                    logging.info(f"Eval rollout results:")
                    print(pd.DataFrame(rollout_results['metrics']).describe())

                # Launch rollout containers on each device
                for device in range(self.num_devices):
                    port = 8000 + device
                    # Split dataset for each device
                    device_dataset_path = os.path.join(self.output_dir, f"temp_dataset_{steps}_device_{device}")
                    device_dataset = _dataset.select(range(device*(len(_dataset)//self.num_devices), (device+1)*(len(_dataset)//self.num_devices)))
                    device_dataset.save_to_disk(device_dataset_path)

                    # 3. Spawn rollout container for each device
                    self.spawn_rollout_container(
                        dataset_path=device_dataset_path,
                        num_rollouts=self.num_rollouts,
                        device=device,
                        steps=steps,
                        vllm_port=port
                    )
                
                # Wait for all rollouts to complete
                while True:
                    all_completed = [not self.check_container_exists(container_name=f"rollout-{device}") for device in range(self.num_devices)]
                    if all(all_completed):
                        break
                    time.sleep(10)

                # Load and combine rollout results from all devices
                rollout_results = {
                    "messages":[],
                    "advantages":[],
                    "metrics":[]
                }

                for device in range(self.num_devices):
                    device_dataset_path = os.path.join(self.output_dir, f"temp_dataset_{steps}_device_{device}")
                    rollout_results_path = os.path.join(device_dataset_path, "rollout_results.pkl")
                    with open(rollout_results_path, 'rb') as f:
                        result = pickle.load(f)
                        rollout_results['messages'].extend(result['messages'])
                        rollout_results['advantages'].extend(result['advantages'])
                        rollout_results['metrics'].extend(result['metrics'])
                
                # 4. Kill all vLLM servers
                for device in range(self.num_devices):
                    self.kill_docker_container(container_name=f"vllm-server-{device}")
                logging.info("All vLLM servers stopped")
                
                # 5. Process rollouts
                processed_inputs = self.process_rollouts(rollout_results)
                
                for device in range(self.num_devices):
                    device_dataset_path = os.path.join(self.output_dir, f"temp_dataset_{steps}_device_{device}_tokenized")
                    device_dataset = processed_inputs.select(range(device*(len(processed_inputs)//self.num_devices), (device+1)*(len(processed_inputs)//self.num_devices)))
                    device_dataset.save_to_disk(device_dataset_path)
                    logging.info(f"Processed rollout results for device {device}")
                
                # 6. Run distributed inference to get reference logprobs
                for device in range(self.num_devices):
                    device_dataset_path = os.path.join(self.output_dir, f"temp_dataset_{steps}_device_{device}_tokenized")
                    self.spawn_ref_inference_container(base_model_path=self.base_model_path_quantized, dataset_path=device_dataset_path, device=device)
                
                while True:
                    all_completed = [not self.check_container_exists(container_name=f"ref-inference-{device}") for device in range(self.num_devices)]
                    if all(all_completed):
                        break
                    time.sleep(10)  # Check every 10 seconds
                logging.info("Computed reference logprobs")

                # 7. Start training container
                self.spawn_training_container(
                    checkpoint_path=checkpoint_path,
                    dataset_paths=[os.path.join(self.output_dir, f"temp_dataset_{steps}_device_{device}_tokenized_ref_logprobs") for device in range(self.num_devices)],
                    max_steps=self.max_steps
                )
                
                # 8. Wait for training to complete
                time.sleep(10)  # Give container time to start
                while True:
                    if not self.check_container_exists(container_name="training"):
                        break
                    time.sleep(30)  # Check every 30 seconds

                steps += 1
                
            except Exception as e:
                logging.error(f"Error during training: {e}")
                # Kill all vLLM servers on error
                for device in range(self.num_devices):
                    self.kill_docker_container(container_name=f"vllm-server-{device}")
                raise
            
        logging.info("Training completed successfully")

if __name__ == "__main__":
    manager = TrainingManager(
        base_model_path="Qwen2.5-7B-Instruct",
        base_model_path_quantized="Qwen2.5-7B-Instruct-bnb-4bit",
        lora_model_path="Qwen2.5-7B-Instruct-qlora",
        dataset_path="dataset_curated",
        output_dir="Qwen2.5-7B-Instruct-qlora",
        max_steps=250,
        learning_rate=1e-4,
        batch_size=32,
        num_rollouts=32,
        beta=0.04,
        num_devices=1,  # Use 2 GPUs by default
        eval_steps=20,
        save_steps=20,
        max_length=1024,
    )
    
    manager.train(resume_from_step=0)
 