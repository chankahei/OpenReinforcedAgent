import torch
import transformers
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig, TrainerCallback
from peft import AutoPeftModelForCausalLM, PeftModel
from typing import Optional, Dict, List
from packaging import version
from dataclasses import dataclass
from grpo_rollout import run_llm_rollout, calculate_rewards, get_mrr, get_answer_similarity, get_format_reward
from torch.nn import functional as F
from utils import pad_and_truncate, selective_log_softmax
from datasets import concatenate_datasets, load_from_disk
import argparse
import json
import os
import glob
import re
from accelerate import PartialState
from typing import Optional, Dict, List, Any
from accelerate.logging import get_logger
from transformers.utils import is_peft_available
from transformers.modeling_utils import PreTrainedModel
import logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

class StepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        control.should_training_stop = True
        return control

class GRPOTrainer(Trainer):
    def __init__(
        self,
        model,
        args,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        beta=0.04,  # KL penalty coefficient
        epsilon=0.2,  # Clamping coefficient
        **kwargs,
    ):
        super().__init__(model, args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs)
        self.beta = beta
        self.epsilon = epsilon
        self._metrics = {"kl": [], "per-token-loss": []}
        self._last_loaded_step = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Except the following keys in the inputs:
        - input_ids: the rollout input_ids
        - ref_logprobs: the logprobs computed from the reference model
        - completion_ids: a mask indicating which tokens are completions
        - advantage: the advantage computed from the rollout
        """
        if return_outputs:
            raise ValueError("The RLTrainer does not support returning outputs")

        # Get inputs
        input_ids = inputs.get("input_ids").to(self.model.device)
        attention_mask = inputs.get("attention_mask").to(self.model.device)
        labels = inputs.get("labels").to(self.model.device)
        advantages = inputs.get("advantages").to(self.model.device)
        ref_logprobs = inputs.get("ref_logprobs").to(self.model.device)
        
        if any(x is None for x in [input_ids, attention_mask, labels, advantages, ref_logprobs]):
            raise ValueError("Missing required inputs: input_ids, attention_mask, labels, advantage, or ref_logprobs")
        
        # Compute logprobs for current model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Create a mask for valid labels (not -100)
        mask = labels.ne(-100)
        logprobs = selective_log_softmax(logits[mask], labels[mask])

        # Compute approx KL divergence
        kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1

        coef_1 = torch.exp(logprobs - logprobs.detach())
        ## we did not consider the case where old_logprob != logprobs (i.e. the rollout model != current model)
        ## coef_1 will be evaluated to 1 when old_logprob == logprobs
        # coef_2 = torch.clamp(coef_1, 1-self.epsilon, 1+self.epsilon)
        per_token_loss1 = coef_1 * advantages[mask]
        # per_token_loss2 = coef_2 * advantages[mask]
        ## Compute policy loss with KL penalty
        # policy_loss = -torch.min(per_token_loss1, per_token_loss2) + self.beta * kl
        ## negative because we want to minimize the policy loss
        policy_loss = -per_token_loss1 + self.beta * kl

        loss = policy_loss.mean()

        # Update metrics
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(kl.mean()).mean().item())
        self._metrics["per-token-loss"].append(self.accelerator.gather_for_metrics(per_token_loss1.mean()).mean().item())
        
        return loss

    # turn off save embedding layer
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")

            supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                    )
                else:
                    logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    if self.args.save_safetensors:
                        safetensors.torch.save_file(
                            state_dict, os.path.join(output_dir, 'model.safetensors'), metadata={"format": "pt"}
                        )
                    else:
                        torch.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
            else:
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors, save_embedding_layers=False
                )

            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True, help="Path to the model directory")
    parser.add_argument("--dataset-paths", required=True, nargs="+")
    parser.add_argument("--max-steps", type=int, default=10000)
    
    args = parser.parse_args()

    datasets = [load_from_disk(dataset_path) for dataset_path in args.dataset_paths]
    # Concat datasets
    dataset = concatenate_datasets(datasets)
    if len(dataset) % PartialState().num_processes != 0:
        # Drop last few examples
        dataset = dataset.select(range(len(dataset) - (len(dataset) % PartialState().num_processes)))

    if 'checkpoint' in args.checkpoint_path:
        output_path = args.checkpoint_path.split('checkpoint-')[0]
        current_step = int(args.checkpoint_path.split('checkpoint-')[1])
    else:
        output_path = args.checkpoint_path
        current_step = 0

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        # bnb_4bit_quant_storage=torch.uint8,
        bnb_4bit_quant_storage=torch.bfloat16, # needed for FSDP / DS3
    )

    # Training model (base model with LoRA)
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.checkpoint_path,
        is_trainable=True,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=True,
        device_map={'':PartialState().local_process_index}
    )

    from liger_kernel.transformers import apply_liger_kernel_to_qwen2

    apply_liger_kernel_to_qwen2(
        model = model.base_model.model,
        rope = True,
        cross_entropy = False,
        fused_linear_cross_entropy = False,
        rms_norm = True,
        swiglu = True,
    )
    
    gradient_accumulation_steps = len(dataset) // PartialState().num_processes
    # 4. Setup training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        remove_unused_columns=False,
        bf16=True,
        learning_rate=1e-5,
        weight_decay=0.01,
        output_dir=output_path,
        #optim="adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        save_steps=1,
        save_strategy="steps",
        max_steps=args.max_steps,
        warmup_ratio=0.05,
        ddp_find_unused_parameters=False
    )

    def data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # transform to torch tensor and some reshaping
        input_ids = torch.stack([torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features])
        attention_mask = torch.stack([torch.tensor(feature["attention_mask"], dtype=torch.long) for feature in features])
        labels = torch.stack([torch.tensor(feature["labels"], dtype=torch.long) for feature in features])
        advantages = torch.stack([torch.tensor(feature["advantages"], dtype=torch.bfloat16) for feature in features])

        ref_logprobs = [feature["ref_logprobs"] for feature in features]
        ref_logprobs = torch.tensor([item for sublist in ref_logprobs for item in sublist], dtype=torch.bfloat16)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantages": advantages,
            "ref_logprobs": ref_logprobs
        }

    # 6. Initialize trainer and train
    trainer = GRPOTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
        args=training_args,
        beta=0.04,  # KL penalty coefficient
        epsilon=0.2,  # Clamping coefficient
        callbacks=[StepCallback]
    )
    
    try:
        if 'checkpoint' in args.checkpoint_path:
            training_stats = trainer.train(
                resume_from_checkpoint=args.checkpoint_path,
            )
        else:
            training_stats = trainer.train()
        
        # save training stats
        if PartialState().local_process_index == 0:
            with open(os.path.join(output_path, f'training_stats_{current_step}.json'), 'w') as f:
                training_stats = training_stats._asdict()
                training_stats.update({'learning_rate': trainer._get_learning_rate()})
                training_stats.update(trainer._metrics)
                json.dump(training_stats, f, indent=2)
    except Exception as e:
        with open(os.path.join(output_path, f'training_stats_{current_step}.json'), 'w') as f:
            json.dump({'error': str(e)}, f, indent=2)



