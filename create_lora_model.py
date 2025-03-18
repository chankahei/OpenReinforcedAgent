from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import torch
import os
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_quant_storage=torch.uint8,
    bnb_4bit_quant_storage=torch.bfloat16 # needed for fsdp / ds3
)

def create_lora_model(
    base_model_path: str,
    output_dir: str,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    bias: str = "none",
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
):
    """
    Create a LoRA version of the base model.
    
    Args:
        base_model_path: Path to the base model
        output_dir: Directory to save the LoRA model
        r: LoRA attention dimension
        lora_alpha: Alpha parameter for LoRA scaling
        lora_dropout: Dropout probability for LoRA layers
        bias: Bias type for LoRA. Can be 'none', 'all' or 'lora_only'
        target_modules: List of module names to apply LoRA to
    """
    logging.info(f"Loading base model from {base_model_path}")
    
    # Load base model in bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        attn_implementation = "flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    # model.save_pretrained(base_model_path+"-bnb-4bit")
    
    # del model

    # model = AutoModelForCausalLM.from_pretrained(base_model_path+"-bnb-4bit")
    # # model = prepare_model_for_kbit_training(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # tokenizer.save_pretrained(base_model_path+"-bnb-4bit")

    # Define LoRA Config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM
    )
    
    # Get PEFT model
    model = get_peft_model(
        model, 
        lora_config, 
        autocast_adapter_dtype=False,
    )
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Save LoRA model and config
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # modify adapter config
    # with open(os.path.join(output_dir, "adapter_config.json"), "r") as f:
    #     adapter_config = json.load(f)
    #     adapter_config["base_model_name_or_path"] = base_model_path+"-bnb-4bit"
    # with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
    #     json.dump(adapter_config, f, indent=2)

    logging.info(f"LoRA model saved to {output_dir}")
    return model, tokenizer

if __name__ == "__main__":
    # Create LoRA version of the model
    model, tokenizer = create_lora_model(
        base_model_path='/workspace/Qwen2.5-3B-Instruct',
        output_dir="/workspace/Qwen2.5-3B-Instruct-qlora",
    )