import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_from_disk
import logging
import argparse
import os
from utils import selective_log_softmax
from functools import reduce

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_quant_storage=torch.uint8,
    bnb_4bit_quant_storage=torch.bfloat16 # needed for fsdp / ds3
)

def run_inference(model_path, dataset_path):
    """Run inference on a specific portion of the dataset"""
    logging.basicConfig(level=logging.INFO)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    ).to('cuda')
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    
    def compute_ref_logprobs(batch):
        _input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).to("cuda")
        _attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).to("cuda")
        _labels = torch.tensor(batch["labels"], dtype=torch.long).to("cuda")
        with torch.no_grad():
            outputs = model(_input_ids, attention_mask=_attention_mask)
            logits = outputs.logits
            mask = _labels.ne(-100)
            logprobs = selective_log_softmax(logits[mask], _labels[mask]).detach().cpu().tolist()
            
            del _input_ids, _attention_mask, outputs, logits
            torch.cuda.empty_cache()
            mask = list(reduce(lambda a, b: a + [a[-1] + b], mask.sum(dim=1).cpu().tolist(), [0]))
            logprobs = [logprobs[i:j] for i,j in zip(mask[:-1], mask[1:])] 

        return {
            "ref_logprobs": logprobs
        }
    
    dataset = dataset.map(compute_ref_logprobs, batched=True, batch_size=2)

    dataset.save_to_disk(args.dataset_path+ "_ref_logprobs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model_path,
        dataset_path=args.dataset_path
    )
