import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env.models"), override=True)
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "Qwen2.5-7B-Instruct")
BASE_MODEL_PATH_QUANTIZED = os.getenv("BASE_MODEL_PATH_QUANTIZED", "Qwen2.5-7B-Instruct-bnb-4bit")
LORA_MODEL_PATH = os.getenv("LORA_MODEL_PATH", "Qwen2.5-7B-Instruct-qlora")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "Qwen2.5-7B-Instruct-qlora")