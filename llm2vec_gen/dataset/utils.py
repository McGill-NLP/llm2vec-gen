from .dataset import Dataset
import re

DATASET_MAPPING = {
    "dataset": Dataset,
    "Llama-3.2-1B-Instruct": Dataset,
    "Llama-3.2-3B-Instruct": Dataset,
    "Llama-3.1-8B-Instruct": Dataset,
    "Qwen2.5-0.5B-Instruct": Dataset,
    "Qwen2.5-1.5B-Instruct": Dataset,
    "Qwen2.5-3B-Instruct": Dataset,
    "Qwen2.5-7B-Instruct": Dataset,
    "Qwen3-0.6B": Dataset,
    "Qwen3-1.7B": Dataset,
    "Qwen3-4B": Dataset,
    "Qwen3-8B": Dataset,
    "Gemini": Dataset,
    "original": Dataset,
}


def load_dataset(name, *args, **kwargs):
    return DATASET_MAPPING[name](*args, **kwargs)

def safe_split_name(model_name: str) -> str:
    """Turn a model name into a valid HF split name (alphanumeric + underscore)."""
    name = re.sub(r"[^a-zA-Z0-9_.]", "_", model_name)
    name = name.replace(".", "")
    name = name.strip("_") or "default"
    return name