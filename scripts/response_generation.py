import argparse
import json
import logging
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")
    logging.info(f"Model loaded to device: {model.device}")
    model.eval()
    logging.info(f"Loaded model: {args.model_name}")

    thinking = False
    if "Qwen3" in args.model_name:
        logging.info(f"Qwen3 model detected, will remove <think>...</think> from the output")
        thinking = True

    dataset = load_dataset(args.dataset_path, split="original")
    assert args.shard_id < args.num_shards, f"Shard ID {args.shard_id} must be less than number of shards {args.num_shards}"
    shard_len = int(np.ceil(len(dataset) / args.num_shards))
    start_idx = args.shard_id * shard_len
    dataset = dataset.select(range(start_idx, start_idx + shard_len))
    logging.info(f"Loaded dataset: {args.dataset_name} with {len(dataset)} samples: from {args.shard_id * shard_len} to {(args.shard_id + 1) * shard_len}")

    # save as jsonl
    _dir = f"{args.output_dir}/{args.dataset_name}/{args.model_name.split('/')[-1]}"
    os.makedirs(_dir, exist_ok=True)
    addr = f"{_dir}/{args.dataset_name}_{args.model_name.split('/')[-1]}_max-{args.max_new_tokens}_bs-{args.batch_size}_generations_shard-{args.shard_id}_{args.num_shards}.jsonl"
    
    if os.path.exists(addr):
        with open(addr, "r") as f:
            num_lines = sum(1 for _ in f)
            logging.info(f"Already parsed {num_lines} samples, resuming from line {num_lines}")
            start_idx += num_lines
            dataset = dataset.select(range(num_lines, len(dataset)))
    
    f = open(addr, "a+", encoding="utf-8")
    # Hugging Face `Dataset` slicing returns a dict-of-lists; this is faster and
    # avoids IterableDataset batching quirks.
    for i, batch_start in enumerate(range(0, len(dataset), args.batch_size)):
        batch = dataset[batch_start : batch_start + args.batch_size]
        questions = batch["question"]
        original_ids = batch["original_id"] if "original_id" in batch else [None] * len(questions)
        if i % 10 == 0:
            logging.info(f"Processing sample {i * args.batch_size + 1}/{len(dataset)}")

        messages = [
            [{"role": "user", "content": q}] for q in questions
        ]
        
        with torch.no_grad():
            if thinking:
                batch_texts = [
                    tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
                    for msgs in messages
                ]
            else:
                batch_texts = [
                    tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                    for msgs in messages
                ]

            input_ids = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length
            )
            input_ids = input_ids.to(model.device)
        
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )

            for j in range(len(questions)):
                output_text = tokenizer.decode(output_ids[j, input_ids.input_ids.shape[1]:], skip_special_tokens=True)
                result = {
                        "id_": start_idx + i * args.batch_size + j,
                        "question": messages[j][-1]["content"],
                        "answer": output_text.strip(),
                    }
                if original_ids[j] is not None:
                    result["original_id"] = original_ids[j]
                f.write(json.dumps(result) + "\n")

            if i % 10 == 0:
                f.flush()

        del input_ids, output_ids
        torch.cuda.empty_cache()

    logging.info(f"Saved generations to {addr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="tulu")
    parser.add_argument("--dataset_path", type=str, default="vaibhavad/tulu-3-sft-mixture")  # or "vaibhavad/tulu-3-sft-mixture-hard-negative"
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    main(args)
