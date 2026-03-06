import argparse
import os

import datasets
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm2vec_gen.models import LLM2VecGenModel


def main(encoder_model_path: str, batch_size: int, dataset_name: str = "sanity-check", output_dir: str = None):
    assert output_dir is not None, "Output directory is required"
    model = LLM2VecGenModel.from_pretrained(encoder_model_path)

    # Remove MLPs so that we can get the hidden states of the encoder directly
    model.model.alignment_mlp = None
    model.model.reconstruction_mlp = None

    if dataset_name == "sanity-check":
        dataset = datasets.load_dataset(
            "vaibhavad/sanity-check-generative-v2", split="train"
        )
    elif dataset_name == "advbenchir":
        dataset = datasets.load_dataset(
            "McGill-NLP/AdvBench-IR", split="train"
        )
    elif dataset_name == "generation":
        dataset = datasets.load_dataset(
            "vaibhavad/generation-eval-data", split="train"
        )
    elif dataset_name == "nq":
        dataset = datasets.load_dataset("mteb/nq", "queries")
        dataset = dataset["queries"]
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    dataset = dataset.shuffle(seed=42).select(range(100))
    # tqdm

    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = []
    inputs = []

    for batch in tqdm(dataloader):
        input_texts = batch["question"] if "question" in batch else batch["query"] if "query" in batch else batch["text"]

        _, dec_embeddings = model.encode(
            input_texts,
            get_recon_hidden_states=True,
        )

        # pass it tthrough lm head
        lm_head = model.model.decoder_model.lm_head
        logits = lm_head(dec_embeddings)

        # get top 5 tokens
        topk_values, topk_indices = torch.topk(logits, k=5, dim=-1)

        batch_results = []
        for b in range(logits.shape[0]):
            seq_tokens = []
            pos_top5_strings = []
            for s in range(logits.shape[1]):
                pos_top5_ids = topk_indices[b, s].tolist()
                pos_top5_strings += [model.tokenizer.decode([tid]) for tid in pos_top5_ids]
                formatted_pos = f"[{' | '.join(pos_top5_strings)}]"
                seq_tokens.append(formatted_pos)
            batch_results.append(set(pos_top5_strings))
        results.extend(batch_results)
        inputs.extend(input_texts)
    
    df = pd.DataFrame({"input": inputs, "output": results})
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/logit_lens_{dataset_name}.csv", index=False)
    print(f"Results saved to {output_dir}/logit_lens_{dataset_name}.csv")


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_model_path", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="sanity-check", choices=["sanity-check", "advbenchir", "generation", "nq"])
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.encoder_model_path
    main(args.encoder_model_path, args.batch_size, dataset_name=args.dataset, output_dir=output_dir)
