import argparse
import os

import datasets
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from llm2vec_gen.models import LLM2VecGenModel


def get_generations(model, input_text: str, max_new_tokens: int):

    answer = model.generate(
        input_text,
        max_new_tokens=max_new_tokens,
    )
    return answer

def main(encoder_model_path: str, max_new_tokens: int, dataset_name: str = "sanity-check", output_dir: str = None):
    assert output_dir is not None, "Output directory is required"

    model = LLM2VecGenModel.from_pretrained(encoder_model_path)

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
    elif dataset_name == "test":
        qs = [
            "what is artificial intelligence", 
            "where do polar bears live and what's their habitat", 
            "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?",
            "The 2000 British film Snatch was later adapted into a television series for what streaming service?",
            "Which year and which conference was the 14th season for this conference as part of the NCAA Division that the Colorado Buffaloes played in with a record of 2-6 in conference play?"
        ]
        dataset = Dataset.from_list(
            [{"question": q} for q in qs]
        )
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    dataset = dataset.shuffle(seed=42).select(range(min(100, len(dataset))))

    results = []

    for sample in tqdm(dataset):
        input_text = sample["question"] if "question" in sample else sample["query"] if "query" in sample else sample["text"]
        output_text = get_generations(model, input_text, max_new_tokens=max_new_tokens)

        results.append(
            {
                "input_text": input_text,
                "output_text": output_text,
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(f"{output_dir}/generations_{dataset_name}.csv", index=False)
    print(f"Results saved to {output_dir}/generations_{dataset_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_model_path", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--dataset", type=str, default="sanity-check", choices=["sanity-check", "advbenchir", "generation", "nq", "test"])
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.encoder_model_path
    main(args.encoder_model_path, args.max_new_tokens, dataset_name=args.dataset, output_dir=output_dir)
