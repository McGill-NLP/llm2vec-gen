"""
Combine response JSONL shards and upload to a Hugging Face dataset.
Uses one split per model; if the dataset already exists, the model split is updated.
"""

import argparse
import glob
import logging
import pandas as pd
import re
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def safe_split_name(model_name: str) -> str:
    """Turn a model name into a valid HF split name (alphanumeric + underscore)."""
    name = re.sub(r"[^a-zA-Z0-9_.]", "_", model_name)
    name = name.replace(".", "")
    name = name.strip("_") or "default"
    return name


def load_shards(paths: list[str], reference_dataset: Dataset, positive_reference_dataset: Optional[Dataset] = None) -> list[dict]:
    """Load and merge records from one or more JSONL shard files."""
    response_data = None
    for path in sorted(paths):
        path = Path(path)
        if not path.exists():
            logger.warning("Shard not found: %s", path)
            continue
        df_ = pd.read_json(path, lines=True)
        if response_data is None:
            response_data = df_
        else:
            response_data = pd.concat([response_data, df_], ignore_index=True, sort=False)

    response_data = response_data.sort_values(by='id_')
    assert len(response_data) == len(reference_dataset), (
        f"Response data length {len(response_data)} does not match dataset length {len(reference_dataset)}"
    )

    data = []
    for idx, example in enumerate(tqdm(reference_dataset, desc="Combining responses with original questions")):
        example_dict: Dict[str, Any] = dict(example)

        ans = response_data.iloc[idx]["answer"]

        if positive_reference_dataset is not None:
            question = example_dict["negative_question"]
            assert question == response_data.iloc[idx]["question"], (
                f"Question mismatch for sample {idx}'s questions: {question} != {response_data.iloc[idx]['question']}"
            )
            pos_example = positive_reference_dataset[idx]
            assert example_dict.get("id") == pos_example["id"], (
                f"ID mismatch for sample {idx}'s questions: {example_dict.get('id')} != {pos_example['id']}"
            )
            data.append(
                {
                    "id": example_dict.get("id"),
                    "question": pos_example["question"],
                    "answer": pos_example["answer"],
                    "negative_question": question,
                    "negative_answer": ans,
                }
            )
        else:
            question = example_dict["question"]
            assert question == response_data.iloc[idx]["question"], (
                f"Question mismatch for sample {idx}'s questions: {question} != {response_data.iloc[idx]['question']}"
            )
            data.append(
                {
                    "id": example_dict.get("id"),
                    "question": question,
                    "answer": ans,
                }
            )

    return data

def run_responses(args: argparse.Namespace) -> Dataset:
    paths = []
    for p in args.shards:
        if "*" in p or "?" in p:
            paths.extend(glob.glob(p))
        else:
            paths.append(p)
    paths = sorted(set(paths))

    if not paths:
        raise FileNotFoundError(
            f"No shard files found. Check paths or glob: {args.shards}"
        )
    logger.info("Found %d shard(s).", len(paths))

    # Reference questions must exist in the repo_id dataset already
    reference_dataset = load_dataset(args.repo_id, split="original")

    negative_reference_dataset = None
    if args.negative_repo_id is not None:
        negative_reference_dataset = load_dataset(args.negative_repo_id, split="original")
        valid_ids = set(negative_reference_dataset["id"])
        reference_dataset = reference_dataset.filter(lambda example: example["id"] in valid_ids)
        records = load_shards(paths, negative_reference_dataset, positive_reference_dataset=reference_dataset)
    else:
        records = load_shards(paths, reference_dataset)
    logger.info("Loaded %d records.", len(records))

    new_split = Dataset.from_list(records)
    return new_split

def run_original_responses(args: argparse.Namespace) -> DatasetDict:
    name = args.model_name
    assert name in ["original", "gemini"], "Model name must be either original or gemini"

    def _get_num_messages(examples):
        examples["num_messages"] = [len(ex) for ex in examples["messages"]]
        return examples

    dataset_name = args.original_dataset_name
    ds = load_dataset(dataset_name, split="train")
    ds = ds.map(_get_num_messages, batched=True)
    ds = ds.filter(lambda example: example["num_messages"] == 2)

    positive_ds = None
    if args.negative_repo_id is not None:
        positive_ds = load_dataset(args.repo_id, split=name)
        valid_ids = set(ds["id"])
        positive_ds = positive_ds.filter(lambda example: example["id"] in valid_ids)

    data = []
    for idx, example in enumerate(tqdm(ds, desc="Processing original tulu responses")):
        example_dict: Dict[str, Any] = dict(example)
        messages: List[Dict[str, str]] = example_dict["messages"]
        if positive_ds is None:
            data.append(
                {
                    "id": example_dict.get("id"),
                    "question": messages[0]["content"],
                    "answer": messages[1]["content"],
                }
            )
        else:
            # When we want to upload original tulu responses and hard negative questions
            pos_example = positive_ds[idx]
            assert pos_example["id"] == example_dict.get("id"), (
                f"ID mismatch for sample {idx}'s questions: {pos_example['id']} != {example_dict.get('id')}"
            )
            data.append(
                {
                    "id": example_dict.get("id"),
                    "question": pos_example["question"],
                    "answer": pos_example["answer"],
                    "negative_question": messages[0]["content"],
                    "negative_answer": "N/A",
                }
            )

    dataset_split = Dataset.from_list(data)
    return dataset_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine response shards and upload to a Hugging Face dataset.",
    )
    parser.add_argument(
        "--shards",
        default=None,
        nargs="+",
        help="Paths to JSONL shard files, or a single glob pattern (e.g. 'data/tulu/model/*.jsonl').",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face dataset repo id (e.g. 'username/dataset-name').",
    )
    parser.add_argument(
        "--negative_shards",
        default=None,
        nargs="+",
        help="Paths to JSONL shard files, or a single glob pattern (e.g. 'data/tulu/model/*.jsonl').",
    )
    parser.add_argument(
        "--negative_repo_id",
        type=str,
        default=None,
        help="Hugging Face dataset repo id for negative responses (e.g. 'username/dataset-name'). Used only for uploading hard negative responses.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model that generated the responses; used as the dataset split name.",
    )
    parser.add_argument(
        "--original_dataset_name",
        type=str,
        default="vaibhavad/tulu-3-sft-mixture",
        help="Dataset name to load the original messages from. Used only for uploading original tulu responses.",
    )
    args = parser.parse_args()

    if args.model_name == "gemini":
        assert args.shards is None, "Shards must be None for uploading original tulu responses."
        dataset_split = run_original_responses(args)
        repo_id = args.negative_repo_id if args.negative_repo_id is not None else args.repo_id
        dataset_split.push_to_hub(
            repo_id, 
            split=args.model_name
        )
    elif args.model_name == "original":
        assert args.shards is None, "Shards must be None for uploading original tulu responses."
        dataset_split = run_original_responses(args)
        dataset_dict = DatasetDict({"original": dataset_split})
        repo_id = args.negative_repo_id if args.negative_repo_id is not None else args.repo_id
        dataset_dict.push_to_hub(
            repo_id
        )
    else:
        assert args.shards is not None, "Shards must be provided for generations."
        dataset_dict = run_responses(args)
        split_name = safe_split_name(args.model_name)
        repo_id = args.negative_repo_id if args.negative_repo_id is not None else args.repo_id
        dataset_dict.push_to_hub(
            repo_id, 
            split=split_name
        )
        
    logger.info("Pushed to https://huggingface.co/datasets/%s", repo_id)


if __name__ == "__main__":
    main()
