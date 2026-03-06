import argparse
import json
import os
from typing import Any, Dict, List

import mteb
import numpy as np
import torch
from mteb.encoder_interface import PromptType

from tqdm import tqdm

from llm2vec_gen import LLM2VecGenModel

# Define task sets
SMALL_TASKS = [
    "ArguAna",
    "ClimateFEVERHardNegatives",
    "MindSmallReranking",
    "ArXivHierarchicalClusteringP2P",
    "MedrxivClusteringS2S.v2",
    "SprintDuplicateQuestions",
    "Banking77Classification",
    "MTOPDomainClassification",
    "STS14",
    "STS22.v2",
]

def corpus_to_texts(
    corpus: list[dict[str, str]] | dict[str, list[str]] | list[str],
    sep: str = "\n",
) -> list[str]:
    if isinstance(corpus, dict):
        return [
            (corpus["title"][i] + sep + corpus["text"][i]).strip()  # type: ignore
            if "title" in corpus
            else corpus["text"][i].strip()  # type: ignore
            for i in range(len(corpus["text"]))  # type: ignore
        ]
    else:
        if isinstance(corpus[0], str):
            return corpus
        return [
            (doc["title"] + sep + doc["text"]).strip()
            if "title" in doc
            else doc["text"].strip()
            for doc in corpus
        ]

class ModelWrapper:
    def __init__(
        self,
        model: LLM2VecGenModel,
        task_to_instructions: Dict[str, str],
        max_input_length: int = 512,
        **kwargs,
    ):
        self.task_to_instructions = task_to_instructions
        self.model = model
        self.max_input_length = max_input_length

    def _single_batch_encode(
        self,
        sentences: List[str],
        **kwargs: Any,
    ) -> torch.Tensor:
        embeddings = self.model.encode(
            sentences,
            max_length=self.max_input_length,
        )
        return embeddings

    def _batch_encode(
        self,
        sentences: List[List[str]],
        **kwargs: Any,
    ) -> torch.Tensor:
        instruction_sentences = [
            sentence[0].strip()
            + " "
            + sentence[1].strip()
            for sentence in sentences
        ]
        length_sorted_idx = np.argsort([-len(sen) for sen in instruction_sentences])
        instruction_sentences_sorted = [
            instruction_sentences[i] for i in length_sorted_idx
        ]
        # pass in batch of kwargs.batch_size
        all_embeddings = []
        batch_size = kwargs.pop("batch_size", 32)
        for start_index in tqdm(
            range(0, len(instruction_sentences_sorted), batch_size),
            desc="Encoding",
        ):
            sentences_batch = instruction_sentences_sorted[
                start_index : start_index + batch_size
            ]
            embeddings = self._single_batch_encode(
                sentences_batch, **kwargs
            )
            all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]
        all_embeddings = all_embeddings.to(torch.float32)
        return all_embeddings

    def encode(
        self,
        sentences: List[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,  # noqa
    ) -> torch.Tensor:
        if prompt_type == PromptType.passage:
            sentences = corpus_to_texts(sentences, sep=" ")
            corpus_sentences = [
                ["Summarize the following passage:", sentence] for sentence in sentences
            ]
            enc = self._batch_encode(
                corpus_sentences, **kwargs
            )
            return enc.cpu()
        else:
            prompt_name = kwargs.pop("prompt_name", None)
            assert prompt_name is not None or task_name is not None, (
                "Prompt name or task name is required"
            )
            if prompt_name is not None:
                assert prompt_name in self.task_to_instructions, (
                    f"Prompt name {prompt_name} not found in task_to_instructions"
                )
            elif task_name is not None:
                assert task_name in self.task_to_instructions, (
                    f"Task name {task_name} not found in task_to_instructions"
                )
            if prompt_name is not None:
                instruction = self.task_to_instructions[prompt_name]
            elif task_name is not None:
                instruction = self.task_to_instructions[task_name]

            instruction_sentences = [[instruction, sentence] for sentence in sentences]
            enc = self._batch_encode(instruction_sentences, **kwargs)
            return enc.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        default=None,
        help="Path to the encoder model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the output directory. If not provided, the encoder model path will be used.",
    )
    parser.add_argument(
        "--task_to_instructions_fp",
        type=str,
        default="generative.json",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--task_set",
        type=str,
        default="lite",
        help="Choose between lite or all task set for evaluation, or put the mteb task's name",
    )

    args = parser.parse_args()

    enc_dec_model = LLM2VecGenModel.from_pretrained(args.model_path)

    with open(args.task_to_instructions_fp, "r") as f:
        task_to_instructions = json.load(f)

    wrapper_model = ModelWrapper(
        enc_dec_model,
        task_to_instructions=task_to_instructions,
        max_input_length=512,
    )

    tasks = []
    if args.task_set == "all":
        tasks = mteb.get_benchmark("MTEB(eng, v2)")
    elif args.task_set == "lite":
        all_tasks = mteb.get_benchmark("MTEB(eng, v2)")

        for t in all_tasks:
            if t.metadata.name in SMALL_TASKS:
                tasks.append(t)
    else:
        tasks = [args.task_set]

    evaluation = mteb.MTEB(tasks=tasks)
    path_name = (
        "mteb_eval"
        + ("_all" if args.task_set == "all" else "" if args.task_set == "lite" else f"_{args.task_set}")
    )

    output_dir = os.path.join(
        args.output_dir,
        path_name,
        args.task_to_instructions_fp.split(".")[0]
    )

    results = evaluation.run(
        wrapper_model,
        output_folder=output_dir,
        eval_splits=["test"],
        encode_kwargs={"batch_size": args.batch_size},
    )
