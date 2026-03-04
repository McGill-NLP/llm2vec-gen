from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import torch
from transformers import PreTrainedTokenizerFast

from llm2vec_gen.dataset.base_dataset import DataSample


@dataclass
class CustomCollator:
    tokenizer: PreTrainedTokenizerFast
    special_tokens: List[str]
    teacher_tokenizer: Optional[PreTrainedTokenizerFast] = None
    teacher_special_tokens: Optional[List[str]] = None
    padding: str = "max_length"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    @staticmethod
    def _add_special_tokens_if_needed(
        input_ids: torch.LongTensor, special_token_ids: torch.LongTensor
    ) -> torch.LongTensor:
        """Add special tokens to the input sequence if they are missing.

        Args:
            input_ids: Tensor of input token IDs
            special_token_ids: Tensor of special token IDs to add if missing

        Returns:
            Tensor with special tokens added where needed
        """
        N = special_token_ids.shape[0]
        special_token_mask = torch.isin(input_ids, special_token_ids).sum(dim=1) < N
        if special_token_mask.sum() == 0:
            return input_ids
        input_ids[special_token_mask, -N:] = special_token_ids
        return input_ids

    def _tokenize_texts(
        self, texts: List[str], add_special_tokens: bool = False, special_tokens=None, tokenizer=None
    ) -> Dict[str, Any]:
        """
        Tokenize a list of texts with consistent parameters.

        Args:
            texts: List of texts to tokenize.
            add_special_tokens: Whether to add special tokens to the texts.
            special_tokens: List of special tokens to add to the texts.
            tokenizer: Tokenizer to use for tokenization.
        
        Returns:
            Dictionary containing the tokenized texts.
        """
        if special_tokens is None:
            special_tokens = self.special_tokens
        if tokenizer is None:
            tokenizer = self.tokenizer

        features = tokenizer(
            texts,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if add_special_tokens and len(special_tokens) > 0:
            special_token_ids = torch.LongTensor(
                tokenizer.convert_tokens_to_ids(special_tokens)
            )
            features["input_ids"] = self._add_special_tokens_if_needed(
                cast(torch.LongTensor, features["input_ids"]),
                special_token_ids,
            )

        return dict(features)

    def __call__(self, features: List[DataSample]) -> Dict[str, Any]:
        """Process a batch of features according to the specified training objectives.

        Args:
            features: List of DataSample objects containing questions and answers

        Returns:
            Dictionary containing processed features for each training objective
        """
        # Prepare texts
        query_texts = [
            sample.question.strip() + "".join(self.special_tokens)
            for sample in features
        ]
        answer_texts = [
            # TODO: hardcoded to llama format
            sample.answer.strip() + "<|end_of_text|>" for sample in features
        ]
        teacher_special_tokens = self.teacher_special_tokens if self.teacher_special_tokens is not None else self.special_tokens
        teacher_tokenizer = self.teacher_tokenizer if self.teacher_tokenizer is not None else self.tokenizer
        teacher_answer_texts = [
            sample.answer.strip() + "".join(teacher_special_tokens) for sample in features
        ]

        # Tokenize all text types
        query_features = self._tokenize_texts(query_texts, add_special_tokens=True, special_tokens=self.special_tokens)
        answer_features = self._tokenize_texts(answer_texts)
        repeat_answer_texts = [
            ("Repeat this text word by word. " + sample.answer.strip() + "".join(self.special_tokens)) for sample in features
        ]
        repeat_answer_features = self._tokenize_texts(
            repeat_answer_texts, add_special_tokens=True, special_tokens=self.special_tokens
        )
        teacher_answer_features = self._tokenize_texts(
            teacher_answer_texts, add_special_tokens=True, special_tokens=teacher_special_tokens, tokenizer=teacher_tokenizer
        )

        # Prepare labels
        labels = cast(torch.LongTensor, answer_features["input_ids"]).clone()
        labels = torch.where(
            labels == self.tokenizer.pad_token_id, torch.tensor(-100), labels
        )

        negative_query_features = None
        negative_answer_features = None
        negative_teacher_answer_features = None
        negative_repeat_answer_features = None
        negative_labels = None
        if features[0].negative_question is not None:
            negative_query_texts = [
                sample.negative_question.strip() + "".join(self.special_tokens) for sample in features
            ]
            negative_query_features = self._tokenize_texts(
                negative_query_texts, add_special_tokens=True, special_tokens=self.special_tokens
            )
            negative_answer_texts = [
                sample.negative_answer.strip() + "<|end_of_text|>" for sample in features
            ]
            negative_answer_features = self._tokenize_texts(
                negative_answer_texts
            )

            negative_repeat_answer_texts = [
                ("Repeat this text word by word. " + sample.negative_answer.strip() + "".join(self.special_tokens)) for sample in features
            ]
            negative_repeat_answer_features = self._tokenize_texts(
                negative_repeat_answer_texts, add_special_tokens=True, special_tokens=self.special_tokens
            )

            negative_teacher_answer_texts = [
                sample.negative_answer.strip() + "".join(teacher_special_tokens) for sample in features
            ]
            negative_teacher_answer_features = self._tokenize_texts(
                negative_teacher_answer_texts, add_special_tokens=True, special_tokens=teacher_special_tokens, tokenizer=teacher_tokenizer
            )

            negative_labels = cast(torch.LongTensor, negative_answer_features["input_ids"]).clone()
            negative_labels = torch.where(
                negative_labels == self.tokenizer.pad_token_id, torch.tensor(-100), negative_labels
            )

        # Combine all features
        return {
            **{f"query_{k}": v for k, v in query_features.items()},
            **{f"answer_{k}": v for k, v in answer_features.items()},
            **({f"repeat_answer_{k}": v for k, v in repeat_answer_features.items()} 
                if repeat_answer_features is not None else {}),
            **{f"teacher_answer_{k}": v for k, v in teacher_answer_features.items()},
            **({f"negative_query_{k}": v for k, v in negative_query_features.items()} 
                if negative_query_features is not None else {}),
            **({f"negative_answer_{k}": v for k, v in negative_answer_features.items()} 
                if negative_answer_features is not None else {}),
            **({f"negative_repeat_answer_{k}": v for k, v in negative_repeat_answer_features.items()} 
                if negative_repeat_answer_features is not None else {}),
            **({f"negative_teacher_answer_{k}": v for k, v in negative_teacher_answer_features.items()} 
                if negative_teacher_answer_features is not None else {}),
            "labels": labels,
            **({f"negative_labels": negative_labels} if negative_labels is not None else {}),
        }
