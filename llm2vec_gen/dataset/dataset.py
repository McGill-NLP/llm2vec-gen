from typing import Any, Dict, Iterator, List, cast
import logging
from datasets import Dataset, load_dataset

from .base_dataset import BaseDataset, DataSample

logger = logging.getLogger(__name__)

class Dataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str = "McGill-NLP/llm2vec-gen-tulu",  # TODO: change to the actual dataset name
        split: str = "original",
        **kwargs: Dict[str, Any]
    ):
        """
        Initialize the dataset wrapper and immediately load data.

        Args:
            dataset_name: Hugging Face dataset identifier to load.
            split: Dataset split to use (currently only `"train"` is supported).
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.load_data()

    def load_data(self):
        """
        Load samples from the configured Hugging Face dataset into `self.data`.

        Each example is expected to contain at least `question` and `answer`
        fields, and may optionally include `id`, `negative_question`, and
        `negative_answer`.
        """
        ds = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Loaded dataset {self.dataset_name} from {self.split} generations.")
        ds = cast(Dataset, ds)

        for idx, example in enumerate(ds):
            example_dict: Dict[str, Any] = dict(example)
            assert "question" in example_dict and "answer" in example_dict, "Question and answer must be present in the dataset attributes"
            self.data.append(
                DataSample(
                    original_id=example_dict.get("id"),
                    id_=idx,
                    question=example_dict["question"],
                    answer=example_dict["answer"],
                    negative_question=example_dict.get("negative_question"),
                    negative_answer=example_dict.get("negative_answer")
                )
            )
    
    def batch_dataset(self, batch_size: int) -> Iterator[List[DataSample]]:
        """
        Yield the dataset in consecutive batches of `DataSample` objects.

        Args:
            batch_size: Number of samples to include in each batch.

        Returns:
            List of `DataSample` instances of length up to `batch_size`.
        """
        for i in range(0, len(self.data), batch_size):
            yield self.data[i:i + batch_size]

