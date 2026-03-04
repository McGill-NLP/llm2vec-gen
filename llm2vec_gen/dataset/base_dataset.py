from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset


@dataclass
class DataSample:
    id_: int
    question: str
    answer: str
    original_id: Optional[str] = None
    negative_question: Optional[str] = None
    negative_answer: Optional[str] = None


class BaseDataset(Dataset):
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_data(self):
        raise NotImplementedError("Subclasses must implement this method")
