from dataclasses import asdict, dataclass
from typing import List, Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = None
    pretrained_teacher_path: Optional[str] = None
    pretrained_teacher_path_2: Optional[str] = None
    tokenizer_name: Optional[str] = None
    tokenizer_padding_side: Optional[str] = None
    cache_dir: Optional[str] = None
    model_revision: str = "main"
    token: Optional[str] = None
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    torch_dtype: Optional[str] = None
    low_cpu_mem_usage: bool = False
    special_tokens: Optional[List[str]] = None
    special_token_config_name: Optional[str] = None
    max_input_length: int = 512
    token_initialization: str = "random"
    add_reconstruction_mlp: bool = False
    add_alignment_mlp: bool = False
    encoding_mode: str = "last_10_tokens"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DataArguments:
    dataset_name: Optional[str] = None
    hf_dataset_name: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    split: str = "train"
    config: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    use_peft_for_encoder: bool = False
    encoder_lora_r: int = 0
    encoder_lora_alpha: int = 0
    encoder_lora_dropout: float = 0.05
    pretrained_encoder_path: Optional[str] = None
    pretrained_decoder_path: Optional[str] = None
    freeze_peft_on_encoder: bool = False
    use_encoder_as_decoder: bool = False
    freeze_decoder_fully: bool = False
    freeze_shared_token_weights: bool = False
    recon_loss_weight: float = 1.0
    align_loss_weight: float = 0.0
    contrastive_learning: bool = False
    use_hard_negatives: bool = False
    margin_loss_weight: float = 0.0
    margin_threshold: float = 0.0
    negative_loss_weight: float = 0.0
    infonce_loss_weight: float = 0.0
    contrastive_loss_weight: float = 0.0
    contrastive_scale: float = 20.0
    contrastive_function: str = "cosine"
    contrastive_hardness_weighting_alpha: float = 0.0
    other_sub_losses_weight: float = 0.0

    stop_after_steps: Optional[int] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class RunArguments:
    wandb_run_id: Optional[str] = None
    wandb_run_group: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_run_notes: Optional[str] = None

    def to_dict(self):
        return asdict(self)
