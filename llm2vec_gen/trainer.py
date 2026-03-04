import logging
import os
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
from torch.nn import ModuleList
from transformers.trainer import TRAINING_ARGS_NAME, Trainer, _is_peft_model, load_sharded_checkpoint
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
)
from transformers import AutoModel
from peft import PeftModel

from llm2vec_gen.models import EncoderDecoderModel


logger = logging.getLogger(__name__)


class StopAfterStepsCallback(TrainerCallback):
    """
    Simple callback that stops training after a fixed number of update steps
    without modifying the scheduler or max_steps.
    """

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if self.max_steps > 0 and state.global_step >= self.max_steps:
            control.should_training_stop = True
        return control


class CustomTrainer(Trainer):
    """Custom trainer class that extends Huggingface's Trainer for specialized training and evaluation.

    This trainer implements custom functionality for:
    - Multiple training objectives (generative and autoencoding)
    - MSE loss between encoder states
    - Model checkpointing and loading
    """

    def __init__(
        self,
        *args,
        teacher_model: Optional[PeftModel|AutoModel] = None,
        recon_loss_weight: float = 1.0,
        align_loss_weight: float = 0.0,
        use_hard_negatives: bool = False,
        contrastive_learning: bool = False,
        negative_loss_weight: float = 0.0,
        infonce_loss_weight: float = 0.0,
        margin_loss_weight: float = 0.0,
        margin_threshold: float = 0.0,
        contrastive_loss_weight: float = 0.0,
        contrastive_scale: float = 1.0,
        contrastive_function: str = "cosine",
        contrastive_hardness_weighting_alpha: float = 0.0,
        other_sub_losses_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.recon_loss_weight = recon_loss_weight
        self.align_loss_weight = align_loss_weight
        self.use_hard_negatives = use_hard_negatives

        self.contrastive_learning = contrastive_learning
        self.negative_loss_weight = negative_loss_weight
        self.infonce_loss_weight = infonce_loss_weight
        self.margin_loss_weight = margin_loss_weight
        self.margin_threshold = margin_threshold
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_scale = contrastive_scale
        self.contrastive_function = contrastive_function
        self.contrastive_hardness_weighting_alpha = contrastive_hardness_weighting_alpha
        self.other_sub_losses_weight = other_sub_losses_weight

        # Optional: stop training after a fixed number of steps without changing max_steps/scheduler
        stop_after_steps = getattr(self.args, "stop_after_steps", None)
        if stop_after_steps and stop_after_steps > 0:
            self.add_callback(StopAfterStepsCallback(stop_after_steps))

    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model = cast(EncoderDecoderModel, self.model)

        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _issue_warnings_after_load(self, load_result):
        if len(load_result.missing_keys) != 0:
            if self.model.encoder_model._keys_to_ignore_on_save is not None and set(
                load_result.missing_keys
            ) == set(self.model.encoder_model._keys_to_ignore_on_save):
                self.model.encoder_model.tie_weights()
            else:
                logger.warning(
                    f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}."
                )
        if len(load_result.unexpected_keys) != 0:
            logger.warning(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        adapter_safe_weights_file = os.path.join(
            resume_from_checkpoint, "encoder", ADAPTER_SAFE_WEIGHTS_NAME
        )
        safe_weights_file = os.path.join(resume_from_checkpoint, "encoder", SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(
            resume_from_checkpoint, "encoder", SAFE_WEIGHTS_INDEX_NAME
        )

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    safe_weights_file,
                    safe_weights_index_file,
                    adapter_safe_weights_file,
                ]
            )
        ):
            raise ValueError(
                f"Can't find a valid checkpoint at {resume_from_checkpoint}"
            )

        logger.info(f"Loading model from {resume_from_checkpoint}.")
        self.model = EncoderDecoderModel.from_pretrained(resume_from_checkpoint)

    def compute_loss(
        self,
        model: EncoderDecoderModel,
        model_inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        raise NotImplementedError("Not implemented")
