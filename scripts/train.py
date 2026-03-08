import logging
import os
import sys
from typing import Any, Dict, List, Tuple, Union, cast
import datetime

import datasets
import hydra
import numpy as np
import torch
import transformers
import yaml
from omegaconf import DictConfig, OmegaConf
from peft import TrainableTokensConfig, get_peft_model
from peft.peft_model import PeftConfig, PeftModel
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from transformers.integrations import is_wandb_available
from transformers.trainer_utils import (
    get_last_checkpoint,
)
from transformers.utils.versions import require_version

from llm2vec_gen.config import (
    CustomTrainingArguments,
    DataArguments,
    ModelArguments,
    RunArguments,
)
from llm2vec_gen.dataset import CustomCollator
from llm2vec_gen.dataset.utils import load_dataset, safe_split_name
from llm2vec_gen.losses import (
    compute_contrastive_loss,
    compute_margin_loss,
    compute_mse_loss,
    get_teacher_embeddings,
)
from llm2vec_gen.models import (
    EncoderDecoderModel,
    ProjectionModel,
    apply_peft,
)
from llm2vec_gen.trainer import CustomTrainer
from llm2vec_gen.utils import FILENAME_ATTRS_TO_EXCLUDE, save_args_to_yaml

require_version("datasets>=2.14.0", "To fix: pip install datasets --upgrade")

logger = logging.getLogger(__name__)


class LLM2VecGenTrainer(CustomTrainer):

    def compute_loss(
        self,
        model: EncoderDecoderModel,
        model_inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Compute training loss for the model.

        Handles multiple training objectives and MSE loss between encoder states.

        Args:
            model: Model to compute loss for
            model_inputs: Input tensors and data
            return_outputs: Whether to return model outputs along with loss

        Returns:
            Loss tensor if return_outputs is False, otherwise tuple of (loss, outputs)
        """

        # Handle non-MSE loss objectives
        student_model_inputs_names = (
            "query_input_ids",
            "query_attention_mask",
            "answer_input_ids",
            "answer_attention_mask",
            "labels",
        )
        student_model_inputs = {
            k: v for k, v in model_inputs.items() if k in student_model_inputs_names
        }
        student_encoder_states, outputs = model(**student_model_inputs)

        student_repeat_answer_model_inputs = {
            "query_input_ids": model_inputs["repeat_answer_input_ids"],
            "query_attention_mask": model_inputs["repeat_answer_attention_mask"],
            "answer_input_ids": model_inputs["answer_input_ids"],
            "answer_attention_mask": model_inputs["answer_attention_mask"],
            "labels": model_inputs["labels"],
        }

        if outputs is None:
            recon_loss = 0
            loss = 0
            assert self.recon_loss_weight == 0, "recon_loss_weight must be 0"
        else:
            loss = outputs["loss"]
            recon_loss = loss.detach().item()

        total_loss = 0
        if self.recon_loss_weight > 0:
            total_loss += self.recon_loss_weight * loss

        # Handle MSE loss if teacher model is available
        if self.teacher_model is not None:
            repeat_answer_student_encoder_states, repeat_answer_outputs = None, None

            teacher_encoder_states = get_teacher_embeddings(model_inputs["teacher_answer_input_ids"], model_inputs["teacher_answer_attention_mask"], self.teacher_model, student_encoder_states=student_encoder_states)
            total_student_q_teacher_d_loss, mse_loss = compute_mse_loss(student_encoder_states.mean(dim=1), teacher_encoder_states, mse_loss_weight=self.align_loss_weight)
            total_loss += total_student_q_teacher_d_loss
            
            if self.other_sub_losses_weight > 0:                
                repeat_answer_student_encoder_states, repeat_answer_outputs = model(**student_repeat_answer_model_inputs)
                total_student_d_teacher_d_loss, _ = compute_mse_loss(repeat_answer_student_encoder_states.mean(dim=1), teacher_encoder_states, mse_loss_weight=self.align_loss_weight)
                total_loss += self.other_sub_losses_weight * total_student_d_teacher_d_loss

            if self.use_hard_negatives:
                assert "negative_query_input_ids" in model_inputs
                negative_student_model_inputs = {
                    "query_input_ids": model_inputs["negative_query_input_ids"],
                    "query_attention_mask": model_inputs["negative_query_attention_mask"],
                    "answer_input_ids": model_inputs["negative_answer_input_ids"],
                    "answer_attention_mask": model_inputs["negative_answer_attention_mask"],
                    "labels": model_inputs["negative_labels"],
                }
                assert "negative_repeat_answer_input_ids" in model_inputs and "repeat_answer_input_ids" in model_inputs
                student_neg_repeat_answer_model_inputs = {
                    "query_input_ids": model_inputs["negative_repeat_answer_input_ids"],
                    "query_attention_mask": model_inputs["negative_repeat_answer_attention_mask"],
                    "answer_input_ids": model_inputs["negative_answer_input_ids"],
                    "answer_attention_mask": model_inputs["negative_answer_attention_mask"],
                    "labels": model_inputs["negative_labels"],
                }

                negative_student_encoder_states, negative_outputs = model(**negative_student_model_inputs)
                neg_teacher_encoder_states = get_teacher_embeddings(model_inputs["negative_teacher_answer_input_ids"], model_inputs["negative_teacher_answer_attention_mask"], self.teacher_model, student_encoder_states=negative_student_encoder_states)
                total_student_q_neg_teacher_d_neg_loss, neg_mse_loss = compute_mse_loss(negative_student_encoder_states.mean(dim=1), neg_teacher_encoder_states, mse_loss_weight=self.align_loss_weight)
                total_loss += total_student_q_neg_teacher_d_neg_loss


                if self.other_sub_losses_weight > 0:
                    negative_repeat_answer_student_encoder_states, negative_repeat_answer_outputs = model(**student_neg_repeat_answer_model_inputs)
                    total_student_d_neg_teacher_d_neg_loss, _ = compute_mse_loss(negative_repeat_answer_student_encoder_states.mean(dim=1), neg_teacher_encoder_states, mse_loss_weight=self.align_loss_weight)
                    total_loss += self.other_sub_losses_weight * total_student_d_neg_teacher_d_neg_loss

                if self.margin_loss_weight > 0:
                    margin_loss = compute_margin_loss(student_encoder_states.mean(dim=1), negative_student_encoder_states.mean(dim=1), margin_threshold=self.margin_threshold, gather_across_devices=torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
                    total_loss += self.margin_loss_weight * margin_loss

                if self.contrastive_loss_weight > 0:
                    if repeat_answer_student_encoder_states is None:
                        repeat_answer_student_encoder_states, repeat_answer_outputs = model(**student_repeat_answer_model_inputs)

                    contrastive_loss = compute_contrastive_loss(student_encoder_states.mean(dim=1), repeat_answer_student_encoder_states.mean(dim=1), negative_repeat_answer_student_encoder_states.mean(dim=1), scale=self.contrastive_scale, function=self.contrastive_function, gather_across_devices=torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1, hardness_weighting_alpha=self.contrastive_hardness_weighting_alpha)
                    total_loss += self.contrastive_loss_weight * contrastive_loss
            
        query_embeds = student_encoder_states
        with torch.no_grad():
            doc_embeds = (
                teacher_encoder_states
                if self.teacher_model is not None
                else self.accelerator.unwrap_model(model).encode(
                    model_inputs["teacher_answer_input_ids"],
                    model_inputs["teacher_answer_attention_mask"],
                    inference=True,
                )[0]
            )

            # if dim 1 is 1, squeeze it, else if dim 1 is the number of special tokens, take mean over dim 1
            if query_embeds.shape[1] > 1 and len(query_embeds.shape) > 2:
                query_embeds = query_embeds.mean(dim=1, keepdim=True)
            if doc_embeds.shape[1] > 1 and len(doc_embeds.shape) > 2:
                doc_embeds = doc_embeds.mean(dim=1, keepdim=True)

            query_embeds = query_embeds.squeeze(1)
            doc_embeds = doc_embeds.squeeze(1)

        # Log losses to wandb if available
        if (
            is_wandb_available()
            and self.args.report_to is not None
            and "wandb" in self.args.report_to
        ):
            import wandb

            # Calculate norms of the encoder states
            # Compute L2 norm along last dimension and average over batch
            doc_norm = torch.norm(doc_embeds, dim=-1).mean().item()
            query_norm = torch.norm(query_embeds, dim=-1).mean().item()

            # Calculate cosine similarities
            # First normalize the vectors
            doc_embeds_unit = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)
            query_embeds_unit = query_embeds / query_embeds.norm(dim=-1, keepdim=True)

            # Compute all pairwise cosine similarities
            cosine_sim = torch.matmul(
                doc_embeds_unit, query_embeds_unit.t()
            )  # (batch_size, batch_size)

            # Average positive pairs (diagonal elements)
            avg_positive_cosine = cosine_sim.diagonal().mean().item()

            # Average negative pairs (non-diagonal elements)
            # Create mask to exclude diagonal
            batch_size = cosine_sim.size(0)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=cosine_sim.device)
            avg_negative_cosine = cosine_sim[mask].mean().item()

            wandb.log(
                {
                    "train/recon_loss": recon_loss,
                    "train/align_loss": mse_loss.item()
                    if self.teacher_model is not None
                    else 0,
                    "train/neg_mse_loss": neg_mse_loss.item()
                    if self.use_hard_negatives and self.margin_loss_weight > 0
                    else 0,
                    "train/margin_loss": margin_loss.item()
                    if self.use_hard_negatives and self.margin_loss_weight > 0
                    else 0,
                    "train/contrastive_loss": contrastive_loss.item()
                    if self.use_hard_negatives and self.contrastive_loss_weight > 0
                    else 0,
                    "train/doc_norm": doc_norm,
                    "train/query_norm": query_norm,
                    "train/avg_positive_cosine": avg_positive_cosine,
                    "train/avg_negative_cosine": avg_negative_cosine,
                }
            )

        return cast(torch.Tensor, total_loss)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra config"""

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config_dict, dict):
        config_dict = {}

    # Create argument objects from config sections
    model_args = ModelArguments(
        **config_dict.get("model", {}), **config_dict.get("special_tokens", {})
    )
    data_args = DataArguments(**config_dict.get("data", {}))
    training_args = CustomTrainingArguments(**config_dict.get("training", {}))
    run_args = RunArguments(**config_dict.get("run", {}))
    # Create a unique output dir of each run
    # serialize all arguments to a single string
    meta_json = {
        **model_args.to_dict(),
        **data_args.to_dict(),
        **training_args.to_dict(),
        **run_args.to_dict(),
    }
    final_json = {
        x: meta_json[x] for x in meta_json if x not in FILENAME_ATTRS_TO_EXCLUDE
    }
    assert run_args.wandb_run_id is not None, "wandb_run_id is required for saving input config and checkpoints"
    wandb_id = run_args.wandb_run_id
    wandb_id_final = f"{wandb_id}-{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    with open(
        os.path.join(training_args.output_dir, f"{wandb_id}_input_config.yml"), "w"
    ) as f:
        yaml.dump(
            {
                **{"wandb_run_id": wandb_id_final},
                **final_json,
            },
            f,
        )

    training_args.output_dir = os.path.join(training_args.output_dir, wandb_id)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        if training_args.overwrite_output_dir:
            logger.info(
                f"Output directory ({training_args.output_dir}) already exists but we will overwrite the data."
            )
        else:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets

    kwargs = {
        "split": safe_split_name(data_args.dataset_name),
        "dataset_name": data_args.hf_dataset_name,
        "num_special_tokens": len(model_args.special_tokens),
    }
    if data_args.config is not None:
        kwargs["config"] = data_args.config
    train_dataset = load_dataset(data_args.dataset_name, **kwargs)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }

    teacher_tokenizer, teacher_special_tokens, teacher_model = None, None, None
    if model_args.pretrained_teacher_path is not None:
        peft_config = PeftConfig.from_pretrained(model_args.pretrained_teacher_path)
        base_model_name_or_path = peft_config.base_model_name_or_path
        teacher_tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_teacher_path if "mntp" in model_args.pretrained_teacher_path else base_model_name_or_path)
        if teacher_tokenizer.pad_token is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        teacher_tokenizer.padding_side = "left"
        teacher_special_tokens = []

        teacher_config = AutoConfig.from_pretrained(model_args.pretrained_teacher_path if "mntp" in model_args.pretrained_teacher_path else base_model_name_or_path, trust_remote_code=True)

        teacher_model = AutoModel.from_pretrained(
            model_args.pretrained_teacher_path if "mntp" in model_args.pretrained_teacher_path else base_model_name_or_path,
            trust_remote_code=True,
            config=teacher_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        teacher_model = PeftModel.from_pretrained(teacher_model, model_args.pretrained_teacher_path)
        teacher_model = teacher_model.merge_and_unload() if "mntp" in model_args.pretrained_teacher_path else teacher_model

        if model_args.pretrained_teacher_path_2 is not None:
            teacher_model = PeftModel.from_pretrained(teacher_model, model_args.pretrained_teacher_path_2)

        for name, param in teacher_model.named_parameters():
            param.requires_grad = False


    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = model_args.tokenizer_padding_side
    special_tokens = cast(List[str], model_args.special_tokens)

    config.tokenizer_original_num_tokens = len(tokenizer)
    if len(tokenizer) == config.vocab_size:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        config.num_tokens = len(tokenizer)
    else:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        config.num_tokens = config.vocab_size + len(set(special_tokens))
    config.num_special_tokens = len(special_tokens)

    config.special_tokens = tokenizer.convert_tokens_to_ids(special_tokens)
    config.special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    # TODO(mm): support additional models
    config.torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, cast(str, model_args.torch_dtype))
    )

    # initialize the encoder model
    # load the pretrained encoder model if provided
    enc_dec_model = None
    pretrained_encoder_path = training_args.pretrained_encoder_path is not None and os.path.exists(training_args.pretrained_encoder_path)
    if pretrained_encoder_path:
        enc_dec_model = EncoderDecoderModel.from_pretrained(training_args.pretrained_encoder_path)
    else:
        encoder_model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=config.torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
        encoder_model.resize_token_embeddings(len(tokenizer))
        if not training_args.use_peft_for_encoder:
            peft_config = TrainableTokensConfig(
                token_indices=config.special_tokens_ids,
                # TODO: assumes embedding matrix is called 'embed_tokens' - True for Llama and Qwen models
            )
            encoder_model = get_peft_model(encoder_model, peft_config=peft_config)
        else:
            encoder_model = apply_peft(encoder_model, special_tokens_ids=config.special_tokens_ids, lora_r=training_args.encoder_lora_r, lora_alpha=training_args.encoder_lora_alpha, lora_dropout=training_args.encoder_lora_dropout)

        # initialize the decoder model
        # if no pretrained decoder path is provided, use the encoder model
        if training_args.use_encoder_as_decoder:
            decoder_model = encoder_model
        else:
            decoder_model = AutoModelForCausalLM.from_pretrained(
                training_args.pretrained_decoder_path
                if training_args.pretrained_decoder_path
                else model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                torch_dtype=config.torch_dtype,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            )
            # will be frozen
            for name, param in decoder_model.named_parameters():
                param.requires_grad = False

        reconstruction_mlp_model = None
        alignment_mlp_model = None
        if enc_dec_model is None:
            if model_args.add_reconstruction_mlp:
                reconstruction_mlp_model = ProjectionModel(
                    input_dim=encoder_model.config.hidden_size,
                    output_dim=decoder_model.config.hidden_size,
                    dtype=encoder_model.config.torch_dtype,
                )
                reconstruction_mlp_model = cast(ProjectionModel, reconstruction_mlp_model)
                
            if model_args.add_alignment_mlp:
                alignment_mlp_model = ProjectionModel(
                    input_dim=encoder_model.config.hidden_size,
                    output_dim=teacher_model.config.hidden_size if teacher_model is not None else decoder_model.config.hidden_size,
                    dtype=encoder_model.config.torch_dtype,
                )
                alignment_mlp_model = cast(ProjectionModel, alignment_mlp_model)
        
        enc_dec_model = EncoderDecoderModel(
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            encoding_mode=model_args.encoding_mode,
            reconstruction_mlp=reconstruction_mlp_model,
            alignment_mlp=alignment_mlp_model,
            save_decoder=not training_args.use_encoder_as_decoder and not training_args.freeze_decoder_fully,
        )

    # Log number of trainable parameters
    trainable_params, all_params = enc_dec_model.get_nb_trainable_parameters()
    percent_trainable = 100 * trainable_params / all_params
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_params:,d} || % trainable: {percent_trainable:.4f}"
    )

    # Preprocessing the datasets. The collator takes care of tokenization
    data_collator = CustomCollator(
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        padding="longest",
        max_length=model_args.max_input_length,
        return_tensors="pt",
        teacher_special_tokens=teacher_special_tokens,
        teacher_tokenizer=teacher_tokenizer
    )

    # Initialize wandb manually
    if (
        is_wandb_available
        and training_args.report_to is not None
        and "wandb" in training_args.report_to
    ):
        import wandb

        # Initialize wandb
        wandb_config = {
            **model_args.to_dict(),
            **data_args.to_dict(),
            **training_args.to_dict(),
            "all_params": trainable_params,
            "trainable_params": all_params,
            "percent_trainable": percent_trainable,
        }
        wandb.init(
            config=wandb_config,
            id=wandb_id_final,
            group=run_args.wandb_run_group if run_args.wandb_run_group else None,
            name=run_args.wandb_run_name if run_args.wandb_run_name else wandb_id,
            notes=run_args.wandb_run_notes if run_args.wandb_run_notes else None,
            resume="allow",
            # settings=wandb.Settings(init_timeout=300),
        )
        # Append the run name to the output directory
        # training_args.output_dir = f"{training_args.output_dir}_{wandb.run.name}"

    # Create the output dir if it does not exist yet
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Save all arguments to single yaml file
    for args in [run_args, data_args, model_args, training_args]:
        save_args_to_yaml(args, training_args.output_dir, name="run_config.yml")

    # Initialize our Trainer
    trainer = LLM2VecGenTrainer(
        model=enc_dec_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        recon_loss_weight=training_args.recon_loss_weight,
        align_loss_weight=training_args.align_loss_weight,
        use_hard_negatives=training_args.use_hard_negatives,
        margin_loss_weight=training_args.margin_loss_weight,
        negative_loss_weight=training_args.negative_loss_weight,
        infonce_loss_weight=training_args.infonce_loss_weight,
        margin_threshold=training_args.margin_threshold,
        contrastive_loss_weight=training_args.contrastive_loss_weight,
        contrastive_scale=training_args.contrastive_scale,
        contrastive_function=training_args.contrastive_function,
        contrastive_hardness_weighting_alpha=training_args.contrastive_hardness_weighting_alpha,
        other_sub_losses_weight=training_args.other_sub_losses_weight,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
