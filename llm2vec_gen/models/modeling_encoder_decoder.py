import json
import os
import random
import re
from typing import Any, Dict, Optional, Tuple

import torch
from peft.peft_model import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from .utils import safe_load_config

def prepend_prompt_to_left_padded_tok_embeds(
    tok_embeds: torch.FloatTensor,
    dec_attention_mask: torch.LongTensor,
    prompt_embeds: torch.FloatTensor,
    labels: torch.LongTensor,
) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
    """
    Prepend prompt_embeds after existing PADs in a left-padded decoder batch.

    Args:
      tok_embeds:    (B, T, H) — token embeddings for decoder_input_ids, including PADs (left-padded).
      dec_attention_mask: (B, T) — with 0 for PAD, 1 for real tokens.
      prompt_embeds: (B, K, H) — the soft prompt embeddings from encoder (fixed K).

    Returns:
      inputs_embeds: (B, T + K_max, H) — new embeddings with PADs, then prompt, then real tokens,
                     padded (on the right) so all samples share same length.
      attention_mask: (B, T + K_max) — 0 for PADs (original), 1 for prompt & real tokens, 0 for any new padding on right.
    """
    bsz, T, H = tok_embeds.shape
    K = prompt_embeds.size(1)
    device = tok_embeds.device
    dtype = tok_embeds.dtype

    new_embeds_list = []
    new_masks_list = []
    new_labels_list = []

    for i in range(bsz):
        mask_i = dec_attention_mask[i]         # (T,)
        emb_i = tok_embeds[i]                  # (T, H)
        labels_i = labels[i]                  # (T,)

        # Separate PAD embeddings vs real-token embeddings
        pad_mask = (mask_i == 0)
        real_mask = (mask_i == 1)

        pad_emb = emb_i[pad_mask]              # (num_pad, H)
        real_emb = emb_i[real_mask]            # (num_real, H)

        # Build: [pad_emb] + [prompt_embeds[i]] + [real_emb]
        new_emb_i = torch.cat([pad_emb, prompt_embeds[i], real_emb], dim=0)
        # Build corresponding attention mask: 0 for pad, 1 for prompt, 1 for real tokens
        new_mask_i = torch.cat([
            torch.zeros(pad_emb.size(0), dtype=dec_attention_mask.dtype, device=device),
            torch.ones(K, dtype=dec_attention_mask.dtype, device=device),
            torch.ones(real_emb.size(0), dtype=dec_attention_mask.dtype, device=device),
        ], dim=0)

        pad_labels = labels_i[pad_mask]
        real_labels = labels_i[real_mask]

        new_labels_i = torch.cat([
            pad_labels,
            torch.full((K,), -100, dtype=labels.dtype, device=device),
            real_labels,
        ], dim=0)

        new_embeds_list.append(new_emb_i)
        new_masks_list.append(new_mask_i)
        new_labels_list.append(new_labels_i)

    return (
        torch.stack(new_embeds_list, dim=0),
        torch.stack(new_masks_list, dim=0),
        torch.stack(new_labels_list, dim=0),
    )


class ProjectionModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dtype: torch.dtype = torch.float32,
        size: int = 1,
        pooling_mode: Optional[str] = "linear",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = dtype
        self.size = size
        self.pooling_mode = pooling_mode
        if pooling_mode.endswith("linear"):
            if size == 1:
                self.projection = torch.nn.Linear(input_dim, output_dim)
            else:
                self.projection = torch.nn.ModuleList(
                    [torch.nn.Linear(input_dim, output_dim) for _ in range(size)]
                )
            self.projection = self.projection.to(dtype=dtype)
        else:
            raise ValueError(f"Invalid pooling mode: {pooling_mode}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pooling_mode == "linear":
            if isinstance(self.projection, torch.nn.ModuleList):
                return torch.cat(
                    [
                        self.projection[i](hidden_states[:, i]).unsqueeze(1)
                        for i in range(hidden_states.shape[1])
                    ],
                    dim=1,
                )

            return self.projection(hidden_states)

        elif self.pooling_mode == "mean_linear":
            hidden_states = hidden_states.reshape(
                hidden_states.shape[0], -1, self.input_dim
            )
            hidden_states = torch.mean(hidden_states, dim=1, keepdim=True)
            if isinstance(self.projection, torch.nn.ModuleList):
                return torch.cat(
                    [
                        self.projection[i](hidden_states[:, i]).unsqueeze(1)
                        for i in range(hidden_states.shape[1])
                    ],
                    dim=1,
                )

            return self.projection(hidden_states)

        import re

        first_n_pattern = re.match(r"first_(\d+)_linear", self.pooling_mode)
        if first_n_pattern:
            hidden_states = hidden_states[:, :, : self.input_dim]

            if isinstance(self.projection, torch.nn.ModuleList):
                return torch.cat(
                    [
                        self.projection[i](hidden_states[:, i]).unsqueeze(1)
                        for i in range(hidden_states.shape[1])
                    ],
                    dim=1,
                )

            return self.projection(hidden_states)

    def save(self, path: str):
        # Save model state dict and config together
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim,
                    "dtype": str(self.dtype),
                    "size": self.size,
                    "pooling_mode": self.pooling_mode,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, size: Optional[int] = None):
        checkpoint = torch.load(path)
        config = checkpoint["config"]
        dtype = getattr(torch, config["dtype"].replace("torch.", ""))  # recover dtype
        model = cls(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            dtype=dtype,
            size=config.get("size", size if size is not None else 1),
            pooling_mode=config.get("pooling_mode", "linear"),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model


class EncoderDecoderModel(torch.nn.Module):
    def __init__(
        self,
        encoder_model: PeftModel,
        decoder_model: AutoModelForCausalLM|PeftModel,
        encoding_mode: str = "last_10_tokens",
        reconstruction_mlp: Optional[ProjectionModel] = None,
        alignment_mlp: Optional[ProjectionModel] = None,
        save_decoder: bool = False,
    ):
        super().__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.encoding_mode = encoding_mode
        self.reconstruction_mlp = reconstruction_mlp
        self.alignment_mlp = alignment_mlp
        self.save_decoder = save_decoder
        self.config = encoder_model.config

    def __call__(
        self,
        query_input_ids: torch.LongTensor,
        query_attention_mask: torch.Tensor,
        answer_input_ids: Optional[torch.LongTensor] = None,
        answer_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # decoder_inputs_embeds is the hidden states before applying the encoding-specific projection model
        hidden_states, hidden_states_attention_mask, decoder_input_hidden_states = (
            self.encode(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                return_decoder_inputs=True,
            )
        )

        decoder_input_embeds = self.decoder_model.get_input_embeddings()(answer_input_ids)

        decoder_input_embeds, decoder_attention_mask, decoder_labels = prepend_prompt_to_left_padded_tok_embeds(
            tok_embeds=decoder_input_embeds,
            dec_attention_mask=answer_attention_mask,
            prompt_embeds=decoder_input_hidden_states,
            labels=labels,
        )

        decoder_outputs = self.decoder_model(
            inputs_embeds=decoder_input_embeds,
            attention_mask=decoder_attention_mask,
            labels=decoder_labels,
        )
        return hidden_states, decoder_outputs

    def get_nb_trainable_parameters(self):
        trainable_params, all_params = 0, 0
        # Count encoder parameters
        for p in self.encoder_model.parameters():
            if p.requires_grad:
                trainable_params += p.numel()
            all_params += p.numel()

        if self.reconstruction_mlp is not None:
            for p in self.reconstruction_mlp.parameters():
                if p.requires_grad:
                    trainable_params += p.numel()
                all_params += p.numel()

        if self.alignment_mlp is not None:
            for p in self.alignment_mlp.parameters():
                if p.requires_grad:
                    trainable_params += p.numel()
                all_params += p.numel()

        # Only count decoder parameters if it's a different model
        if (self.decoder_model is not self.encoder_model) and self.save_decoder:
            for p in self.decoder_model.parameters():
                if p.requires_grad:
                    trainable_params += p.numel()
                all_params += p.numel()

        return trainable_params, all_params

    def save_pretrained(
        self,
        output_path,
        state_dict=None,
        safe_serialization=True,
    ):
        if self.save_decoder:
            os.makedirs(os.path.join(output_path, "decoder"), exist_ok=True)
            self.decoder_model.save_pretrained(
                os.path.join(output_path, "decoder"),
                state_dict=state_dict,
                safe_serialization=safe_serialization,
            )

        self.encoder_model.save_pretrained(
            os.path.join(output_path, "encoder"),
            state_dict=state_dict,
            safe_serialization=safe_serialization,
        )
        self.encoder_model.config.save_pretrained(os.path.join(output_path, "encoder"))

        if self.reconstruction_mlp is not None:
            self.reconstruction_mlp.save(
                os.path.join(output_path, "reconstruction_mlp"),
            )

        if self.alignment_mlp is not None:
            self.alignment_mlp.save(
                os.path.join(output_path, "alignment_mlp"),
            )

        config = {
            "encoding_mode": self.encoding_mode,
            "save_decoder": self.save_decoder,
        }
        with open(
            os.path.join(output_path, "encoder_decoder_config.json"), "w"
        ) as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        causal_lm_decoder: bool = False,
        **kwargs: Any,
    ) -> "EncoderDecoderModel":
        """
        Load an EncoderDecoderModel from a directory saved with save_pretrained.

        Args:
            path: Directory containing the saved encoder, optional decoder, and MLPs.
            **kwargs: Passed through to PeftModel.from_pretrained when loading encoder/decoder.

        Returns:
            Loaded EncoderDecoderModel.
        """

        tokenizer = AutoTokenizer.from_pretrained(path)
        is_local_dir = os.path.isdir(path)

        run_config = None

        # Case 1: `path` is a local directory that already contains `run_config.yml`
        if is_local_dir:
            local_run_config = os.path.join(path, "run_config.yml")
            if os.path.isfile(local_run_config):
                run_config = safe_load_config(local_run_config)
            else:
                # Backwards‑compatible fall‑back for training checkpoints
                match = re.search(r"outputs/([^/]+)/checkpoint-", path)
                if match:
                    run_id = match.group(1)
                    run_config = safe_load_config(f"outputs/{run_id}/run_config.yml")
        else:
            # Likely a Hub repo ID: try fetching `run_config.yml` from the repo
            try:
                from huggingface_hub import hf_hub_download

                run_config_path = hf_hub_download(
                    repo_id=path,
                    filename="run_config.yml",
                )
                run_config = safe_load_config(run_config_path)
            except Exception as e:  # pragma: no cover - defensive
                raise ValueError(
                    f"Could not load 'run_config.yml' for model path '{path}'. "
                    "Ensure that 'run_config.yml' is present in the directory or Hugging Face repo."
                ) from e

        if run_config is None:
            raise ValueError(
                f"Could not find 'run_config.yml' for model path '{path}'. "
                "Expected it either in the given directory, or under 'outputs/<run_id>/run_config.yml'."
            )

        # Load encoder base model and apply PEFT adapters stored under encoder/
        encoder_model = AutoModel.from_pretrained(
            run_config["model_name_or_path"],
            torch_dtype=run_config["torch_dtype"],
        )
        encoder_model.resize_token_embeddings(len(tokenizer))

        # Config & adapters live under the "encoder" subfolder (local or HF repo)
        config = AutoConfig.from_pretrained(path, subfolder="encoder")
        encoder_model = PeftModel.from_pretrained(
            encoder_model,
            path,
            subfolder="encoder",
            **kwargs,
        )
        config.num_special_tokens = len(run_config["special_tokens"])
        encoder_model.config = config

        # Default values in case encoder_decoder_config.json is absent
        encoding_mode = run_config.get("encoding_mode", "last_10_tokens")
        _save_decoder = False

        if is_local_dir:
            config_path = os.path.join(path, "encoder_decoder_config.json")
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    saved_config: Dict[str, Any] = json.load(f)
                    encoding_mode = saved_config.get("encoding_mode", encoding_mode)
                    _save_decoder = saved_config.get("save_decoder", _save_decoder)
        else:
            try:
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(
                    repo_id=path,
                    filename="encoder_decoder_config.json",
                )
                with open(config_path) as f:
                    saved_config = json.load(f)
                    encoding_mode = saved_config.get("encoding_mode", encoding_mode)
                    _save_decoder = saved_config.get("save_decoder", _save_decoder)
            except Exception:
                # If this file is missing from the repo, fall back to defaults.
                pass

        # Decoder: either share encoder weights or load a separate causal LM + adapters
        if not _save_decoder:
            if not causal_lm_decoder:
                decoder_model = encoder_model
            else:
                decoder_model = AutoModelForCausalLM.from_pretrained(
                    run_config["model_name_or_path"],
                    torch_dtype=run_config["torch_dtype"],
                )
        else:
            # Decoder PEFT adapters are saved under "decoder" subfolder
            base_decoder = AutoModelForCausalLM.from_pretrained(
                run_config["model_name_or_path"],
                torch_dtype=run_config["torch_dtype"],
            )
            decoder_model = PeftModel.from_pretrained(
                base_decoder,
                path,
                subfolder="decoder",
                **kwargs,
            )

        # Optional projection heads: load from local directory or HF repo
        reconstruction_mlp = None
        if is_local_dir:
            reconstruction_mlp_path = os.path.join(path, "reconstruction_mlp")
            if os.path.isfile(reconstruction_mlp_path):
                reconstruction_mlp = ProjectionModel.load(reconstruction_mlp_path)
        else:
            try:
                from huggingface_hub import hf_hub_download

                reconstruction_mlp_path = hf_hub_download(
                    repo_id=path,
                    filename="reconstruction_mlp",
                )
                reconstruction_mlp = ProjectionModel.load(reconstruction_mlp_path)
            except Exception:
                pass

        alignment_mlp = None
        if is_local_dir:
            alignment_mlp_path = os.path.join(path, "alignment_mlp")
            if os.path.isfile(alignment_mlp_path):
                alignment_mlp = ProjectionModel.load(alignment_mlp_path)
        else:
            try:
                from huggingface_hub import hf_hub_download

                alignment_mlp_path = hf_hub_download(
                    repo_id=path,
                    filename="alignment_mlp",
                )
                alignment_mlp = ProjectionModel.load(alignment_mlp_path)
            except Exception:
                pass

        return cls(
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            encoding_mode=encoding_mode,
            reconstruction_mlp=reconstruction_mlp,
            alignment_mlp=alignment_mlp,
            save_decoder=_save_decoder,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.encoder_model, "gradient_checkpointing_enable"):
            self.encoder_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        if self.decoder_model is not self.encoder_model and hasattr(
            self.decoder_model, "gradient_checkpointing_enable"
        ):
            self.decoder_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

        if self.reconstruction_mlp is not None and hasattr(
            self.reconstruction_mlp, "gradient_checkpointing_enable"
        ):
            self.reconstruction_mlp.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

        if self.alignment_mlp is not None and hasattr(
            self.alignment_mlp, "gradient_checkpointing_enable"
        ):
            self.alignment_mlp.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def encode(
        self,
        query_input_ids,
        query_attention_mask,
        return_decoder_inputs=False,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
    ):

        def _encode(
            query_input_ids: torch.LongTensor,
            query_attention_mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            encoder_outputs = self.encoder_model(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
            )

            hidden_states = encoder_outputs.last_hidden_state
            last_n_pattern = re.match(r"last_(\d+)_tokens", self.encoding_mode)

            if last_n_pattern:
                n_tokens = int(last_n_pattern.group(1))
                # Validate that n_tokens doesn't exceed number of special tokens
                if n_tokens > self.encoder_model.config.num_special_tokens:
                    raise ValueError(
                        f"Number of tokens ({n_tokens}) in {self.encoding_mode} cannot exceed "
                        f"total number of special tokens ({self.encoder_model.config.num_special_tokens})"
                    )
                hidden_states = hidden_states[:, -n_tokens:, :] # Shape: [batch_size, n_tokens, hidden_size]
                hidden_states_attention_mask = query_attention_mask[:, -n_tokens:]  # Shape: [batch_size, n_tokens]
            else:
                raise ValueError(f"Invalid encoding mode: {self.encoding_mode}")

            return hidden_states, hidden_states_attention_mask
    
        if decoder_inputs_embeds is None:
            decoder_input_hidden_states, hidden_states_attention_mask = _encode(query_input_ids, query_attention_mask)
            if self.reconstruction_mlp is not None:
                decoder_input_hidden_states = self.reconstruction_mlp(decoder_input_hidden_states)
        else:
            decoder_input_hidden_states = decoder_inputs_embeds
            hidden_states_attention_mask = None
        encoder_hidden_states = decoder_input_hidden_states
        if self.alignment_mlp is not None:
            encoder_hidden_states = self.alignment_mlp(
                encoder_hidden_states
            )

        if return_decoder_inputs:
            return encoder_hidden_states, hidden_states_attention_mask, decoder_input_hidden_states
        return encoder_hidden_states, hidden_states_attention_mask

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 20,
        return_embeddings: bool = False,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
    ):
        encoder_hidden_states, hidden_states_attention_mask, decoder_inputs_embeds = (
            self.encode(
                query_input_ids=input_ids,
                query_attention_mask=attention_mask,
                return_decoder_inputs=True,
                decoder_inputs_embeds=decoder_inputs_embeds,
            )
        )

        decoded_ids = torch.tensor([], device=input_ids.device, dtype=torch.long)
        decoder_attention_mask = torch.ones(decoder_inputs_embeds.shape[0], decoder_inputs_embeds.shape[1], dtype=torch.long, device=decoder_inputs_embeds.device)

        eos_token_ids = [self.config.eos_token_id] if isinstance(self.config.eos_token_id, int) else self.config.eos_token_id
        if isinstance(self.config, LlamaConfig):
            eos_token_ids += [128001]
        if isinstance(self.config, Qwen2Config) or isinstance(self.config, Qwen3Config):
            eos_token_ids += [151643]

        for i in range(max_new_tokens):
            # Decode
            decoder_outputs = self.decoder_model(
                inputs_embeds=decoder_inputs_embeds,
                attention_mask=decoder_attention_mask,
                labels=None,
            )
            # Extract predicted token
            next_token_logits = decoder_outputs.logits[:, -1, :].clone()
            next_token_id = next_token_logits.argmax(dim=-1)[:, None]
            # Concat predicted token with preceding decoder input ids
            next_token_embeds = self.decoder_model.get_input_embeddings()(next_token_id)
            decoder_inputs_embeds = torch.cat([decoder_inputs_embeds, next_token_embeds], dim=1)
            decoded_ids = torch.cat([decoded_ids, next_token_id], dim=1)
            decoder_attention_mask = torch.cat([decoder_attention_mask, torch.ones(next_token_id.shape[0], 1, dtype=torch.long, device=decoder_inputs_embeds.device)], dim=1)
            del decoder_outputs
            if (
                next_token_id.shape[0] == 1
                and next_token_id[0][0].item() in eos_token_ids
            ):
                break
        if return_embeddings:
            return decoded_ids, encoder_hidden_states, hidden_states_attention_mask
        return decoded_ids
