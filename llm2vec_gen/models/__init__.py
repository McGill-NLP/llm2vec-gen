"""
Models module for llm2vec-gen.
This module contains model implementations and related utilities.
"""

# Import key classes/functions to expose at package level
from typing import List, Optional, cast, Union

from peft import LoraConfig, PeftModel, get_peft_model
import torch
from transformers import AutoTokenizer

from .modeling_encoder_decoder import EncoderDecoderModel, ProjectionModel

__all__ = [
    "EncoderDecoderModel",
    "ProjectionModel",
    "apply_peft",
    "LLM2VecGenModel",
]

class LLM2VecGenModel:
        def __init__(self, model, tokenizer, device=None):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        @classmethod
        def from_pretrained(cls, name_or_path: str, **kwargs):
            tok = AutoTokenizer.from_pretrained(name_or_path)
            mdl = EncoderDecoderModel.from_pretrained(name_or_path, causal_lm_decoder=True, **kwargs)
            return cls(mdl, tok)

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

        @torch.no_grad()
        def encode(self, texts: Union[str, List[str]], max_length: int=512, get_recon_hidden_states: bool=False):
            """
            Inputs:
              - texts (str | List[str]): Input text(s). Special tokens from
                `tokenizer.additional_special_tokens` are appended internally.
              - max_length (int): Max token length for tokenization (padding+truncation).
              - get_recon_hidden_states (bool): If True, also return decoder inputs
                ("reconstruction hidden states") which can be used for `generate(...)`.

            Outputs:
              - if get_recon_hidden_states=False:
                  embeddings (torch.Tensor): [batch_size, hidden_dim]
              - if get_recon_hidden_states=True:
                  (embeddings, recon_hidden_states)
                    - embeddings (torch.Tensor): [batch_size, hidden_dim]
                    - recon_hidden_states (torch.Tensor): typically
                      [batch_size, compression_token_count, hidden_dim]
            """
            special_tokens = self.tokenizer.additional_special_tokens
            if isinstance(texts, str):
                texts = [texts]
            texts = [text + "".join(special_tokens) for text in texts]
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            special_token_ids = torch.LongTensor(
                self.tokenizer.convert_tokens_to_ids(special_tokens)
            )
            input_ids = cast(torch.LongTensor, batch["input_ids"])
            attention_mask = cast(torch.Tensor, batch["attention_mask"])

            input_ids = LLM2VecGenModel._add_special_tokens_if_needed(
                input_ids,
                special_token_ids,
            )

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            if get_recon_hidden_states:
                h, _, dec = self.model.encode(
                    query_input_ids=input_ids,
                    query_attention_mask=attention_mask,
                    return_decoder_inputs=get_recon_hidden_states,
                )
                return h.mean(dim=1), dec

            h, _ = self.model.encode(
                query_input_ids=input_ids,
                query_attention_mask=attention_mask,
                return_decoder_inputs=get_recon_hidden_states,
            )
            return h.mean(dim=1)  # [batch, hidden]

        @torch.no_grad()
        def generate(self, input_text: str="", max_new_tokens: int=256, get_align_hidden_states: bool=False, recon_hidden_states: Optional[torch.Tensor]=None):
            """
            Inputs:
              - input_text (str): Optional input text. Special tokens from
                `tokenizer.additional_special_tokens` are appended internally.
                If recon_hidden_states is not provided, this input text is used to 
                generate the response.
              - max_new_tokens (int): Maximum number of new tokens to generate.
              - get_align_hidden_states (bool): If True, also return the final embedding
                of the input text.
              - recon_hidden_states (torch.Tensor | None): Decoder input embeddings used
                for generation. If provided, it will override the input text. Expected shape:
                  - [1, compression_token_count, hidden_dim]

            Outputs:
              - if get_align_hidden_states=False:
                  output_text (str)
              - if get_align_hidden_states=True:
                  (output_text, embedding)
                    - output_text (str)
                    - embedding (torch.Tensor): [1, hidden_dim]
            """
            assert isinstance(input_text, str), "input_text must be a string"
            if len(recon_hidden_states.shape) == 2:
                recon_hidden_states = recon_hidden_states.unsqueeze(0)
            assert len(recon_hidden_states.shape) == 3 and recon_hidden_states.shape[0] == 1, "recon_hidden_states must have shape [1, compression token size, hidden size]"
            special_tokens = self.tokenizer.additional_special_tokens
            input_text_with_special_tokens = input_text.strip() + "".join(
                special_tokens
            )
            input_ids = self.tokenizer([input_text_with_special_tokens], return_tensors="pt")
            input_ids = input_ids.to("cuda")

            output_ids = self.model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                decoder_inputs_embeds=recon_hidden_states,
                return_embeddings=get_align_hidden_states,
            )
            hidden_states = None
            if get_align_hidden_states:
                output_ids, hidden_states, hidden_states_attention_mask = output_ids

            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, max_length=max_new_tokens
            )
            if get_align_hidden_states:
                return output_text[0], hidden_states.mean(dim=1)
            return output_text[0]


def apply_peft(
    model,
    lora_r: int = 128,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    special_tokens_ids: Optional[List[int]] = None,
) -> PeftModel:
    """Apply Parameter-Efficient Fine-Tuning (PEFT) to the model using LoRA.

    This function configures and applies LoRA (Low-Rank Adaptation) to specific modules
    in the model. For LLaMA and Mistral models, it targets attention and feed-forward
    network components.

    Args:
        model: The base model to apply PEFT to.
        lora_r (int, optional): Rank of the LoRA update matrices. Defaults to 8.
        lora_alpha (int, optional): Scaling factor for the LoRA updates. Defaults to 16.
        lora_dropout (float, optional): Dropout probability for LoRA layers. Defaults to 0.05.

    Returns:
        PeftModel: The model wrapped with LoRA adapters.

    """
    if model.config.__class__.__name__ in ["LlamaConfig", "Qwen3Config"]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    else:
        raise ValueError(f"Unsupported model config: {model.config.__class__.__name__}")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
        # TODO: assumes embedding matrix is called 'embed_tokens' - True for Llama and Qwen models
        trainable_token_indices={"embed_tokens": special_tokens_ids} if special_tokens_ids is not None else None
    )

    peft_model = get_peft_model(model, config)

    return cast(PeftModel, peft_model)
