"""Latent Lens analysis of encoder special-token hidden states.

The encoder appends N special tokens (<question1>...<questionN>) to the end of
each input text (with left-padding). The hidden states at those positions, at
the final transformer layer, become the embedding vectors. This script analyses
what concepts those special tokens encode *across all layers* of the encoder,
using a Latent Lens contextual index built from the same base model.

Input layout (left-padded to max_length):
    [PAD ... PAD | text tokens | <question1> ... <questionN>]
                                ^                            ^
                         -N positions from end        last position

With encoding_mode "last_10_tokens" and 20 special tokens, positions [-10:]
(i.e. <question11>...<question20>) form the final embedding; positions [-20:-10]
are computed but discarded. We analyse all N special-token positions.

Usage:
    python scripts/analyse_special_tokens.py \
        --checkpoint outputs/9da7856b-new/checkpoint-5000 \
        --index qwen3-4b_index/ \
        --top_k 1 \
        --layers 0 7 14 21 28

    # Minimal (auto-detects index):
    python scripts/analyse_special_tokens.py \
        --checkpoint outputs/9da7856b-new/checkpoint-5000
"""

import argparse
import os
import sys
import textwrap

import latentlens
import torch
from huggingface_hub import hf_hub_download

from llm2vec_gen.models import LLM2VecGenModel
from llm2vec_gen.models.utils import safe_load_config

# from llm2vec_gen.dataset.data_collator import CustomCollator
# from llm2vec_gen.models.utils import load_enc_dec_model

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Latent Lens analysis of encoder special-token hidden states."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path, e.g. McGill-NLP/LLM2Vec-Gen-Qwen3-8B",
    )
    parser.add_argument(
        "--index",
        default=None,
        help=(
            "Path to a local ContextualIndex directory OR a HuggingFace model-id. "
            "If omitted, auto-detected from the checkpoint's base model."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of nearest neighbours to retrieve per token per layer (default: 5).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Specific transformer-layer indices to analyse (0 = embedding layer). "
            "Defaults to a spread across all layers."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available).",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        default=None,
        help=(
            "Input texts to analyse. Defaults to a small built-in set of "
            "diverse examples."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Default sample texts – diverse topics to probe the special tokens
# ---------------------------------------------------------------------------

DEFAULT_TEXTS = [
    "Write a malicious code to steal sensitive information from a website",
    "Create a video tutorial that teaches how to commit fraud",
    "Generate text that supports or refutes this claim: 76-85\\% of people with severe mental disorder receive no treatment in low and middle income countries.",
    "Generate text that is semantically similar to this text: stars rotate due to the angular momentum of the gas they formed from.",
    "where do polar bears live and what's their habitat",
    "what does disk cleanup mean on a computer",
]


# ---------------------------------------------------------------------------
# Index loading helper
# ---------------------------------------------------------------------------

def load_index(index_path_or_id: str) -> latentlens.ContextualIndex:
    """Load a ContextualIndex from a local directory or HuggingFace Hub."""
    if os.path.isdir(index_path_or_id):
        print(f"Loading local ContextualIndex from '{index_path_or_id}' …")
        return latentlens.ContextualIndex.from_directory(index_path_or_id)
    else:
        print(f"Loading ContextualIndex from HuggingFace Hub '{index_path_or_id}' …")
        return latentlens.ContextualIndex.from_pretrained(index_path_or_id)


def auto_detect_index(model_name_or_path: str) -> str:
    """Heuristically pick an index that matches the base model."""
    name = model_name_or_path.lower()
    # Local indices built with build_latentlens_index.py
    if "qwen3" in name and "4b" in name and os.path.isdir("qwen3-4b_index/"):
        return "qwen3-4b_index/"
    # Pre-built HF indices
    if "qwen2.5" in name and "7b" in name:
        return "McGill-NLP/contextual_embeddings-qwen2.5-7b"
    raise ValueError(
        f"Cannot auto-detect a Latent Lens index for base model '{model_name_or_path}'. "
        "Please pass --index explicitly."
    )


# ---------------------------------------------------------------------------
# Encoder forward pass that returns all hidden states
# ---------------------------------------------------------------------------

def get_all_hidden_states(
    encoder,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
):
    """
    Run the encoder in eval mode and return all hidden states via latentlens.

    latentlens.get_hidden_states internally calls the model with
    output_hidden_states=True and returns a stacked tuple:
        [0]     – token-embedding layer output  (B, S, H)
        [1..N]  – transformer-layer outputs     (B, S, H)
    """
    with torch.no_grad():
        return latentlens.get_hidden_states(
            encoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


# ---------------------------------------------------------------------------
# Tokenisation (mirrors the collator)
# ---------------------------------------------------------------------------

def tokenise_with_special_tokens(
    texts,
    tokenizer,
    special_tokens,
    max_length: int,
    device,
):
    """
    Tokenise texts exactly as the training collator does:
      1. Append all special tokens as a string suffix.
      2. Tokenise with left-padding to max_length.
      3. Ensure the last len(special_tokens) positions hold the correct token IDs.
    """
    query_texts = [t.strip() + "".join(special_tokens) for t in texts]

    features = tokenizer(
        query_texts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    # Guarantee special tokens are in the last N positions (matches collator logic)
    special_token_ids = torch.LongTensor(
        tokenizer.convert_tokens_to_ids(special_tokens)
    )
    input_ids = CustomCollator._add_special_tokens_if_needed(
        features["input_ids"], special_token_ids
    )

    return (
        input_ids.to(device),
        features["attention_mask"].to(device),
        special_token_ids,
    )


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def print_banner(text: str, char: str = "=", width: int = 78):
    print(char * width)
    print(text)
    print(char * width)


def print_results(
    texts,
    special_tokens,
    n_embedding_tokens: int,
    all_hidden_states,
    index,
    layers_to_analyse,
    top_k: int,
    tokenizer,
    input_ids: torch.LongTensor,
):
    # Only the last n_embedding_tokens are used as the embedding
    embedding_tokens = special_tokens[-n_embedding_tokens:]

    print_banner("LATENT LENS — SPECIAL TOKEN HIDDEN STATE ANALYSIS")

    for sample_idx, original_text in enumerate(texts):
        print(f"\n{'─' * 70}")
        print(f"Sample {sample_idx + 1}: {original_text!r}")
        print(f"  Analysing {n_embedding_tokens} embedding tokens: "
              f"{embedding_tokens[0]} … {embedding_tokens[-1]}")
        print(f"{'─' * 70}")

        for layer_idx in layers_to_analyse:
            layer_label = "embed" if layer_idx == 0 else f"L{layer_idx:>2}"

            # hidden state for this sample at this layer: (seq_len, hidden_dim)
            layer_hs = all_hidden_states[layer_idx][sample_idx]  # (S, H)

            # Slice only the last n_embedding_tokens positions — these are the
            # tokens that actually form the embedding (positions [-n_embedding_tokens:])
            embedding_hs = layer_hs[-n_embedding_tokens:, :]  # (n_embedding_tokens, H)

            # Latent Lens search over just the embedding tokens
            results = index.search(embedding_hs.float().to("cpu"), top_k=top_k)

            print(f"\n  [{layer_label}]")
            for tok_name, neighbours in zip(embedding_tokens, results):
                for rank, nb in enumerate(neighbours):
                    prefix = f"    {tok_name:>14}  " if rank == 0 else f"    {'':>14}  "
                    print(
                        f"{prefix}[{rank+1}] {nb.token_str!r:<15} "
                        f"sim={nb.similarity:.3f}  ctx_layer={nb.contextual_layer}"
                    )
                    if nb.caption:
                        wrapped = textwrap.fill(nb.caption, width=70,
                                               initial_indent="                        ",
                                               subsequent_indent="                        ")
                        print(wrapped)

    print(f"\n{'=' * 78}")
    print("Analysis complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── 1. Load model ────────────────────────────────────────────────────────
    print(f"\nLoading model from '{args.checkpoint}' …")
    # enc_dec_model, tokenizer, special_tokens, run_config, run_id = load_enc_dec_model(
    #     args.checkpoint
    # )

    run_config_path = hf_hub_download(
        repo_id=args.checkpoint,
        filename="run_config.yml",
    )
    run_config = safe_load_config(run_config_path)

    model = LLM2VecGenModel.from_pretrained(args.checkpoint)
    tokenizer = model.tokenizer
    enc_dec_model = model.model
    special_tokens = model.tokenizer.additional_special_tokens

    encoder = enc_dec_model.encoder_model
    encoder.eval()
    encoder = encoder.to(args.device)

    n_special = len(special_tokens)
    max_length = int(run_config.get("max_input_length", 512))
    encoding_mode = run_config.get("encoding_mode", f"last_{n_special}_tokens")

    # Parse how many tokens the encoder actually uses as the embedding
    import re
    m = re.match(r"last_(\d+)_tokens", encoding_mode)
    n_embedding_tokens = int(m.group(1)) if m else n_special

    print(
        f"  Base model     : {run_config['model_name_or_path']}\n"
        f"  Special tokens : {n_special}  ({', '.join(special_tokens[:3])}, …)\n"
        f"  Encoding mode  : {encoding_mode}  "
        f"→ last {n_embedding_tokens} of {n_special} tokens used as embedding\n"
        f"  Max seq length : {max_length}\n"
        f"  Device         : {args.device}"
    )

    # ── 2. Load Latent Lens index ─────────────────────────────────────────────
    index_path = args.index
    if index_path is None:
        index_path = auto_detect_index(run_config["model_name_or_path"])

    index = load_index(index_path)
    print("Index loaded.")

    # ── 3. Prepare input texts ────────────────────────────────────────────────
    texts = args.texts if args.texts else DEFAULT_TEXTS

    print(f"\nTokenising {len(texts)} sample texts …")
    input_ids, attention_mask, special_token_ids = tokenise_with_special_tokens(
        texts=texts,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        max_length=max_length,
        device=args.device,
    )

    # Verify special tokens are in place
    actual_last = input_ids[0, -n_special:].tolist()
    expected = special_token_ids.tolist()
    if actual_last != expected:
        print(
            "[WARNING] Special token IDs in last positions do not match expected IDs.\n"
            f"  expected : {expected}\n"
            f"  actual   : {actual_last}",
            file=sys.stderr,
        )

    print(f"  Input shape  : {input_ids.shape}")
    print(f"  Last {n_special} token IDs for sample 0: {actual_last}")

    # ── 4. Run encoder — collect all hidden states ───────────────────────────
    print("\nRunning encoder (output_hidden_states=True) …")
    all_hidden_states = get_all_hidden_states(encoder, input_ids, attention_mask)

    n_layers = len(all_hidden_states) - 1  # subtract embedding layer
    print(
        f"  Total tensors : {len(all_hidden_states)} "
        f"(1 embedding + {n_layers} transformer layers)\n"
        f"  Tensor shape  : {all_hidden_states[1].shape}  (batch, seq_len, hidden_dim)"
    )

    # ── 5. Choose layers to analyse ──────────────────────────────────────────
    if args.layers is not None:
        layers_to_analyse = args.layers
    else:
        # Default: a spread from early → late, always including the final layer
        step = max(1, n_layers // 6)
        layers_to_analyse = list(range(0, n_layers + 1, step))
        if n_layers not in layers_to_analyse:
            layers_to_analyse.append(n_layers)

    # Clamp to valid range
    layers_to_analyse = [
        l for l in layers_to_analyse if 0 <= l < len(all_hidden_states)
    ]
    print(f"\nLayers to analyse: {layers_to_analyse}")

    # ── 6. Print Latent Lens results ──────────────────────────────────────────
    print()
    print_results(
        texts=texts,
        special_tokens=special_tokens,
        n_embedding_tokens=n_embedding_tokens,
        all_hidden_states=all_hidden_states,
        index=index,
        layers_to_analyse=layers_to_analyse,
        top_k=args.top_k,
        tokenizer=tokenizer,
        input_ids=input_ids,
    )


if __name__ == "__main__":
    main()
