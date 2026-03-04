import yaml
import re
from typing import List, cast
from transformers import AutoTokenizer

def custom_constructor(loader, tag_suffix, node):
    # Return the raw string value instead of trying to construct the object
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        # For sequence nodes, return the first value
        values = loader.construct_sequence(node)
        return values[0] if values else None
    return None


def safe_load_config(config_path):
    # Add constructor for all python/object/apply tags
    yaml.add_multi_constructor(
        "tag:yaml.org,2002:python/object/apply:",
        custom_constructor,
        Loader=yaml.SafeLoader,
    )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_enc_dec_model(model_path, causal_lm_decoder: bool = False):
    from llm2vec_gen.models.modeling_encoder_decoder import EncoderDecoderModel

    # assert args.encoder_model_path is of regex type output/{run_id}/checkpoint-*
    assert re.match(r"outputs/.*/checkpoint-*", model_path), (
        "model_path must be of regex type outputs/{run_id}/checkpoint-*"
    )

    # extract run_id from args.encoder_model_path
    match = re.search(r"outputs/([^/]+)/checkpoint-", model_path)
    if not match:
        raise ValueError(f"Could not extract run_id from path: {model_path}")
    run_id = match.group(1)

    # load run config
    run_config = safe_load_config(f"outputs/{run_id}/run_config.yml")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    special_tokens = run_config["special_tokens"]
    special_tokens = cast(List[str], special_tokens)

    enc_dec_model = EncoderDecoderModel.from_pretrained(model_path, causal_lm_decoder=causal_lm_decoder)
    enc_dec_model.eval()
    return enc_dec_model, tokenizer, special_tokens, run_config, run_id