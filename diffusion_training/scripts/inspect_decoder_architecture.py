import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import LatentHATDecoder, build_decoder_config
from src.train import count_trainable_parameters
from src.utils import load_phase1_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to phase1 YAML config.")
    return parser.parse_args()


def main() -> None:
    config = load_phase1_config(parse_args().config)
    upstream = AutoConfig.from_pretrained(config.model.pretrained_model_id, trust_remote_code=True)
    decoder_config = build_decoder_config(upstream, config.model)
    decoder = LatentHATDecoder(decoder_config)
    decoder_norm = nn.RMSNorm(decoder_config.hidden_size, eps=decoder_config.rms_norm_eps)
    lm_head = nn.Linear(decoder_config.hidden_size, decoder_config.vocab_size, bias=False)
    model = nn.ModuleDict({"decoder": decoder, "decoder_norm": decoder_norm, "lm_head": lm_head})

    cross_config = decoder_config.cross_attention_config
    first_cross = decoder.decoder_layers[0].cross_attention
    print(f"cross_attention_heads={cross_config.num_attention_heads}")
    print(f"cross_attention_kv_heads={cross_config.attention_num_kv_heads}")
    print(f"cross_attention_hidden_size={cross_config.hidden_size}")
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        weight = getattr(first_cross, name).weight
        print(f"{name}.weight={tuple(weight.shape)}")
    print(f"fresh_decoder_trainable_parameters={count_trainable_parameters(model)}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
