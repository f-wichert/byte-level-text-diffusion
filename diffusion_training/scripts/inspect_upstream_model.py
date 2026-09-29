import contextlib
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


MODEL_ID = "Aleph-Alpha/tfree-hat-pretrained-7b-base"


def inspect() -> None:
    model = AutoModelForCausalLM.from_pretrained(
        trust_remote_code=True,
        pretrained_model_name_or_path=MODEL_ID,
    ).to("cuda", torch.bfloat16)

    print(model)
    print("\nKey attributes:")
    for name in ("encoder", "encoder_connector", "backbone", "decoder_connector", "decoder", "layer_norm", "lm_head"):
        module = getattr(model, name, None)
        print(f"{name}: {type(module).__name__ if module is not None else None}")

    print("\nDecoder config:")
    decoder_config = model.config.decoder_config
    cross_config = decoder_config.cross_attention_config
    print(f"decoder_config.num_hidden_layers={decoder_config.num_hidden_layers}")
    print(f"decoder_config.num_attention_heads={decoder_config.num_attention_heads}")
    print(f"decoder_config.num_key_value_heads={decoder_config.num_key_value_heads}")
    print(f"decoder_config.cross_attn_every_layer={decoder_config.cross_attn_every_layer}")
    print(f"cross_attention_config.hidden_size={cross_config.hidden_size}")
    print(f"cross_attention_config.hidden_size_q={cross_config.hidden_size_q}")
    print(f"cross_attention_config.hidden_size_kv={cross_config.hidden_size_kv}")
    print(f"cross_attention_config.num_attention_heads={cross_config.num_attention_heads}")
    print(f"cross_attention_config.attention_num_kv_heads={cross_config.attention_num_kv_heads}")


def main() -> None:
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle):
                inspect()
        print(f"Wrote model inspection to {output_path}")
    else:
        inspect()


if __name__ == "__main__":
    main()
