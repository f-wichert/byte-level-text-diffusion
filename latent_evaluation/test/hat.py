import torch
import transformers
import itertools
from transformers import AutoModelForCausalLM

MODEL_ID = "Aleph-Alpha/tfree-hat-pretrained-7b-base"
MODEL_REVISION = "c4797ee9ad934afd5adb27b490d0a3b1d84e4895"
INCLUDE_BACKBONE_LATENTS = False
DEVICE = "cuda:0"


def _identity_decorator(fn):
    return fn


# The model's remote code imports `dynamic_rope_update` from the transformers
# top-level package, but transformers 4.46.3 doesn't export it. For this model's
# config, rope_type is always "default", so a no-op decorator is sufficient.
if not hasattr(transformers, "dynamic_rope_update"):
    transformers.dynamic_rope_update = _identity_decorator

if not torch.cuda.is_available():
    raise RuntimeError("tfree-hat requires a CUDA GPU for FlashAttention-based encoder analysis.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=True,   # required — uses custom HATForCausalLM class
    dtype=torch.bfloat16,
    device_map={"": DEVICE},
)

# Keep the analysis path on one explicit device. Some remote-code submodules can
# still end up on CPU after loading, so we move the parts we actually call.
model.encoder.to(DEVICE)
model.encoder_connector.to(DEVICE)
if INCLUDE_BACKBONE_LATENTS:
    model.backbone.to(DEVICE)

# Inspect the top-level architecture
# print(model)

# The three components should be accessible as submodules
encoder  = model.encoder   # small character-level transformer
backbone = model.backbone  # large word-level transformer  
decoder  = model.decoder   # small character-level transformer



def encode_hat_text(model, text: str, include_backbone: bool = False):
    model.eval()

    words = model.splitter.encode(text)  # list[list[int]], one byte-list per word/chunk
    word_texts = [model.splitter.decode(w) for w in words]

    flat_ids = list(itertools.chain.from_iterable(words))
    cum_lens = [0]
    for w in words:
        cum_lens.append(cum_lens[-1] + len(w))

    encoder_device = model.encoder.embedding_layer.weight.device
    input_ids = torch.tensor([flat_ids], dtype=torch.long, device=encoder_device)
    byte_position_ids = torch.arange(len(flat_ids), dtype=torch.int32, device=encoder_device).unsqueeze(0)
    word_position_ids = torch.arange(len(words), dtype=torch.int32, device=encoder_device).unsqueeze(0)
    cumulative_seq_lengths_per_word = torch.tensor(cum_lens, dtype=torch.int32, device=encoder_device)

    with torch.no_grad():
        encoder_out = model.encoder(
            input_ids=input_ids,
            cumulative_seq_lengths_per_word=cumulative_seq_lengths_per_word,
            byte_position_ids=byte_position_ids,
            word_position_ids=word_position_ids,
        )
        byte_latents = encoder_out.hidden_states  # [1, num_bytes, encoder_hidden]

        connector_device = model.encoder_connector.latent_query.device
        word_latents = model.encoder_connector(
            byte_latents.to(connector_device),
            cumulative_seq_lengths_per_word=cumulative_seq_lengths_per_word.to(connector_device),
            word_position_ids=word_position_ids.to(connector_device),
            byte_position_ids=byte_position_ids.to(connector_device),
        )  # [1, num_words, backbone_hidden]

        backbone_latents = None
        if include_backbone:
            backbone_device = next(model.backbone.parameters()).device
            backbone_latents = model.backbone(
                hidden_states=word_latents.to(backbone_device),
                position_ids=word_position_ids.to(backbone_device),
            ).hidden_states  # [1, num_words, backbone_hidden]

    return {
        "words": word_texts,
        "byte_ids": flat_ids,
        "byte_latents": byte_latents.squeeze(0).cpu(),
        "word_latents": word_latents.squeeze(0).cpu(),
        "backbone_latents": None if backbone_latents is None else backbone_latents.squeeze(0).cpu(),
    }


out = encode_hat_text(model, "The latent space of tfree-hat is interesting.", include_backbone=INCLUDE_BACKBONE_LATENTS)
print(out["words"])
print(out["byte_latents"].shape)
print(out["word_latents"].shape)
print(None if out["backbone_latents"] is None else out["backbone_latents"].shape)
