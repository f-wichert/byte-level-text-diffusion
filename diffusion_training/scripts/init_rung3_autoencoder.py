import argparse
import dataclasses
import datetime
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion_train import load_frozen_autoencoder
from src.model import DiffusionAutoencoder
from src.train import _dtype_from_name, _loss_sum_and_tokens, build_validation_loaders, move_batch_to_device
from src.utils import load_diffusion_config, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Base diffusion YAML (source AE checkpoint + data mixture).")
    parser.add_argument("--whitening", default="artifacts/latents/latent_whitening-compress256-k32.pt")
    parser.add_argument("--output", default="checkpoints/rung3-init/step_000000.pt")
    parser.add_argument("--gate-docs", type=int, default=16, help="Held-out documents for the CE gate.")
    parser.add_argument("--tolerance", type=float, default=0.003,
                        help="Max allowed |CE_new - CE_old|; the truncation cost alone is ~1e-4.")
    parser.add_argument("--device")
    return parser.parse_args()


def funnel_linear_indices(state: dict[str, torch.Tensor], prefix: str) -> list[int]:
    """Sorted Sequential indices of the Linear layers under ``prefix`` in a state dict."""
    indices = sorted(
        int(key.split(".")[1])
        for key in state
        if key.startswith(prefix + ".") and key.endswith(".weight")
    )
    if not indices:
        raise ValueError(f"No funnel layers found under {prefix!r}; is this a compressed-AE checkpoint?")
    return indices


@torch.no_grad()
def teacher_forced_ce(model: torch.nn.Module, batches: list[dict], device: torch.device) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in batches:
        batch = move_batch_to_device(batch, device)
        output = model(**batch)
        loss_sum, tokens = _loss_sum_and_tokens(output["logits"], batch["labels"])
        total_loss += float(loss_sum.item())
        total_tokens += tokens
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)

    artifact = torch.load(args.whitening, map_location="cpu")
    if artifact.get("kind") != "whitening":
        raise ValueError(f"{args.whitening} is not a whitening artifact (kind={artifact.get('kind')!r}).")
    basis = artifact["basis"].double()          # [raw_dim, k]
    scale = artifact["scale"].double()          # [k]
    mean = artifact["mean"].double()            # [raw_dim]
    k = int(artifact["k"])

    # Gate documents first (CPU-resident), so both models are scored on identical bytes.
    loaders = build_validation_loaders(config)
    batches: list[dict] = []
    for loader in loaders.values():
        for batch in loader:
            batches.append(batch)
            if len(batches) >= args.gate_docs:
                break
        if len(batches) >= args.gate_docs:
            break

    # Source model: the frozen phase-2 AE, loaded exactly as the trainers load it. This is
    # also the backwards-compat gate for the hidden_dims change: a strict checkpoint load
    # through the default (geometric) funnel path.
    source = load_frozen_autoencoder(config, device)
    ce_old = teacher_forced_ce(source, batches, device)
    source_state = {key: value.detach().cpu() for key, value in source.state_dict().items()}
    del source
    torch.cuda.empty_cache()

    enc_indices = funnel_linear_indices(source_state, "encoder_compression")
    dec_indices = funnel_linear_indices(source_state, "decoder_decompression")
    enc_last = f"encoder_compression.{enc_indices[-1]}"
    dec_first = f"decoder_decompression.{dec_indices[0]}"
    # Interior widths = the source funnel's, so every non-boundary layer copies verbatim.
    hidden_dims = [int(source_state[f"encoder_compression.{i}.weight"].shape[0]) for i in enc_indices[:-1]]

    new_model_config = dataclasses.replace(
        config.model,
        latent_compression_dim=k,
        latent_compression_hidden_dims=hidden_dims,
    )
    print(f"source funnel interior {hidden_dims} -> k={k}; composing {enc_last} and {dec_first}")

    new_state = dict(source_state)
    w_enc = source_state[f"{enc_last}.weight"].double()          # [256, 512]
    b_enc = source_state[f"{enc_last}.bias"].double()            # [256]
    white = (basis / scale).T                                    # [k, 256] = diag(1/s) B^T
    new_state[f"{enc_last}.weight"] = (white @ w_enc).to(dtype)
    new_state[f"{enc_last}.bias"] = (white @ (b_enc - mean)).to(dtype)

    w_dec = source_state[f"{dec_first}.weight"].double()         # [512, 256]
    b_dec = source_state[f"{dec_first}.bias"].double()           # [512]
    new_state[f"{dec_first}.weight"] = (w_dec @ (basis * scale)).to(dtype)
    new_state[f"{dec_first}.bias"] = (w_dec @ mean + b_dec).to(dtype)

    new_model = DiffusionAutoencoder.from_pretrained(
        new_model_config,
        torch_dtype=dtype,
        device=device,
        freeze_encoder=True,
        local_files_only=config.autoencoder.local_files_only,
    )
    new_model.load_state_dict({key: value.to(device) for key, value in new_state.items()}, strict=True)
    ce_new = teacher_forced_ce(new_model, batches, device)

    gap = ce_new - ce_old
    print(f"[gate] teacher-forced CE on {len(batches)} held-out docs: "
          f"source {ce_old:.4f}  composed {ce_new:.4f}  gap {gap:+.4f} (tolerance {args.tolerance})")
    if abs(gap) > args.tolerance:
        print("GATE FAIL -- checkpoint NOT written.")
        sys.stdout.flush()
        os._exit(1)

    output_path = Path(args.output)
    save_checkpoint(
        output_path,
        model=new_model,
        step=0,
        extra={
            "model_config": dataclasses.asdict(new_model_config),
            "source_checkpoint": config.autoencoder.checkpoint,
            "whitening_artifact": str(args.whitening),
            "gate": {"ce_source": ce_old, "ce_composed": ce_new, "gap": gap, "docs": len(batches)},
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(f"GATE PASS -- wrote {output_path}")
    print("\nR1 model config block (must match this checkpoint exactly):")
    print(f"  latent_compression_dim: {k}")
    print(f"  latent_compression_layers: {len(enc_indices)}")
    print(f"  latent_compression_hidden_dims: {hidden_dims}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
