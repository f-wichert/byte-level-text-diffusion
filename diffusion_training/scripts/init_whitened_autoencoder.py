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
    parser.add_argument("config", help="Diffusion YAML whose autoencoder is the funnel-less source AE.")
    parser.add_argument("--whitening", required=True, help="kind='whitening' artifact fit on the source's raw latents.")
    parser.add_argument("--output", required=True, help="Step-0 checkpoint path to write.")
    parser.add_argument("--gate-docs", type=int, default=16, help="Held-out documents for the CE gate.")
    parser.add_argument("--tolerance", type=float, default=0.003,
                        help="Max allowed |CE_new - CE_old|; expect ~ the truncation cost (gate 2 of the fit).")
    parser.add_argument("--device")
    return parser.parse_args()


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
    if config.model.latent_compression_dim is not None:
        raise ValueError(
            "Source config has a funnel (latent_compression_dim="
            f"{config.model.latent_compression_dim}); this script creates one and would stack "
            "a second on top. Use init_rung3_autoencoder.py to fold into an existing funnel."
        )
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)

    artifact = torch.load(args.whitening, map_location="cpu")
    if artifact.get("kind") != "whitening":
        raise ValueError(f"{args.whitening} is not a whitening artifact (kind={artifact.get('kind')!r}).")
    basis = artifact["basis"].double()          # [raw_dim, k]
    scale = artifact["scale"].double()          # [k]
    mean = artifact["mean"].double()            # [raw_dim]
    k = int(artifact["k"])
    raw_dim = int(artifact["raw_dim"])

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

    source = load_frozen_autoencoder(config, device)
    ce_old = teacher_forced_ce(source, batches, device)
    source_state = {key: value.detach().cpu() for key, value in source.state_dict().items()}
    del source
    torch.cuda.empty_cache()

    if any(key.startswith("encoder_compression.") for key in source_state):
        raise ValueError("Source checkpoint already contains funnel weights; refusing to stack.")
    source_latent_dim = config.diffusion.latent_dim
    if raw_dim != source_latent_dim:
        raise ValueError(f"Artifact raw_dim {raw_dim} != source latent dim {source_latent_dim}.")

    new_model_config = dataclasses.replace(
        config.model,
        latent_compression_dim=k,
        latent_compression_layers=1,
        latent_compression_hidden_dims=None,
    )
    print(f"creating single-Linear funnel pair {raw_dim} <-> {k} from {args.whitening}")

    new_state = dict(source_state)
    white = (basis / scale).T                                        # [k, raw_dim] = diag(1/s) B^T
    new_state["encoder_compression.0.weight"] = white.to(dtype)
    new_state["encoder_compression.0.bias"] = (-(white @ mean)).to(dtype)
    new_state["decoder_decompression.0.weight"] = (basis * scale).to(dtype)   # [raw_dim, k] = B diag(s)
    new_state["decoder_decompression.0.bias"] = mean.to(dtype)

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
    print("\nFollow-up model config block (must match this checkpoint exactly):")
    print(f"  latent_compression_dim: {k}")
    print("  latent_compression_layers: 1")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
