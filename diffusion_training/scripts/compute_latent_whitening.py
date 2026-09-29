import argparse
import datetime
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion_train import encode_latent_batch, load_frozen_autoencoder
from src.train import build_training_loader, move_batch_to_device
from src.utils import load_diffusion_config, module_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Diffusion YAML config (data mixture + autoencoder checkpoint).")
    parser.add_argument("--output", default="artifacts/latents/latent_whitening.pt")
    parser.add_argument("--k", type=int, default=32, help="Eigendirections kept (the working dim).")
    parser.add_argument("--num-sequences", type=int, default=1024)
    parser.add_argument("--holdout-batches", type=int, default=8, help="Batches for the isotropy gate.")
    parser.add_argument("--decode-docs", type=int, default=16, help="Documents for the decode round-trip gate.")
    parser.add_argument("--skip-positions", type=int, default=8,
                        help="Document-initial word positions excluded from the FIT (no-left-context "
                             "outliers); the decode gate still round-trips every position.")
    parser.add_argument("--device")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    device = torch.device(args.device or config.training.device)
    raw_dim = config.model.latent_compression_dim or config.diffusion.latent_dim

    autoencoder = load_frozen_autoencoder(config, device)
    encode_dtype = module_dtype(autoencoder, torch.bfloat16)
    loader, data_info = build_training_loader(config)

    total = torch.zeros(raw_dim, device=device, dtype=torch.float64)
    outer = torch.zeros(raw_dim, raw_dim, device=device, dtype=torch.float64)
    word_count = 0
    sequence_count = 0
    holdout_z: list[torch.Tensor] = []
    decode_batches: list[dict] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        latent_batch = encode_latent_batch(
            autoencoder,
            batch,
            max_words=config.diffusion.max_words,
            latent_dim=raw_dim,
            dtype=encode_dtype,
            device=device,
            latent_stats=None,
        )
        if sequence_count < args.num_sequences:
            fit_mask = latent_batch["mask"].clone()
            fit_mask[:, : args.skip_positions] = False
            valid = latent_batch["z"][fit_mask].to(torch.float64)
            if valid.numel():
                total += valid.sum(dim=0)
                outer += valid.T @ valid
                word_count += valid.size(0)
            sequence_count += int(latent_batch["mask"].size(0))
            if sequence_count % 128 < latent_batch["mask"].size(0):
                print(f"sequences={sequence_count} words={word_count}", flush=True)
        else:
            mask = latent_batch["mask"].clone()
            mask[:, : args.skip_positions] = False
            holdout_z.append(latent_batch["z"][mask].float())
            if len(decode_batches) * latent_batch["mask"].size(0) < args.decode_docs:
                decode_batches.append({"batch": batch, "latents": latent_batch})
            if len(holdout_z) >= args.holdout_batches:
                break

    mean = (total / max(1, word_count)).float()
    cov = outer / max(1, word_count) - torch.outer(total, total) / max(1, word_count) ** 2
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.flip(0).clamp_min(0)
    eigvecs = eigvecs.flip(1)
    basis = eigvecs[:, : args.k].float()
    scale = eigvals[: args.k].sqrt().float().clamp_min(1e-6)

    kept = float(eigvals[: args.k].sum() / eigvals.sum())
    print(f"\nfit: words={word_count} sequences={sequence_count} raw_dim={raw_dim} k={args.k}")
    print(f"eigen-std: top {scale[0]:.3f} k-th {scale[-1]:.4f} median(all) {eigvals[raw_dim // 2].sqrt():.4f}")
    print(f"variance kept by top-{args.k}: {100 * kept:.2f}%")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "whitening",
            "mean": mean.cpu(),
            "basis": basis.cpu(),
            "scale": scale.cpu(),
            "eigenvalues": eigvals.float().cpu(),
            "k": args.k,
            "raw_dim": raw_dim,
            "num_words": word_count,
            "num_sequences": sequence_count,
            "skip_positions": args.skip_positions,
            "autoencoder_checkpoint": config.autoencoder.checkpoint,
            "config": str(args.config),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        output_path,
    )
    print(f"output={output_path}")

    # Gate 1: the whitened holdout must be isotropic. Judge on the covariance EIGENVALUES,
    # not the per-dim stds -- matching only the diagonal is the failure this replaces.
    if holdout_z:
        held = torch.cat(holdout_z, dim=0)
        w = ((held - mean.to(device)) @ basis.to(device)) / scale.to(device)
        w_cov = (w - w.mean(0)).T @ (w - w.mean(0)) / (w.size(0) - 1)
        w_ev = torch.linalg.eigvalsh(w_cov.double())
        print(f"\n[gate 1] whitened holdout ({w.size(0)} words):")
        print(f"  per-dim std: min {w.std(0).min():.3f} max {w.std(0).max():.3f} (target ~1)")
        print(f"  covariance eigenvalues: min {w_ev.min():.3f} max {w_ev.max():.3f} (target ~1)")
        print(f"  norm mean {w.norm(dim=-1).mean():.2f} (target ~{args.k ** 0.5:.2f})")

    # Gate 2: decode round-trip. Whiten->unwhiten equals projection onto the kept subspace,
    # so the CE gap against the original latents is exactly the truncation cost.
    basis_d = basis.to(device)
    scale_d = scale.to(device)
    mean_d = mean.to(device)
    totals = {"orig": 0.0, "round": 0.0}
    tokens = 0
    docs = 0
    for entry in decode_batches:
        batch, latent_batch = entry["batch"], entry["latents"]
        z, lengths = latent_batch["z"].float(), latent_batch["lengths"]
        w = ((z - mean_d) @ basis_d) / scale_d
        z_round = (w * scale_d) @ basis_d.T + mean_d
        for name, z_use in (("orig", z), ("round", z_round)):
            for index in range(z.size(0)):
                length = int(lengths[index].item())
                if length <= 0:
                    continue
                boundary = batch["word_boundaries"][index].to(device=device, dtype=torch.long)
                labels = batch["labels"][index].clone()
                labels[max(0, int(boundary[length].item()) - 1):] = -100
                logits = autoencoder.decode(
                    batch["byte_ids"][index].unsqueeze(0),
                    [z_use[index, :length].to(encode_dtype).unsqueeze(0)],
                    [boundary[: length + 1].to(torch.int32)],
                )[0]
                lab = labels.to(device)
                totals[name] += float(F.cross_entropy(
                    logits.view(-1, logits.size(-1)), lab.view(-1), ignore_index=-100, reduction="sum"
                ))
                if name == "orig":
                    tokens += int((lab != -100).sum())
                    docs += 1
    if tokens:
        print(f"\n[gate 2] decode round-trip ({docs} held-out documents, {tokens} bytes):")
        print(f"  teacher-forced CE, original latents:      {totals['orig'] / tokens:.4f}")
        print(f"  teacher-forced CE, whiten->unwhiten k={args.k}: {totals['round'] / tokens:.4f}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
