import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_teacher_forced import teacher_forced_sample
from scripts.sample_diffusion import sample_latents
from src.diffusion_model import build_diffusion_backbone
from src.train import _dtype_from_name
from src.utils import load_checkpoint, load_diffusion_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to diffusion YAML config.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Overfit checkpoint from overfit_diffusion.py --save-path (embeds the target latents).",
    )
    parser.add_argument(
        "--t-start",
        default="0.5",
        help="Comma-separated start times for the noised-real regime (near 1 = little noise).",
    )
    parser.add_argument("--sampling-steps", type=int, help="Euler steps (default: config value).")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean of per-position ``values`` [B, W] over valid (mask=True) positions."""
    m = mask.float()
    return float((values * m).sum() / m.sum().clamp_min(1.0))


def compare(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Per-position cosine / L2 / norms between two [B, W, D] latent sets, masked-mean reduced."""
    pred = pred.float()
    target = target.float()
    cos = torch.cosine_similarity(pred, target, dim=-1)          # [B, W]
    l2 = (pred - target).norm(dim=-1)                            # [B, W]
    return {
        "cosine": masked_mean(cos, mask),
        "l2": masked_mean(l2, mask),
        "pred_norm": masked_mean(pred.norm(dim=-1), mask),
        "true_norm": masked_mean(target.norm(dim=-1), mask),
    }


def match_to_nearest(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Reorder ``pred`` [N, W, D] so row i is the prediction nearest (min masked L2) to target i.

    Unconditional samples are not paired to a specific memorized mode, so for N>1 we greedily
    assign each target its closest distinct prediction. Returns the reordered preds and the
    chosen prediction index for each target.
    """
    n = pred.size(0)
    m = mask.float().unsqueeze(-1)
    # dist[t, p] = mean L2 of target t vs prediction p over valid positions of target t.
    dist = torch.empty((n, n))
    for t in range(n):
        diff = (pred.float() - target[t : t + 1].float()) * m[t : t + 1]
        per_pos = diff.norm(dim=-1)                              # [N, W]
        dist[t] = (per_pos * mask[t].float()).sum(dim=-1) / mask[t].float().sum().clamp_min(1.0)
    chosen: list[int] = []
    used: set[int] = set()
    for t in range(n):
        order = torch.argsort(dist[t])
        pick = next(int(p) for p in order if int(p) not in used)
        used.add(pick)
        chosen.append(pick)
    return pred[chosen], chosen


def unstandardize(z: torch.Tensor, latent_stats: dict[str, torch.Tensor] | None) -> torch.Tensor:
    if latent_stats is None:
        return z.float()
    return z.float() * latent_stats["std"] + latent_stats["mean"]


def print_table(title: str, std_metrics: dict[str, float], raw_metrics: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    print(f"  cosine        {std_metrics['cosine']:+.4f}")
    print(f"  L2 distance   {std_metrics['l2']:.4f}")
    print(f"  norm (std)    pred {std_metrics['pred_norm']:.4f}  vs true {std_metrics['true_norm']:.4f}")
    print(f"  norm (raw)    pred {raw_metrics['pred_norm']:.4f}  vs true {raw_metrics['true_norm']:.4f}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = load_diffusion_config(args.config)
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)

    model = build_diffusion_backbone(config.diffusion).to(device=device, dtype=dtype)
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    extra = checkpoint.get("extra", {})
    overfit = extra.get("overfit")
    if overfit is None:
        raise SystemExit(
            "Checkpoint has no 'overfit' targets. Re-run overfit_diffusion.py with --save-path."
        )
    z_data = overfit["z_data"].to(device=device, dtype=dtype)       # [N, L, D] standardized
    mask = overfit["mask"].to(device=device, dtype=torch.bool)      # [N, L]
    num_samples, num_words, latent_dim = z_data.shape

    latent_stats = extra.get("latent_stats")
    if latent_stats is not None:
        latent_stats = {
            "mean": latent_stats["mean"].to(device=device, dtype=torch.float32),
            "std": latent_stats["std"].to(device=device, dtype=torch.float32),
        }

    steps = args.sampling_steps or config.diffusion.sampling_steps
    t_starts = [float(v) for v in args.t_start.split(",") if v.strip()]

    print(f"checkpoint={args.checkpoint} step={checkpoint.get('step')}")
    print(
        f"path={getattr(model.config, 'path', 'linear')} steps={steps} "
        f"standardized={latent_stats is not None} N={num_samples} words={num_words}"
    )
    # Round-trip check on the stored targets (standardized space should sit near elem-std 1).
    true_elem_std = float(z_data.float()[mask].std())
    print(f"target latents: elem std {true_elem_std:.3f} (standardized space target ~1.0)")

    def report(title: str, pred: torch.Tensor) -> None:
        std_m = compare(pred, z_data, mask)
        raw_m = compare(unstandardize(pred, latent_stats), unstandardize(z_data, latent_stats), mask)
        print_table(title, std_m, raw_m)

    # --- Regime 1: from random noise (unconditional) ---
    z_noise = sample_latents(
        model,
        num_samples=num_samples,
        num_words=num_words,
        latent_dim=latent_dim,
        steps=steps,
        device=device,
        dtype=dtype,
    )
    if num_samples > 1:
        z_noise, chosen = match_to_nearest(z_noise, z_data, mask)
        print(f"\n[random-noise samples matched to nearest target: {chosen}]")
    report("RANDOM NOISE  ->  true", z_noise)

    # --- Regime 2: from noised real latents (index-aligned, per t_start) ---
    for t_start in t_starts:
        z_tf = teacher_forced_sample(model, z_data, mask, t_start, steps, device, dtype)
        report(f"NOISED REAL (t_start={t_start:.2f})  ->  true", z_tf)


if __name__ == "__main__":
    main()
