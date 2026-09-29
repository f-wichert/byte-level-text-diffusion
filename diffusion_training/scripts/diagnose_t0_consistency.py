import argparse
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion_model import build_diffusion_backbone
from src.diffusion_train import path_coefficients, velocity_to_x0
from src.train import _dtype_from_name
from src.utils import load_checkpoint, load_diffusion_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--t-values", default="0.02,0.05,0.10,0.20")
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--no-self-cond", action="store_true", help="Build without self-cond (match combined/anchor saves).")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.float()
    return float((values * m).sum() / m.sum().clamp_min(1.0))


def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)

    diffusion_cfg = config.diffusion
    if args.no_self_cond:
        diffusion_cfg = dataclasses.replace(diffusion_cfg, self_conditioning=False)
    model = build_diffusion_backbone(diffusion_cfg).to(device=device, dtype=dtype)
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    overfit = checkpoint.get("extra", {}).get("overfit")
    if overfit is None:
        raise SystemExit("Checkpoint has no embedded 'overfit' target.")
    z = overfit["z_data"].to(device=device, dtype=dtype)  # [B, L, D]
    mask = overfit["mask"].to(device=device, dtype=torch.bool)
    path = getattr(model.config, "path", "linear")
    prediction_mode = getattr(model.config, "prediction", "velocity")

    print(f"checkpoint={args.checkpoint} step={checkpoint.get('step')} path={path} pred={prediction_mode} draws={args.draws}")
    print("mode (a) input swings -> cos_of_mean >> mean_cos & low mag_ratio (gate helps, ceiling=cos_of_mean)")
    print("mode (b) wrong constant -> cos_of_mean ~= mean_cos & mag_ratio ~1 (gate won't help)\n")
    print(f"{'t':>6} {'mean_cos':>9} {'cos_of_mean':>12} {'draw_align':>11} {'mag_ratio':>10}")
    print("-" * 52)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    for t_value in [float(v) for v in args.t_values.split(",") if v.strip()]:
        alpha, sigma, _, _ = path_coefficients(t_value, path)
        t = torch.full((z.size(0),), t_value, device=device)
        x0s = []
        with torch.no_grad():
            for _ in range(args.draws):
                eps = torch.randn(z.shape, generator=gen, device=device, dtype=dtype)
                z_t = (alpha * z + sigma * eps).to(dtype)
                raw = model(z_t, t, mask)
                x0 = raw if prediction_mode == "x0" else velocity_to_x0(z_t, raw, t, path)
                x0s.append(x0.float())
        stack = torch.stack(x0s, dim=0)            # [K, B, L, D]
        per_draw_cos = torch.stack([torch.cosine_similarity(x, z.float(), dim=-1) for x in x0s], dim=0)
        mean_cos = float(per_draw_cos.mean(dim=0).mul(mask.float()).sum() / mask.float().sum().clamp_min(1.0))
        x0_mean = stack.mean(dim=0)                # [B, L, D] input-independent component
        cos_of_mean = masked_mean(torch.cosine_similarity(x0_mean, z.float(), dim=-1), mask)
        draw_align = masked_mean(
            torch.stack([torch.cosine_similarity(x, x0_mean, dim=-1) for x in x0s], dim=0).mean(dim=0), mask
        )
        mag_ratio = masked_mean(x0_mean.norm(dim=-1) / stack.norm(dim=-1).mean(dim=0).clamp_min(1e-6), mask)
        print(f"{t_value:>6.2f} {mean_cos:>9.3f} {cos_of_mean:>12.3f} {draw_align:>11.3f} {mag_ratio:>10.3f}")


if __name__ == "__main__":
    main()
