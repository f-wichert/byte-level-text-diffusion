import argparse
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
    parser.add_argument("config", help="Path to diffusion YAML config.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Overfit checkpoint from overfit_diffusion.py --save-path (embeds the target latents).",
    )
    parser.add_argument(
        "--t-values",
        default="0.02,0.05,0.10,0.15,0.20,0.30,0.50,0.70,0.90,0.98",
        help="Comma-separated t grid (small t = high noise).",
    )
    parser.add_argument("--noise-draws", type=int, default=16, help="Noise samples averaged per t.")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.float()
    return float((values * m).sum() / m.sum().clamp_min(1.0))


@torch.no_grad()
def x0_error_at_t(
    model,
    z_data: torch.Tensor,
    mask: torch.Tensor,
    t_value: float,
    noise_draws: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Average single-forward x_hat0 error at one t over ``noise_draws`` noise samples.

    Returns cosine and relative-L2 of x_hat0 vs z*, for both the first pass (z_self=zeros, what the
    first sampling step sees) and the self-conditioned second pass (z_self = pass-1 x_hat0); plus
    ``cos_in``, the input z_t's own cosine to z* (the no-op reference the model must beat).
    """
    path = getattr(model.config, "path", "linear")
    self_cond = getattr(model.config, "self_conditioning", False)
    prediction_mode = getattr(model.config, "prediction", "velocity")
    alpha, sigma, _, _ = path_coefficients(t_value, path)

    def to_x0(z_t: torch.Tensor, t: torch.Tensor, z_self: torch.Tensor | None) -> torch.Tensor:
        raw = model(z_t, t, mask, z_self)
        return raw if prediction_mode == "x0" else velocity_to_x0(z_t, raw, t, path)

    z_norm = z_data.float().norm(dim=-1)  # [B, W] for relative L2
    acc = {k: 0.0 for k in ("cos_p1", "rel_l2_p1", "cos_p2", "rel_l2_p2", "cos_in")}
    for _ in range(noise_draws):
        eps = torch.randn(z_data.shape, generator=generator, device=device, dtype=dtype)
        z_t = (alpha * z_data + sigma * eps).to(dtype)
        t = torch.full((z_data.size(0),), t_value, device=device)

        x0_p1 = to_x0(z_t, t, None)
        acc["cos_p1"] += masked_mean(torch.cosine_similarity(x0_p1.float(), z_data.float(), dim=-1), mask)
        acc["rel_l2_p1"] += masked_mean((x0_p1.float() - z_data.float()).norm(dim=-1) / z_norm.clamp_min(1e-6), mask)
        acc["cos_in"] += masked_mean(torch.cosine_similarity(z_t.float(), z_data.float(), dim=-1), mask)

        if self_cond:
            x0_p2 = to_x0(z_t, t, x0_p1.to(dtype))
        else:
            x0_p2 = x0_p1
        acc["cos_p2"] += masked_mean(torch.cosine_similarity(x0_p2.float(), z_data.float(), dim=-1), mask)
        acc["rel_l2_p2"] += masked_mean((x0_p2.float() - z_data.float()).norm(dim=-1) / z_norm.clamp_min(1e-6), mask)

    return {k: v / noise_draws for k, v in acc.items()}


def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)

    model = build_diffusion_backbone(config.diffusion).to(device=device, dtype=dtype)
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    overfit = checkpoint.get("extra", {}).get("overfit")
    if overfit is None:
        raise SystemExit("Checkpoint has no 'overfit' targets. Re-run overfit_diffusion.py with --save-path.")
    z_data = overfit["z_data"].to(device=device, dtype=dtype)  # [N, L, D] standardized
    mask = overfit["mask"].to(device=device, dtype=torch.bool)
    num_samples, num_words, _ = z_data.shape

    self_cond = getattr(model.config, "self_conditioning", False)
    prediction_mode = getattr(model.config, "prediction", "velocity")
    print(f"checkpoint={args.checkpoint} step={checkpoint.get('step')}")
    print(
        f"path={getattr(model.config, 'path', 'linear')} prediction={prediction_mode} "
        f"self_cond={self_cond} N={num_samples} words={num_words} noise_draws={args.noise_draws}"
    )
    print("convention: t=0 pure noise (high noise), t=1 clean. single forward, no integration.\n")

    header = f"{'t':>6} {'cos_in':>8} {'cos x0(p1)':>11} {'relL2(p1)':>10}"
    if self_cond:
        header += f" {'cos x0(p2)':>11} {'relL2(p2)':>10}"
    print(header)
    print("-" * len(header))

    generator = torch.Generator(device=device).manual_seed(args.seed)
    t_values = [float(v) for v in args.t_values.split(",") if v.strip()]
    for t_value in t_values:
        m = x0_error_at_t(model, z_data, mask, t_value, args.noise_draws, generator, device, dtype)
        row = f"{t_value:>6.2f} {m['cos_in']:>8.3f} {m['cos_p1']:>11.3f} {m['rel_l2_p1']:>10.3f}"
        if self_cond:
            row += f" {m['cos_p2']:>11.3f} {m['rel_l2_p2']:>10.3f}"
        print(row)

    print(
        "\nx0 cosine -> 1.0 means perfect clean-latent recovery; relL2 -> 0 perfect, ~1.41 "
        "uncorrelated. A useful model beats cos_in at every t."
    )


if __name__ == "__main__":
    main()
