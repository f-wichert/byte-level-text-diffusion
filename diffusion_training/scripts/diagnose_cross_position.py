import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sample_diffusion import sample_latents
from src.diffusion_model import build_diffusion_backbone
from src.diffusion_train import encode_latent_batch, load_frozen_autoencoder, load_latent_stats
from src.train import _dtype_from_name, build_training_loader, cycle_batches, move_batch_to_device
from src.utils import load_checkpoint, load_diffusion_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--num-real-batches", type=int, default=128)
    parser.add_argument("--num-gen-sequences", type=int, default=256)
    parser.add_argument("--gen-words", type=int, default=64)
    parser.add_argument("--pca-k", type=int, default=32)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def collect_real_sequences(autoencoder, config, device, dtype, latent_stats, num_batches):
    """Return a list of [W_s, D] standardized real word-latent sequences."""
    loader, _ = build_training_loader(config)
    batches = cycle_batches(loader)
    seqs: list[torch.Tensor] = []
    for _ in range(num_batches):
        batch = move_batch_to_device(next(batches), device)
        latent_batch = encode_latent_batch(
            autoencoder, batch,
            max_words=config.diffusion.max_words, latent_dim=config.diffusion.latent_dim,
            dtype=dtype, device=device, latent_stats=latent_stats,
        )
        z = latent_batch["z"].float()
        mask = latent_batch["mask"]
        for i in range(z.size(0)):
            length = int(mask[i].sum().item())
            if length >= 2:
                seqs.append(z[i, :length].cpu())
    return seqs


@torch.no_grad()
def sample_generated_sequences(model, config, device, dtype, num_seqs, num_words, steps, chunk=32):
    """Return a list of [num_words, D] standardized generated word-latent sequences."""
    seqs: list[torch.Tensor] = []
    remaining = num_seqs
    while remaining > 0:
        n = min(chunk, remaining)
        z = sample_latents(
            model, num_samples=n, num_words=num_words,
            latent_dim=config.diffusion.latent_dim, steps=steps, device=device, dtype=dtype,
        )
        for i in range(n):
            seqs.append(z[i].float().cpu())
        remaining -= n
    return seqs


def fit_pca(real_seqs, k, device):
    """Top-k PCA directions of pooled real word-latents; features scaled to ~unit var."""
    pooled = torch.cat(real_seqs, dim=0).to(device)
    mean = pooled.mean(dim=0)
    centered = pooled - mean
    _, _, v = torch.pca_lowrank(centered, q=min(k + 8, centered.size(1) - 1), niter=6)
    components = v[:, :k]
    fstd = (centered @ components).std(dim=0).clamp_min(1e-6)
    return mean.cpu(), components.cpu(), fstd.cpu()


def to_features(seqs, mean, components, fstd):
    return [(((s - mean) @ components) / fstd) for s in seqs]


def _pearson_per_col(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    den = a.norm(dim=0) * b.norm(dim=0)
    return (a * b).sum(dim=0) / den.clamp_min(1e-8)


def lagged_feature_corr(feat_seqs, lag, generator):
    """Mean over features of |Pearson corr(f[pos], f[pos+lag])|, plus a shuffled null."""
    xs, ys = [], []
    for f in feat_seqs:
        if f.size(0) > lag:
            xs.append(f[:-lag])
            ys.append(f[lag:])
    if not xs:
        return None
    x = torch.cat(xs, 0)
    y = torch.cat(ys, 0)
    r = _pearson_per_col(x, y).abs().mean().item()
    perm = torch.randperm(y.size(0), generator=generator)
    r_null = _pearson_per_col(x, y[perm]).abs().mean().item()
    return r, r_null, x.size(0)


def next_word_r2(feat_seqs, generator, ridge=1.0):
    """Test-split R^2 of a linear model predicting f[pos+1] from f[pos] (multivariate)."""
    xs, ys = [], []
    for f in feat_seqs:
        if f.size(0) > 1:
            xs.append(f[:-1])
            ys.append(f[1:])
    x = torch.cat(xs, 0)
    y = torch.cat(ys, 0)
    n = x.size(0)
    idx = torch.randperm(n, generator=generator)
    split = n // 2
    tr, te = idx[:split], idx[split:]
    x_tr, y_tr, x_te, y_te = x[tr], y[tr], x[te], y[te]
    xm, ym = x_tr.mean(0), y_tr.mean(0)
    xc, yc = x_tr - xm, y_tr - ym
    k = xc.size(1)
    weight = torch.linalg.solve(xc.t() @ xc + ridge * torch.eye(k), xc.t() @ yc)
    pred = (x_te - xm) @ weight + ym
    ss_res = (y_te - pred).pow(2).sum()
    ss_tot = (y_te - y_te.mean(0)).pow(2).sum()
    return (1.0 - ss_res / ss_tot).item(), n


def norm_autocorr(seqs, lag=1):
    a, b = [], []
    for s in seqs:
        nrm = s.norm(dim=-1)
        if nrm.size(0) > lag:
            a.append(nrm[:-lag])
            b.append(nrm[lag:])
    a = torch.cat(a)
    b = torch.cat(b)
    a = a - a.mean()
    b = b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-8)).item()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    config = load_diffusion_config(args.config)
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)

    autoencoder = load_frozen_autoencoder(config, device)
    model = build_diffusion_backbone(config.diffusion).to(device=device, dtype=dtype)
    checkpoint = load_checkpoint(args.diffusion_checkpoint, model=model, map_location=device)
    model.eval()

    latent_stats = checkpoint.get("extra", {}).get("latent_stats")
    if latent_stats is None and config.autoencoder.latent_stats_path:
        latent_stats = load_latent_stats(config.autoencoder.latent_stats_path, device, config.diffusion.latent_dim)
    if latent_stats is None:
        raise SystemExit("This diagnostic assumes a standardized checkpoint (needs latent_stats).")
    latent_stats = {
        "mean": latent_stats["mean"].to(device=device, dtype=torch.float32),
        "std": latent_stats["std"].to(device=device, dtype=torch.float32),
    }
    steps = args.sampling_steps or config.diffusion.sampling_steps

    print(f"checkpoint={args.diffusion_checkpoint} step={checkpoint.get('step')}")
    print(f"collecting real sequences ({args.num_real_batches} batches)...")
    real_seqs = collect_real_sequences(autoencoder, config, device, dtype, latent_stats, args.num_real_batches)
    print(f"collecting generated sequences ({args.num_gen_sequences} x {args.gen_words} words, {steps} steps)...")
    gen_seqs = sample_generated_sequences(
        model, config, device, dtype, args.num_gen_sequences, args.gen_words, steps
    )

    real_words = sum(s.size(0) for s in real_seqs)
    gen_words = sum(s.size(0) for s in gen_seqs)
    print(f"\nreal: {len(real_seqs)} sequences, {real_words} words "
          f"(mean len {real_words / max(1, len(real_seqs)):.1f})")
    print(f"gen : {len(gen_seqs)} sequences, {gen_words} words "
          f"(len {args.gen_words})")

    mean, components, fstd = fit_pca(real_seqs, args.pca_k, device)
    real_feats = to_features(real_seqs, mean, components, fstd)
    gen_feats = to_features(gen_seqs, mean, components, fstd)

    print(f"\n=== Lagged PCA-feature autocorrelation (mean over {args.pca_k} features of |Pearson r|) ===")
    print(f"{'lag':>4} | {'REAL':>8} {'(null)':>8} | {'GEN':>8} {'(null)':>8}")
    for lag in args.lags:
        r_real = lagged_feature_corr(real_feats, lag, generator)
        r_gen = lagged_feature_corr(gen_feats, lag, generator)
        if r_real is None or r_gen is None:
            continue
        print(f"{lag:>4} | {r_real[0]:>8.4f} {r_real[1]:>8.4f} | {r_gen[0]:>8.4f} {r_gen[1]:>8.4f}")

    print("\n=== Next-word linear predictability (test-split R^2: current word -> next word features) ===")
    r2_real, n_real = next_word_r2(real_feats, generator)
    r2_gen, n_gen = next_word_r2(gen_feats, generator)
    print(f"  REAL: R^2 = {r2_real:>7.4f}  (n_pairs={n_real})")
    print(f"  GEN : R^2 = {r2_gen:>7.4f}  (n_pairs={n_gen})")

    print("\n=== Latent-norm lag-1 autocorrelation (assumption-free scalar) ===")
    print(f"  REAL: {norm_autocorr(real_seqs, 1):>7.4f}")
    print(f"  GEN : {norm_autocorr(gen_seqs, 1):>7.4f}")


if __name__ == "__main__":
    main()
