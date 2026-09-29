import argparse
import datetime
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion_train import encode_latent_batch, load_frozen_autoencoder, load_latent_stats
from src.length_predictor import (  # noqa: F401 (LengthPredictor re-exported for back-compat)
    LengthPredictor,
    LengthPredictorV1,
    LengthPredictorV2,
    load_length_predictor,
)
from src.train import _dtype_from_name, build_training_loader, move_batch_to_device
from src.utils import DiffusionExperimentConfig, load_diffusion_config

DEFAULT_V1_BASELINE = "artifacts/latents/length_predictor-compress256-raw.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Diffusion YAML config; reads its length_predictor section.")
    parser.add_argument("--device")
    parser.add_argument("--check-only", action="store_true", help="Validate the artifact; never train.")
    parser.add_argument("--force", action="store_true", help="Retrain and overwrite even if it matches.")
    parser.add_argument(
        "--v1-baseline",
        default=DEFAULT_V1_BASELINE,
        help="v1 (mlp) artifact evaluated on the same holdout for a fair comparison (raw space only).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- data


def sequence_examples(
    autoencoder: torch.nn.Module,
    batch: dict,
    config: DiffusionExperimentConfig,
    dtype: torch.dtype,
    device: torch.device,
    latent_stats: dict[str, torch.Tensor] | None,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (latents [B, W, D] float32, length classes [B, W] with -100 padding, mask [B, W]).

    ``latent_stats`` is passed to ``encode_latent_batch`` (standardized space) or ``None``
    (raw space); the raw path mirrors what generate.py feeds the predictor at decode time.
    """

    latent_batch = encode_latent_batch(
        autoencoder,
        batch,
        max_words=config.diffusion.max_words,
        latent_dim=config.diffusion.latent_dim,
        dtype=dtype,
        device=device,
        latent_stats=latent_stats,
    )
    z = latent_batch["z"].float()
    mask = latent_batch["mask"]
    labels = torch.full(mask.shape, -100, device=device, dtype=torch.long)
    for index, boundary in enumerate(batch["word_boundaries"]):
        length = int(latent_batch["lengths"][index].item())
        if length <= 0:
            continue
        diffs = (boundary[1:] - boundary[:-1]).to(device=device, dtype=torch.long)[:length]
        labels[index, :length] = diffs.clamp(1, max_length) - 1
    return z, labels, mask


@torch.no_grad()
def evaluate(model: nn.Module, holdout: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> dict[str, float]:
    """Exact-length accuracy / byte MAE / modal baseline over non-padding holdout words."""

    correct = total = 0
    abs_error = 0.0
    label_counts = torch.zeros(model.max_length, dtype=torch.long)
    for z, labels, mask in holdout:
        valid = labels != -100
        predicted = model.predict_lengths(z, mask) - 1
        pv = predicted[valid]
        lv = labels[valid]
        correct += int((pv == lv).sum().item())
        abs_error += float((pv - lv).abs().sum().item())
        total += int(lv.numel())
        label_counts += torch.bincount(lv.cpu(), minlength=model.max_length)
    modal_accuracy = float(label_counts.max().item()) / max(1, total)
    return {
        "accuracy": correct / max(1, total),
        "mae_bytes": abs_error / max(1, total),
        "modal_baseline_accuracy": modal_accuracy,
        "holdout_words": float(total),
    }


# --------------------------------------------------------------------------- matching


def _config_fields(config: DiffusionExperimentConfig) -> dict:
    """The config values an artifact must agree with to count as a match."""

    lp = config.length_predictor
    fields = {
        "arch": lp.arch,
        "space": lp.space,
        "max_length": lp.max_length,
        "input_dim": config.diffusion.latent_dim,
        "autoencoder_checkpoint": config.autoencoder.checkpoint,
    }
    if lp.arch == "context":
        fields["model_dim"] = lp.model_dim
        fields["num_layers"] = lp.num_layers
        fields["num_heads"] = lp.num_heads
    else:
        fields["hidden_dim"] = lp.hidden_dim
    if lp.space == "standardized":
        fields["latent_stats_path"] = config.autoencoder.latent_stats_path
    return fields


def artifact_matches(artifact: dict, config: DiffusionExperimentConfig) -> tuple[bool, list[str]]:
    diffs: list[str] = []
    for key, want in _config_fields(config).items():
        got = artifact.get(key)
        if got != want:
            diffs.append(f"  {key}: artifact={got!r} config={want!r}")
    return (not diffs), diffs


# --------------------------------------------------------------------------- training


def build_model(config: DiffusionExperimentConfig, device: torch.device) -> nn.Module:
    lp = config.length_predictor
    if lp.arch == "context":
        model: nn.Module = LengthPredictorV2(
            input_dim=config.diffusion.latent_dim,
            max_words=config.diffusion.max_words,
            max_length=lp.max_length,
            model_dim=lp.model_dim,
            num_layers=lp.num_layers,
            num_heads=lp.num_heads,
        )
    elif lp.arch == "mlp":
        model = LengthPredictorV1(config.diffusion.latent_dim, lp.hidden_dim, lp.max_length)
    else:
        raise SystemExit(f"Unknown length_predictor.arch {lp.arch!r} (expected 'context' or 'mlp').")
    model.space = lp.space
    return model.to(device)


def forward_logits(model: nn.Module, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if isinstance(model, LengthPredictorV2):
        return model(z, key_padding_mask=~mask)
    return model(z)


def train(
    config: DiffusionExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
    v1_baseline_path: str,
) -> tuple[nn.Module, dict]:
    lp = config.length_predictor
    autoencoder = load_frozen_autoencoder(config, device)

    # Raw space: encode without standardization (mirror generate.py's z_raw). Standardized
    # space: standardize with the config's latent stats.
    latent_stats = None
    if lp.space == "standardized":
        if not config.autoencoder.latent_stats_path:
            raise SystemExit("length_predictor.space=standardized requires autoencoder.latent_stats_path.")
        latent_stats = load_latent_stats(config.autoencoder.latent_stats_path, device, config.diffusion.latent_dim)

    # noise_std robustness: per-dim Gaussian noise scaled by the latent std (needs the stats).
    noise_scale = None
    if lp.noise_std > 0:
        if not config.autoencoder.latent_stats_path:
            raise SystemExit("length_predictor.noise_std>0 requires autoencoder.latent_stats_path for per-dim std.")
        stats = load_latent_stats(config.autoencoder.latent_stats_path, device, config.diffusion.latent_dim)
        noise_scale = lp.noise_std * stats["std"]

    loader, _ = build_training_loader(config)
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lp.lr)

    holdout: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    sequences = 0
    step = 0
    model.train()
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        z, labels, mask = sequence_examples(autoencoder, batch, config, dtype, device, latent_stats, lp.max_length)
        if len(holdout) < lp.holdout_batches:
            holdout.append((z, labels, mask))
            continue
        if noise_scale is not None:
            z = z + noise_scale * torch.randn_like(z)
        logits = forward_logits(model, z, mask)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        sequences += int(batch["byte_ids"].size(0))
        if step % 100 == 0:
            print(f"step={step} sequences={sequences} loss={loss.item():.4f}", flush=True)
        if sequences >= lp.num_sequences:
            break

    model.eval()
    metrics = evaluate(model, holdout)
    metrics["train_sequences"] = float(sequences)

    # Same-holdout v1 baseline comparison (raw space only; the deployed v1 artifact is raw).
    baseline_path = Path(v1_baseline_path)
    if baseline_path.exists() and lp.space == "raw":
        try:
            v1 = load_length_predictor(str(baseline_path), device)
            if v1.input_dim == config.diffusion.latent_dim:
                metrics["v1_baseline_path"] = str(baseline_path)
                metrics["v1_baseline"] = evaluate(v1, holdout)
            else:
                print(
                    f"NOTE: v1 baseline input_dim {v1.input_dim} != latent_dim "
                    f"{config.diffusion.latent_dim}; skipping baseline eval.",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001 - baseline is diagnostic only, never fatal
            print(f"NOTE: could not evaluate v1 baseline {baseline_path}: {exc}", file=sys.stderr)
    elif lp.space != "raw":
        print("NOTE: v1 baseline eval skipped (only comparable in raw space).", file=sys.stderr)

    return model, metrics


def save_artifact(model: nn.Module, config: DiffusionExperimentConfig, metrics: dict, config_path: str, output: Path) -> None:
    lp = config.length_predictor
    artifact = {
        "format_version": 2,
        "arch": lp.arch,
        "state_dict": model.state_dict(),
        "input_dim": config.diffusion.latent_dim,
        "max_length": lp.max_length,
        "space": lp.space,
        "model_dim": lp.model_dim,
        "num_layers": lp.num_layers,
        "num_heads": lp.num_heads,
        "hidden_dim": lp.hidden_dim,  # only meaningful for arch=mlp; kept for the factory
        "autoencoder_checkpoint": config.autoencoder.checkpoint,
        "latent_stats_path": config.autoencoder.latent_stats_path,
        "noise_std": lp.noise_std,
        "metrics": metrics,
        "config_path": config_path,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)


def _print_metrics(metrics: dict) -> None:
    scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in scalar.items()))
    baseline = metrics.get("v1_baseline")
    if baseline:
        print(f"v1_baseline ({metrics.get('v1_baseline_path')}): "
              + " ".join(f"{k}={v:.4f}" for k, v in baseline.items() if isinstance(v, float)))


# --------------------------------------------------------------------------- CLI


def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    lp = config.length_predictor
    if lp.artifact_path is None:
        raise SystemExit(
            "length_predictor.artifact_path is null/unset in the config; nothing to ensure. "
            "Set it (see configs/training/phase3-diffusion.yaml) to enable the predictor."
        )

    output = Path(lp.artifact_path)
    exists = output.exists()
    matches, diffs = (False, [])
    if exists:
        matches, diffs = artifact_matches(torch.load(output, map_location="cpu"), config)

    # Matching artifact, no --force: report and stop.
    if exists and matches and not args.force:
        print(f"MATCH: {output} matches the config.")
        _print_metrics(torch.load(output, map_location="cpu")["metrics"])
        return

    if args.check_only:
        if not exists:
            raise SystemExit(f"MISSING: {output} does not exist (would train).")
        if not matches:
            print(f"MISMATCH: {output} disagrees with the config on:")
            print("\n".join(diffs))
            raise SystemExit(1)
        print(f"MATCH: {output} matches the config.")
        return

    if exists and not matches and not args.force:
        print(f"MISMATCH: {output} disagrees with the config on:")
        print("\n".join(diffs))
        raise SystemExit("Refusing to overwrite; re-run with --force to retrain.")

    # Missing, or --force: train.
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)
    print(f"Training {lp.arch} predictor -> {output} (device={device}, space={lp.space}, "
          f"num_sequences={lp.num_sequences}, noise_std={lp.noise_std})", flush=True)
    model, metrics = train(config, device, dtype, args.v1_baseline)
    _print_metrics(metrics)
    save_artifact(model, config, metrics, args.config, output)
    print(f"output={output}")


if __name__ == "__main__":
    main()
