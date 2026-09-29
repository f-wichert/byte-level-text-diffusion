import dataclasses
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml


@dataclass
class ModelConfig:
    pretrained_model_id: str = "Aleph-Alpha/tfree-hat-pretrained-7b-base"
    sequence_length: int = 2048
    decoder_layers: int = 8
    decoder_hidden_size: int = 1024
    decoder_self_attention_heads: int = 8
    decoder_cross_attention_heads: int = 32
    head_size: int = 128
    latent_compression_dim: int | None = None
    latent_compression_layers: int = 2  # number of Linear layers per funnel; 1 = single linear
    latent_compression_hidden_dims: list[int] | None = None


@dataclass
class DataSliceConfig:
    dataset: str
    proportion: float
    name: str | None = None
    gated: bool = False
    filter_language: str | None = None
    text_field: str | None = None


@dataclass
class DataConfig:
    streaming: bool = True
    sequence_length: int = 2048
    separator: str = "\n\n"
    slices: dict[str, DataSliceConfig] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    seed: int = 1337
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    lr: float = 3.0e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    grad_clip_norm: float = 1.0
    resume_from_checkpoint: str | None = None
    resume_optimizer: bool | None = None
    reset_lr_schedule_on_resume: bool = False
    init_from_checkpoint: str | None = None
    init_start_step: int | None = None
    checkpoint_every_steps: int = 1000
    validate_every_steps: int = 1000
    log_every_steps: int = 10
    logger: str = "tensorboard"
    output_dir: str = "runs/phase1"
    checkpoint_dir: str = "checkpoints/phase1"
    device: str = "cuda"
    dtype: str = "bfloat16"
    diffusion_dtype: str | None = None
    allow_tf32: bool = False
    drop_unavailable_data: bool = True
    validation_split: str = "train"
    validation_holdout_examples_per_slice: int = 64
    train_skip_examples_per_slice: int = 0
    validation_batches_per_slice: int = 2
    validation_batch_size: int = 1
    latent_shuffle_batch_size: int = 2
    anchor_loss_enabled: bool = False
    anchor_loss_lambda: float = 0.5
    anchor_noise_sigma: float = 0.5
    anchor_noise_sigma_ratio: float = 0.15
    anchor_noise_stats_path: str | None = None
    anchor_noise_sigma_uniform: bool = False
    freeze_encoder_compression: bool = False
    uniform_loss_enabled: bool = False
    uniform_loss_lambda: float = 1.0e-4
    uniform_loss_t: float = 2.0
    uniform_loss_max_latents: int = 512
    isotropy_loss_enabled: bool = False
    isotropy_loss_lambda: float = 0.1
    isotropy_loss_max_latents: int = 0
    isotropy_loss_skip_positions: int = 0
    isotropy_loss_pooled_target: bool = False
    isotropy_loss_pooled_beta: float = 0.1
    predictive_loss_enabled: bool = False
    predictive_loss_lambda: float = 0.1
    predictive_loss_temperature: float = 0.1
    predictive_loss_min_context: int = 1
    predictive_loss_max_queries: int = 0
    predictive_loss_detach_latents: bool = False
    predictive_head_dim: int = 128
    predictive_head_layers: int = 2
    predictive_head_heads: int = 4
    predictive_head_max_positions: int = 512
    master_weights: bool = False
    freeze_encoder: bool = True
    joint_encoder: bool = False
    encoder_lr: float | None = None
    encoder_recon_weight: float = 1.0
    wandb_project: str = "tfree-hat-diffusion-autoencoder"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    wandb_tags: list[str] = field(default_factory=list)
    wandb_group: str | None = None


@dataclass
class AutoencoderRuntimeConfig:
    checkpoint: str | None = None
    local_files_only: bool = True
    freeze: bool = True
    latent_stats_path: str | None = None


@dataclass
class DiffusionModelConfig:
    latent_dim: int = 4096
    model_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    mlp_ratio: float = 4.0
    max_words: int = 256
    dropout: float = 0.0
    objective: str = "flow_matching"
    path: str = "linear"
    path_timeshift: float = 1.0
    t_sampling: str = "uniform"
    t_power: float = 1.0
    sampling_steps: int = 50
    backbone: str = "transformer"  # "transformer" | "dit" | "plain_dit"
    self_conditioning: bool = False  # condition each step on the model's own x̂₀ estimate (dit and plain_dit; the plain "transformer" backbone rejects it)
    prediction: str = "velocity"  # "velocity" (current) | "x0" (predict the clean latent)
    ce_anchor_enabled: bool = False
    ce_anchor_weight: float = 0.3
    ce_anchor_x0_clamp: float = 8.0  # clamp x̂₀ before decoding (high-noise stability); 0 disables
    prefix_cond_enabled: bool = False
    prefix_cond_prob: float = 0.5  # per-sequence probability; read only when enabled
    prefix_cond_mode: str = "data"
    prefix_cond_context_noise: float = 0.0
    prefix_cond_dropout: float = 0.0
    learn_pos_embed: bool = False


@dataclass
class LengthPredictorConfig:

    artifact_path: str | None = None
    arch: str = "context"  # "context" (v2 transformer) | "mlp" (v1)
    space: str = "raw"  # raw | standardized
    max_length: int = 32
    model_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4
    hidden_dim: int = 512
    num_sequences: int = 8192
    holdout_batches: int = 32
    lr: float = 1.0e-3
    noise_std: float = 0.0


@dataclass
class PerplexityConfig:

    enabled: bool = False
    every_steps: int = 0
    mode: str = "prompt"  # prompt | unconditional
    prompt_file: str | None = None
    num_samples: int = 16
    seed: int = 20260724
    sampling_steps: int | None = None
    num_words: int | None = None
    num_bytes: int | None = None
    bytes_per_word: int = 6
    solver: str = "euler"
    decode_mode: str = "lengths"  # lengths | splitter
    conditioning: str = "auto"  # auto | interpolant | clean
    evaluator_model: str = "gpt2-large"
    evaluator_device: str | None = None
    evaluator_local_files_only: bool = False
    max_eval_tokens: int = 512
    baseline_at_start: bool = True
    baseline_num_texts: int = 16


@dataclass
class Phase1Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class DiffusionExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    autoencoder: AutoencoderRuntimeConfig = field(default_factory=AutoencoderRuntimeConfig)
    diffusion: DiffusionModelConfig = field(default_factory=DiffusionModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    length_predictor: LengthPredictorConfig = field(default_factory=LengthPredictorConfig)
    perplexity: PerplexityConfig = field(default_factory=PerplexityConfig)


def _make_dataclass(cls: type, values: dict[str, Any] | None, *, section: str):
    values = dict(values or {})
    field_names = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(values) - field_names)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in config section {section!r}: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(field_names))}."
        )
    return cls(**values)


def _check_sections(raw: dict[str, Any], allowed: set[str], *, config_path: Path) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown top-level section(s) in {config_path}: {', '.join(unknown)}. "
            f"Valid sections: {', '.join(sorted(allowed))}."
        )


def _parse_data_config(raw: dict[str, Any]) -> DataConfig:
    data_raw = raw.get("data", {}) or {}
    slices_raw = data_raw.get("slices", {}) or {}
    slices = {
        name: _make_dataclass(
            DataSliceConfig, {"dataset": values.get("dataset"), **values}, section=f"data.slices.{name}"
        )
        for name, values in slices_raw.items()
    }
    data_values = dict(data_raw)
    data_values["slices"] = slices
    return _make_dataclass(DataConfig, data_values, section="data")


def _parse_training_config(raw: dict[str, Any]) -> TrainingConfig:
    training_raw = raw.get("training", {}) or {}
    if "betas" in training_raw:
        training_raw = dict(training_raw)
        training_raw["betas"] = tuple(training_raw["betas"])
    return _make_dataclass(TrainingConfig, training_raw, section="training")


# Repo-root anchored config resolution so scripts work from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_CONFIG_DIR = _REPO_ROOT / "configs" / "training"


def resolve_config_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    fallback = _TRAINING_CONFIG_DIR / candidate.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Config not found: {path!r} (also looked for {candidate.name!r} in {_TRAINING_CONFIG_DIR})"
    )


def load_phase1_config(path: str | os.PathLike[str]) -> Phase1Config:
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    _check_sections(raw, {"model", "data", "training"}, config_path=config_path)
    return Phase1Config(
        model=_make_dataclass(ModelConfig, raw.get("model", {}), section="model"),
        data=_parse_data_config(raw),
        training=_parse_training_config(raw),
    )


def _validate_diffusion_config(config: DiffusionExperimentConfig, *, config_path: Path) -> None:

    diffusion = config.diffusion
    if diffusion.prediction not in ("velocity", "x0"):
        raise ValueError(
            f"{config_path}: diffusion.prediction must be 'velocity' or 'x0', got "
            f"{diffusion.prediction!r}. Every prediction-mode branch tests == 'x0' with velocity "
            "as the fallback, so an unrecognised value here silently trains velocity."
        )
    if diffusion.ce_anchor_enabled and diffusion.ce_anchor_weight <= 0:
        raise ValueError(
            f"{config_path}: diffusion.ce_anchor_enabled is true but ce_anchor_weight is "
            f"{diffusion.ce_anchor_weight}, so the anchor would cost a full decoder "
            "forward+backward per micro-batch and contribute nothing. Set a positive weight or "
            "disable the anchor."
        )


def load_diffusion_config(path: str | os.PathLike[str]) -> DiffusionExperimentConfig:

    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    _check_sections(
        raw,
        {"model", "autoencoder", "diffusion", "data", "training", "length_predictor", "perplexity"},
        config_path=config_path,
    )
    config = DiffusionExperimentConfig(
        model=_make_dataclass(ModelConfig, raw.get("model", {}), section="model"),
        autoencoder=_make_dataclass(AutoencoderRuntimeConfig, raw.get("autoencoder", {}), section="autoencoder"),
        diffusion=_make_dataclass(DiffusionModelConfig, raw.get("diffusion", {}), section="diffusion"),
        data=_parse_data_config(raw),
        training=_parse_training_config(raw),
        length_predictor=_make_dataclass(
            LengthPredictorConfig, raw.get("length_predictor", {}), section="length_predictor"
        ),
        perplexity=_make_dataclass(PerplexityConfig, raw.get("perplexity", {}), section="perplexity"),
    )
    _validate_diffusion_config(config, config_path=config_path)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def module_dtype(module: torch.nn.Module, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    return default


def save_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "step": step,
        "extra": extra or {},
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    allow_missing: set[str] | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if strict:
        model.load_state_dict(checkpoint["model"])
    else:
        result = model.load_state_dict(checkpoint["model"], strict=False)
        unexpected = set(result.unexpected_keys)
        missing = set(result.missing_keys)
        permitted = allow_missing or set()
        if unexpected:
            raise ValueError(f"Unexpected keys loading {path}: {sorted(unexpected)}")
        if not missing <= permitted:
            raise ValueError(
                f"Missing keys loading {path} that are not permitted: {sorted(missing - permitted)}"
            )
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


class Logger:
    def log(self, values: dict[str, float], step: int) -> None:
        raise NotImplementedError

    def log_table(self, key: str, columns: list[str], rows: list[list[Any]], step: int) -> None:
        """Log tabular data (e.g. per-sample eval rows). Cells may be str or numeric.

        Default is a no-op rather than NotImplementedError: tables are a wandb-only
        side channel, and callers should not have to know which backend they hold.
        """
        return None

    def close(self) -> None:
        pass


class NoopLogger(Logger):
    def log(self, values: dict[str, float], step: int) -> None:
        return None


class StdoutLogger(Logger):
    def log(self, values: dict[str, float], step: int) -> None:
        parts = []
        for key, value in values.items():
            if isinstance(value, (list, tuple)):
                mean = sum(value) / max(1, len(value))
                parts.append(f"{key}=hist(n={len(value)},mean={mean:.4g})")
            else:
                parts.append(f"{key}={_to_float(value):.6g}")
        print(f"step={step} {' '.join(parts)}", flush=True)


class TensorBoardLogger(Logger):
    def __init__(self, log_dir: str | os.PathLike[str]):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log(self, values: dict[str, float], step: int) -> None:
        for key, value in values.items():
            if isinstance(value, (list, tuple)):
                self.writer.add_histogram(key, torch.tensor(value, dtype=torch.float32), step)
            else:
                self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()


class WandbLogger(Logger):
    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        project: str,
        entity: str | None = None,
        run_name: str | None = None,
        mode: str = "online",
        tags: Sequence[str] | None = None,
        group: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        import wandb

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        wandb_state_dir = output_path / "wandb-state"
        for env_name, dirname in (
            ("WANDB_CACHE_DIR", "cache"),
            ("WANDB_CONFIG_DIR", "config"),
            ("WANDB_DATA_DIR", "data"),
        ):
            path = wandb_state_dir / dirname
            path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault(env_name, str(path))

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            mode=mode,
            tags=list(tags or []),
            group=group,
            dir=str(output_path),
            config=config,
        )
        # Incremental tables must be the *same* object across log calls; keyed storage
        # lives here rather than in callers so the facade contract stays per-eval rows.
        self._tables: dict[str, Any] = {}

    def log(self, values: dict[str, float], step: int) -> None:
        import wandb

        payload = {
            # Lists/tuples log as histograms (length/cutoff distributions); scalars as before.
            key: (wandb.Histogram(value) if isinstance(value, (list, tuple)) else _to_float(value))
            for key, value in values.items()
        }
        self.run.log(payload, step=step)

    def log_table(self, key: str, columns: list[str], rows: list[list[Any]], step: int) -> None:
        import wandb

        # One INCREMENTAL table per key, held for the life of the run: each call appends
        # its rows and re-logs the same object, so the UI shows the accumulated table
        # (rows carry a step column). A fresh table per call would version instead --
        # the panel then displays only the latest eval's rows.
        table = self._tables.get(key)
        if table is None:
            table = wandb.Table(columns=columns, log_mode="INCREMENTAL")
            self._tables[key] = table
        for row in rows:
            table.add_data(*row)
        self.run.log({key: table}, step=step)

    def close(self) -> None:
        self.run.finish()


def _to_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def create_logger(
    kind: str | TrainingConfig,
    output_dir: str | os.PathLike[str] | None = None,
    config: dict[str, Any] | None = None,
) -> Logger:
    training = kind if isinstance(kind, TrainingConfig) else None
    logger_kind = training.logger if training else str(kind)
    log_dir = output_dir or (training.output_dir if training else None)
    if log_dir is None:
        raise ValueError("output_dir is required when create_logger is called without TrainingConfig.")

    logger_kind = logger_kind.lower()
    if logger_kind == "tensorboard":
        return TensorBoardLogger(log_dir)
    if logger_kind in {"stdout", "terminal", "print"}:
        return StdoutLogger()
    if logger_kind == "wandb":
        if training is None:
            return WandbLogger(log_dir, project="tfree-hat-diffusion-autoencoder", config=config)
        logger = WandbLogger(
            log_dir,
            project=training.wandb_project,
            entity=training.wandb_entity,
            run_name=training.wandb_run_name,
            mode=training.wandb_mode,
            tags=training.wandb_tags,
            group=training.wandb_group,
            config=config,
        )
        config_path = Path(log_dir) / "config_used.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        logger.run.save(str(config_path), policy="now")
        return logger
    if logger_kind in {"none", "noop", ""}:
        return NoopLogger()
    raise ValueError(f"Unsupported logger kind: {logger_kind}")
