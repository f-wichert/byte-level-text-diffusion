import dataclasses
import math
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from src.data import TextCollator, build_mixed_stream, load_dataset_slice, make_dataloader
from src.model import DiffusionAutoencoder
from src.utils import Phase1Config, TrainingConfig, create_logger, save_checkpoint, seed_everything, load_checkpoint


@dataclass
class TrainResult:
    final_step: int
    trainable_parameters: int
    checkpoint_paths: list[str] = field(default_factory=list)
    validation_losses: dict[str, float] = field(default_factory=dict)
    latent_shuffle: dict[str, float] = field(default_factory=dict)
    logger_url: str | None = None


def parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    norm_param_ids = set()
    for module in model.modules():
        module_name = module.__class__.__name__.lower()
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)) or "rmsnorm" in module_name:
            norm_param_ids.update(id(parameter) for parameter in module.parameters(recurse=False))

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or id(parameter) in norm_param_ids:
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def learning_rate_for_step(step: int, training: TrainingConfig, max_steps: int | None = None) -> float:
    max_steps = training.max_steps if max_steps is None else max_steps
    if training.warmup_steps > 0 and step <= training.warmup_steps:
        return training.lr * step / training.warmup_steps

    decay_steps = max(1, max_steps - training.warmup_steps)
    progress = min(1.0, max(0.0, (step - training.warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return training.lr * (training.min_lr_ratio + (1.0 - training.min_lr_ratio) * cosine)


class MasterWeightAdamW:
    """AdamW over float32 master copies of a (typically bfloat16) model's parameters.

    This trainer cannot take the diffusion trainer's escape from pure-bf16 AdamW
    (training.diffusion_dtype casts the whole model; the vendored HAT stack must run
    bfloat16 forward), so this is the classic mixed-precision recipe instead
    (Micikevicius et al. 2018): autograd computes bf16 grads on the model as usual;
    step() copies them into fp32 masters, steps AdamW there, and writes the masters back
    into the model parameters. Sub-ULP updates therefore accumulate in fp32 instead of
    rounding away (at lr 1e-4 a step is below one bf16 ULP at |w| ~ 0.03; measured on
    this project's phase-3 checkpoints, pure-bf16 AdamW left 17%% of parameters
    bit-frozen across 10k steps). Gradients themselves stay bf16-quantized -- ordinary
    bf16 training; the accumulator was the bug.

    Matches torch.optim.AdamW where train() touches it: ``param_groups`` (lr schedule),
    ``zero_grad``, ``step``, ``state_dict``/``load_state_dict``. The state dict includes
    the master tensors: without them a resume would re-round the masters from bf16 and
    silently discard the accumulated sub-ULP state.
    """

    def __init__(self, groups: list[dict[str, Any]], lr: float, betas: tuple[float, float]):
        self._model_params: list[torch.nn.Parameter] = []
        self._masters: list[torch.Tensor] = []
        master_groups: list[dict[str, Any]] = []
        for group in groups:
            masters = []
            for parameter in group["params"]:
                master = parameter.detach().clone().float()
                masters.append(master)
                self._model_params.append(parameter)
                self._masters.append(master)
            master_groups.append({**{k: v for k, v in group.items() if k != "params"}, "params": masters})
        self.optimizer = torch.optim.AdamW(master_groups, lr=lr, betas=betas)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self._model_params:
            parameter.grad = None
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self) -> None:
        # Grads are copied AFTER the caller's clip_grad_norm_ on the model params, so
        # the masters see the clipped values.
        for parameter, master in zip(self._model_params, self._masters):
            master.grad = None if parameter.grad is None else parameter.grad.float()
        self.optimizer.step()
        for parameter, master in zip(self._model_params, self._masters):
            parameter.copy_(master.to(parameter.dtype))

    @torch.no_grad()
    def sync_masters_from_model(self) -> None:
        """Re-copy the model parameters into the fp32 masters.

        Must be called after any external mutation of the model weights that happens
        AFTER this optimizer was constructed (e.g. a resume/warm-start checkpoint load):
        step() writes the masters back into the model, so stale masters would silently
        revert that mutation on the first step. A no-op when nothing was mutated.
        """
        for parameter, master in zip(self._model_params, self._masters):
            master.copy_(parameter.detach().float())

    def state_dict(self) -> dict[str, Any]:
        return {"adamw": self.optimizer.state_dict(), "masters": self._masters}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["adamw"])
        with torch.no_grad():
            for master, saved in zip(self._masters, state["masters"]):
                master.copy_(saved.to(dtype=master.dtype, device=master.device))
            # The masters are authoritative over the (rounded) bf16 checkpoint weights.
            for parameter, master in zip(self._model_params, self._masters):
                parameter.copy_(master.to(parameter.dtype))


def build_optimizer(model: torch.nn.Module, training: TrainingConfig) -> torch.optim.Optimizer | MasterWeightAdamW:
    groups = parameter_groups(model, training.weight_decay)
    if training.master_weights:
        return MasterWeightAdamW(groups, lr=training.lr, betas=training.betas)
    return torch.optim.AdamW(groups, lr=training.lr, betas=training.betas)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        elif key in {"word_boundaries", "cumulative_seq_lengths_per_word"}:
            moved[key] = [boundary.to(device=device, dtype=torch.int32) for boundary in value]
        else:
            moved[key] = value
    return moved


def cycle_batches(loader: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _loss_sum_and_tokens(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    tokens = int((labels != -100).sum().item())
    return loss_sum, tokens


def build_training_loader(config: Phase1Config):
    collator = TextCollator(sequence_length=config.data.sequence_length)
    skip_examples_per_slice = (
        config.training.validation_holdout_examples_per_slice
        + config.training.train_skip_examples_per_slice
    )
    stream, names, probabilities, errors = build_mixed_stream(
        config.data,
        split="train",
        drop_unavailable=config.training.drop_unavailable_data,
        skip_examples_per_slice=skip_examples_per_slice,
    )
    loader = make_dataloader(stream, collator=collator, batch_size=config.training.batch_size)
    return loader, {"loaded_slices": names, "probabilities": probabilities, "errors": errors}


def build_validation_loaders(config: Phase1Config) -> dict[str, Any]:
    collator = TextCollator(sequence_length=config.data.sequence_length)
    loaders = {}
    for name, slice_config in config.data.slices.items():
        try:
            dataset = load_dataset_slice(name, slice_config, split=config.training.validation_split)
            dataset = dataset.take(config.training.validation_holdout_examples_per_slice)
        except Exception:
            if not config.training.drop_unavailable_data:
                raise
            continue
        loaders[name] = make_dataloader(dataset, collator=collator, batch_size=config.training.validation_batch_size)
    return loaders


@torch.no_grad()
def evaluate_per_slice(
    model: torch.nn.Module,
    validation_loaders: dict[str, Any],
    device: torch.device,
    batches_per_slice: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses = {}
    for name, loader in validation_loaders.items():
        total_loss = 0.0
        total_tokens = 0
        for batch_index, batch in enumerate(loader):
            if batch_index >= batches_per_slice:
                break
            batch = move_batch_to_device(batch, device)
            output = model(**batch)
            loss_sum, tokens = _loss_sum_and_tokens(output["logits"], batch["labels"])
            total_loss += float(loss_sum.item())
            total_tokens += tokens
        if total_tokens > 0:
            losses[name] = total_loss / total_tokens
    if was_training:
        model.train()
    return losses


def _fit_latent_to_word_count(z_words: torch.Tensor, word_count: int) -> torch.Tensor:
    current = z_words.size(1)
    if current == word_count:
        return z_words
    if current > word_count:
        return z_words[:, :word_count, :]
    pad = z_words[:, -1:, :].expand(-1, word_count - current, -1)
    return torch.cat([z_words, pad], dim=1)


def latent_uniformity_loss(
    z_words: Iterable[torch.Tensor],
    t: float = 2.0,
    max_latents: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = [z.reshape(-1, z.size(-1)) for z in z_words if z.numel() > 0]
    if not latents:
        fallback = torch.tensor(0.0)
        return fallback, torch.tensor(float("nan"))

    z = torch.cat(latents, dim=0).float()
    if z.size(0) < 2:
        return z.new_tensor(0.0), z.new_tensor(float("nan"))

    if max_latents > 0 and z.size(0) > max_latents:
        indices = torch.randperm(z.size(0), device=z.device)[:max_latents]
        z = z[indices]

    z = F.normalize(z, dim=-1)
    sq_pdist = torch.pdist(z, p=2).pow(2)
    uniform_loss = sq_pdist.mul(-t).exp().mean().log()
    cosine_mean = 1.0 - 0.5 * sq_pdist.mean()
    return uniform_loss, cosine_mean


def latent_isotropy_loss(
    z_words: Iterable[torch.Tensor],
    max_latents: int = 0,
    skip_positions: int = 0,
    target_moments: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    latents = [z.reshape(-1, z.size(-1))[skip_positions:] for z in z_words if z.numel() > 0]
    if not latents:
        return torch.tensor(0.0), {}

    z = torch.cat(latents, dim=0).float()
    if max_latents > 0 and z.size(0) > max_latents:
        indices = torch.randperm(z.size(0), device=z.device)[:max_latents]
        z = z[indices]
    if z.size(0) < 2:
        return z.new_zeros(()), {}

    mean = z.mean(dim=0)
    centered = z - mean
    covariance = centered.T @ centered / (z.size(0) - 1)
    identity = torch.eye(covariance.size(0), device=z.device, dtype=z.dtype)
    if target_moments is None:
        loss = (covariance - identity).pow(2).mean() + mean.pow(2).mean()
    else:
        pooled_mean, pooled_cov = target_moments
        cov_deviation = (pooled_cov - identity).detach()
        loss = (
            2.0 * ((covariance - identity) * cov_deviation).mean()
            + 2.0 * (mean * pooled_mean.detach()).mean()
        )
    detached = z.detach()
    stats = {
        "n": torch.tensor(float(detached.size(0))),
        "sum": detached.sum(dim=0),
        "sum_outer": detached.T @ detached,
    }
    return loss, stats


class PredictiveHead(torch.nn.Module):

    def __init__(
        self,
        latent_dim: int,
        model_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_positions: int = 512,
    ):
        super().__init__()
        self.max_positions = max_positions
        self.proj_in = torch.nn.Linear(latent_dim, model_dim)
        self.positions = torch.nn.Embedding(max_positions, model_dim)
        layer = torch.nn.TransformerEncoderLayer(
            model_dim,
            num_heads,
            model_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers)
        self.proj_out = torch.nn.Linear(model_dim, latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [1, W, latent_dim] fp32 -> [1, W, latent_dim]; row i saw z_1..z_i only."""
        width = z.size(1)
        positions = torch.arange(width, device=z.device).clamp_max(self.max_positions - 1)
        hidden = self.proj_in(z) + self.positions(positions)[None]
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(width, device=z.device)
        hidden = self.encoder(hidden, mask=causal_mask)
        return self.proj_out(hidden)


def latent_predictive_loss(
    z_words: Iterable[torch.Tensor],
    head: PredictiveHead,
    temperature: float = 0.1,
    min_context: int = 1,
    max_queries: int = 0,
    detach_latents: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    docs = [z for z in z_words if z.numel() > 0]
    if not docs:
        return torch.tensor(0.0), {}
    all_z = torch.cat([z.reshape(-1, z.size(-1)) for z in docs], dim=0).float()
    if detach_latents:
        all_z = all_z.detach()
    if all_z.size(0) < 2 or all(z.size(1) <= min_context for z in docs):
        return all_z.new_zeros(()), {}
    candidates = F.normalize(all_z, dim=-1)

    queries: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    offset = 0
    for z in docs:
        width = z.size(1)
        if width > min_context:
            z_input = z.float().detach() if detach_latents else z.float()
            q = head(z_input)[0]
            index = torch.arange(min_context - 1, width - 1, device=q.device)
            queries.append(q[index])
            targets.append(offset + index + 1)
        offset += width
    q = F.normalize(torch.cat(queries, dim=0), dim=-1)
    target = torch.cat(targets, dim=0)

    if max_queries > 0 and q.size(0) > max_queries:
        keep = torch.randperm(q.size(0), device=q.device)[:max_queries]
        q, target = q[keep], target[keep]

    logits = q @ candidates.T / temperature
    loss = F.cross_entropy(logits, target)

    with torch.no_grad():
        n_queries = float(target.numel())
        n_candidates = float(candidates.size(0))
        hits = (logits.argmax(dim=-1) == target).sum().float()
        stats = {
            "top1_correct": hits,
            "n_queries": torch.tensor(n_queries),
            "n_candidates": torch.tensor(n_candidates),
            "mi_lower_bound": torch.tensor(math.log(n_candidates)) - loss.detach().cpu(),
        }
    return loss, stats


@torch.no_grad()
def latent_shuffle_check(model: torch.nn.Module, batch: dict[str, Any], device: torch.device) -> dict[str, float]:
    batch = move_batch_to_device(batch, device)
    byte_ids = batch["byte_ids"]
    labels = batch["labels"]
    boundaries = batch["word_boundaries"]
    z_words = model.encode(byte_ids, boundaries)
    correct_logits = model.decode(byte_ids, z_words, boundaries)
    correct_sum, tokens = _loss_sum_and_tokens(correct_logits, labels)

    if len(z_words) < 2 or tokens == 0:
        return {"correct_ce": float("nan"), "shuffled_ce": float("nan"), "gap": float("nan")}

    shuffled = []
    for index, boundary in enumerate(boundaries):
        source = z_words[(index + 1) % len(z_words)]
        shuffled.append(_fit_latent_to_word_count(source, int(boundary.numel() - 1)))
    shuffled_logits = model.decode(byte_ids, shuffled, boundaries)
    shuffled_sum, _ = _loss_sum_and_tokens(shuffled_logits, labels)

    correct_ce = float(correct_sum.item()) / tokens
    shuffled_ce = float(shuffled_sum.item()) / tokens
    return {"correct_ce": correct_ce, "shuffled_ce": shuffled_ce, "gap": shuffled_ce - correct_ce}


def build_latent_shuffle_loader(config: Phase1Config):
    collator = TextCollator(sequence_length=config.data.sequence_length)
    stream, _, _, _ = build_mixed_stream(
        config.data,
        split=config.training.validation_split,
        drop_unavailable=config.training.drop_unavailable_data,
    )
    return make_dataloader(stream, collator=collator, batch_size=config.training.latent_shuffle_batch_size)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def diffusion_dtype_from_config(training: TrainingConfig) -> torch.dtype:
    return _dtype_from_name(training.diffusion_dtype or training.dtype)


def apply_tf32(training: TrainingConfig) -> None:
    """Enable TF32 matmuls when the config asks for it (no-op by default)."""
    if training.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def train(
    config: Phase1Config,
    model: torch.nn.Module | None = None,
    train_loader: Iterable[dict[str, Any]] | None = None,
    validation_loaders: dict[str, Any] | None = None,
    latent_shuffle_loader: Iterable[dict[str, Any]] | None = None,
    device: str | torch.device | None = None,
) -> TrainResult:
    seed_everything(config.training.seed)
    device = torch.device(device or config.training.device)

    # Predictive-loss guards run BEFORE the (slow) model load so a bad config fails in
    # milliseconds. With both freezes on, the loss would NOT crash -- head params carry
    # grad, so it silently degrades into a probe that trains only the head (the worst
    # failure mode: looks alive, shapes nothing). Hence a config check, not a crash.
    if config.training.predictive_loss_enabled:
        if config.training.freeze_encoder and config.training.freeze_encoder_compression:
            raise ValueError(
                "predictive_loss_enabled needs a gradient path into the latent: freeze_encoder "
                "and freeze_encoder_compression are both true, so the InfoNCE gradient would "
                "reach only the predictive head (a probe, not a shaping loss). Set "
                "freeze_encoder_compression: false (funnel-only shaping) or freeze_encoder: false."
            )
        if config.model.latent_compression_dim is None:
            raise ValueError(
                "predictive_loss_enabled requires model.latent_compression_dim "
                "(the head predicts compressed latents)."
            )
        if config.training.predictive_loss_lambda <= 0:
            raise ValueError(
                "predictive_loss_enabled with predictive_loss_lambda <= 0 would cost a head "
                "forward+backward per micro-batch and contribute nothing."
            )
        gradient_reach = (
            "compression funnel ONLY (encoder+connector frozen)"
            if config.training.freeze_encoder
            else (
                "encoder + connector (funnel frozen)"
                if config.training.freeze_encoder_compression
                else "encoder + connector + compression funnel"
            )
        )
        print(
            f"NOTICE: predictive InfoNCE gradient reaches: {gradient_reach} "
            f"(freeze_encoder={config.training.freeze_encoder}, "
            f"freeze_encoder_compression={config.training.freeze_encoder_compression})",
            flush=True,
        )
        if not config.training.master_weights:
            print(
                "NOTICE: predictive_loss_enabled without master_weights: encoder-side gradients "
                "from a small aux loss are exactly what bf16 AdamW rounds away.",
                flush=True,
            )

    if model is None:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        model = DiffusionAutoencoder.from_pretrained(
            config.model,
            torch_dtype=_dtype_from_name(config.training.dtype),
            device=device,
            freeze_encoder=config.training.freeze_encoder,
        )
    else:
        model = model.to(device)

    # if hasattr(model, "freeze_encoder"):
    #     model.freeze_encoder()
    if config.training.freeze_encoder_compression and getattr(model, "encoder_compression", None) is not None:
        # The compression funnel defines the latent space; freeze BEFORE the optimizer
        # is built so its params never enter the param groups.
        model.encoder_compression.requires_grad_(False)
    model.train()

    # Predictive head: a trainer-owned sidecar, NEVER registered on `model` (the resume
    # path below strict-loads model.state_dict(), so a model-side submodule would break
    # resume of every existing checkpoint). fp32 on purpose (no flash-attn constraint at
    # this size; exact masters). Constructed AFTER the model (flag-on runs share model
    # init bit-exactly with flag-off runs at the same seed) and BEFORE build_optimizer
    # (params added later would be invisible to the optimizer and actively reverted by
    # the master write-back).
    predictive_head: PredictiveHead | None = None
    if config.training.predictive_loss_enabled:
        predictive_head = PredictiveHead(
            latent_dim=config.model.latent_compression_dim,
            model_dim=config.training.predictive_head_dim,
            num_layers=config.training.predictive_head_layers,
            num_heads=config.training.predictive_head_heads,
            max_positions=config.training.predictive_head_max_positions,
        ).to(device=device, dtype=torch.float32)
        predictive_head.train()
    # Flag-off this IS `model` (same object), so optimizer/clip/param-count behavior is
    # byte-identical to the pre-existing code.
    optimizer_model: torch.nn.Module = (
        model
        if predictive_head is None
        else torch.nn.ModuleDict({"model": model, "predictive_head": predictive_head})
    )

    optimizer = build_optimizer(optimizer_model, config.training)

    start_step = 1
    resume_step = 0
    if config.training.resume_from_checkpoint:
        checkpoint = load_checkpoint(
            config.training.resume_from_checkpoint,
            model=model,
            # With a predictive head, the optimizer's trainable set includes the head, so
            # its saved state is loaded manually below AFTER the head weights are known
            # to exist in the checkpoint (an in-load failure would surface as torch's
            # cryptic group-size error before our guard could explain it).
            optimizer=optimizer if (config.training.resume_optimizer and predictive_head is None) else None,
            map_location=device
        )
        resume_step = int(checkpoint["step"])
        start_step = resume_step + 1
        if config.training.max_steps <= resume_step:
            raise ValueError(
                f"max_steps ({config.training.max_steps}) must be greater than the resumed "
                f"checkpoint step ({resume_step})."
            )
        if predictive_head is not None:
            head_state = (checkpoint.get("extra") or {}).get("predictive_head_state")
            if head_state is not None:
                predictive_head.load_state_dict(head_state)
                if config.training.resume_optimizer:
                    optimizer.load_state_dict(checkpoint["optimizer"])
            elif config.training.resume_optimizer:
                raise ValueError(
                    f"resume_from_checkpoint ({config.training.resume_from_checkpoint}) has no "
                    "predictive_head_state (pre-InfoNCE checkpoint) but predictive_loss_enabled "
                    "is on: the optimizer's trainable set changed, so its saved state cannot be "
                    "loaded. Set resume_optimizer: false (the house rule for trainable-set "
                    "changes) to warm-start with a fresh head and fresh optimizer."
                )
            else:
                print(
                    "NOTICE: predictive head initialized FRESH "
                    "(warm start from a pre-InfoNCE checkpoint).",
                    flush=True,
                )
        elif (checkpoint.get("extra") or {}).get("predictive_head_state") is not None:
            print(
                "NOTICE: checkpoint carries a predictive head; predictive_loss_enabled is "
                "false, so it is dropped.",
                flush=True,
            )
        # The optimizer is constructed above, BEFORE this load, so master weights (if on)
        # snapshotted the init weights: re-sync them to the checkpoint weights or step 1
        # would silently revert the load. With resume_optimizer the masters were loaded
        # from the checkpoint and this is a no-op. Ordering is load-bearing: head load ->
        # optional optimizer load -> this sync (which walks optimizer_model, head included).
        if isinstance(optimizer, MasterWeightAdamW):
            optimizer.sync_masters_from_model()

    trainable_parameters = count_trainable_parameters(optimizer_model)

    if train_loader is None:
        train_loader, data_info = build_training_loader(config)
    else:
        data_info = {"loaded_slices": [], "probabilities": [], "errors": {}}
    if validation_loaders is None:
        validation_loaders = build_validation_loaders(config)
    if latent_shuffle_loader is None:
        latent_shuffle_loader = build_latent_shuffle_loader(config)

    logger = create_logger(config.training, config=dataclasses.asdict(config))
    logger_url = getattr(getattr(logger, "run", None), "url", None)
    result = TrainResult(final_step=0, trainable_parameters=trainable_parameters, logger_url=logger_url)
    train_batches = cycle_batches(train_loader)
    latent_batches = cycle_batches(latent_shuffle_loader)
    checkpoint_dir = Path(config.training.checkpoint_dir)

    anchor_noise_std: torch.Tensor | None = None
    if config.training.anchor_loss_enabled and config.training.anchor_noise_stats_path:
        stats_artifact = torch.load(config.training.anchor_noise_stats_path, map_location=device)
        anchor_noise_std = stats_artifact["std"].to(device=device).flatten()

    try:
        logger.log({"params/trainable": float(trainable_parameters)}, step=0)
        # EMA of the per-step pooled isotropy moments (isotropy_loss_pooled_target mode).
        # Not checkpointed: a resume re-warms in one optimizer step.
        iso_ema_mean: torch.Tensor | None = None
        iso_ema_cov: torch.Tensor | None = None
        for step in range(start_step, config.training.max_steps + 1):
            model.train()
            schedule_step = step
            schedule_max_steps = config.training.max_steps
            if config.training.reset_lr_schedule_on_resume and resume_step > 0:
                schedule_step = step - resume_step
                schedule_max_steps = config.training.max_steps - resume_step
            step_lr = learning_rate_for_step(schedule_step, config.training, max_steps=schedule_max_steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_loss_clean = 0.0
            step_loss_noisy = 0.0
            step_loss_noisy_weighted = 0.0
            step_loss_uniform = 0.0
            step_loss_uniform_weighted = 0.0
            step_cosine_mean = 0.0
            step_cosine_count = 0
            step_loss_iso = 0.0
            step_loss_iso_weighted = 0.0
            # Sufficient statistics pooled across the optimizer step's micro-batches:
            # per-micro-batch covariance eigenvalues are estimation noise (Marchenko-
            # Pastur spread ~ (1 +- sqrt(d/N))^2), the pooled ones are readable.
            iso_count = 0.0
            iso_sum: torch.Tensor | None = None
            iso_sum_outer: torch.Tensor | None = None
            step_loss_pred = 0.0
            step_loss_pred_weighted = 0.0
            # Poolable sufficient statistics (pooled acc = sum correct / sum queries;
            # per-micro-batch accuracies are estimation noise, same discipline as iso).
            step_pred_correct = 0.0
            step_pred_queries = 0.0
            step_pred_log_candidates = 0.0
            step_pred_batches = 0
            step_tokens = 0
            started_at = time.perf_counter()

            for _ in range(config.training.gradient_accumulation_steps):
                batch = move_batch_to_device(next(train_batches), device)
                if config.training.anchor_loss_enabled:
                    byte_ids = batch.get("byte_ids", batch.get("input_ids"))
                    word_boundaries = batch.get("word_boundaries", batch.get("cumulative_seq_lengths_per_word"))
                    labels = batch.get("labels")
                    
                    z_words = model.encode(byte_ids, word_boundaries)

                    clean = model.forward_from_latents(
                        byte_ids=byte_ids,
                        z_words=z_words,
                        word_boundaries=word_boundaries,
                        labels=labels
                    )

                    sigma = config.training.anchor_noise_sigma
                    if config.training.anchor_noise_sigma_uniform:
                        sigma = sigma * float(torch.rand(()))
                    if anchor_noise_std is not None:
                        # Standardized-unit noise: per-dim scaled so sigma matches how
                        # diffusion-sampling error is distributed across latent dims.
                        noisy_z_words = [
                            z + sigma * anchor_noise_std.to(device=z.device, dtype=z.dtype) * torch.randn_like(z)
                            for z in z_words
                        ]
                    else:
                        noisy_z_words = [z + sigma * torch.randn_like(z) for z in z_words]

                    noisy = model.forward_from_latents(
                        byte_ids=byte_ids,
                        z_words=noisy_z_words,
                        word_boundaries=word_boundaries,
                        labels=labels
                    )

                    loss = clean["loss"] + config.training.anchor_loss_lambda * noisy["loss"]
                else:
                    output = model(**batch)
                    z_words = output["z_words"]
                    loss = output["loss"]
                if config.training.uniform_loss_enabled:
                    uniform_loss, cosine_mean = latent_uniformity_loss(
                        z_words,
                        t=config.training.uniform_loss_t,
                        max_latents=config.training.uniform_loss_max_latents,
                    )
                    uniform_loss_weighted = config.training.uniform_loss_lambda * uniform_loss
                    loss = loss + uniform_loss_weighted
                    step_loss_uniform += float(uniform_loss.item())
                    step_loss_uniform_weighted += float(uniform_loss_weighted.item())
                    if torch.isfinite(cosine_mean):
                        step_cosine_mean += float(cosine_mean.item())
                        step_cosine_count += 1
                if config.training.isotropy_loss_enabled:
                    iso_target = None
                    if config.training.isotropy_loss_pooled_target and iso_ema_cov is not None:
                        iso_target = (iso_ema_mean, iso_ema_cov)
                    iso_loss, iso_stats = latent_isotropy_loss(
                        z_words,
                        max_latents=config.training.isotropy_loss_max_latents,
                        skip_positions=config.training.isotropy_loss_skip_positions,
                        target_moments=iso_target,
                    )
                    if config.training.isotropy_loss_pooled_target and iso_target is None:
                        # EMA warmup (first step, or first after a resume): statistics
                        # only -- the micro-batch measurement the legacy loss would
                        # chase here is exactly what pooled mode exists to avoid.
                        iso_loss = iso_loss.detach() * 0.0
                    iso_loss_weighted = config.training.isotropy_loss_lambda * iso_loss
                    loss = loss + iso_loss_weighted
                    step_loss_iso += float(iso_loss.item())
                    step_loss_iso_weighted += float(iso_loss_weighted.item())
                    if iso_stats:
                        iso_count += float(iso_stats["n"].item())
                        iso_sum = iso_stats["sum"] if iso_sum is None else iso_sum + iso_stats["sum"]
                        iso_sum_outer = (
                            iso_stats["sum_outer"]
                            if iso_sum_outer is None
                            else iso_sum_outer + iso_stats["sum_outer"]
                        )
                if config.training.predictive_loss_enabled:
                    pred_loss, pred_stats = latent_predictive_loss(
                        z_words,
                        predictive_head,
                        temperature=config.training.predictive_loss_temperature,
                        min_context=config.training.predictive_loss_min_context,
                        max_queries=config.training.predictive_loss_max_queries,
                        detach_latents=config.training.predictive_loss_detach_latents,
                    )
                    pred_loss_weighted = config.training.predictive_loss_lambda * pred_loss
                    loss = loss + pred_loss_weighted
                    step_loss_pred += float(pred_loss.item())
                    step_loss_pred_weighted += float(pred_loss_weighted.item())
                    if pred_stats:
                        step_pred_correct += float(pred_stats["top1_correct"].item())
                        step_pred_queries += float(pred_stats["n_queries"].item())
                        step_pred_log_candidates += math.log(float(pred_stats["n_candidates"].item()))
                        step_pred_batches += 1
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
                (loss / config.training.gradient_accumulation_steps).backward()
                tokens = int((batch["labels"] != -100).sum().item())
                step_loss += float(loss.item()) * tokens
                if config.training.anchor_loss_enabled:
                    step_loss_clean += float(clean["loss"].item()) * tokens
                    step_loss_noisy += float(noisy["loss"].item()) * tokens
                    step_loss_noisy_weighted += float((config.training.anchor_loss_lambda * noisy["loss"]).item()) * tokens
                step_tokens += tokens

            if (
                config.training.isotropy_loss_enabled
                and config.training.isotropy_loss_pooled_target
                and iso_sum is not None
                and iso_count >= 2
            ):
                # Fold this optimizer step's pooled moments into the EMA target. Same
                # population normalization as the logging readout below.
                step_pooled_mean = iso_sum / iso_count
                step_pooled_cov = iso_sum_outer / iso_count - torch.outer(step_pooled_mean, step_pooled_mean)
                beta = config.training.isotropy_loss_pooled_beta
                if iso_ema_cov is None:
                    iso_ema_mean, iso_ema_cov = step_pooled_mean, step_pooled_cov
                else:
                    iso_ema_mean = (1.0 - beta) * iso_ema_mean + beta * step_pooled_mean
                    iso_ema_cov = (1.0 - beta) * iso_ema_cov + beta * step_pooled_cov

            # Joint clip over model+head when the head exists (optimizer_model is model
            # flag-off): MasterWeightAdamW copies grads into the masters AFTER this call,
            # so a head clipped separately -- or not at all -- would enter the masters
            # unclipped. Note train/grad_norm therefore includes the head when enabled.
            grad_norm = clip_grad_norm_(optimizer_model.parameters(), config.training.grad_clip_norm)
            optimizer.step()
            elapsed = max(1e-9, time.perf_counter() - started_at)
            result.final_step = step

            if step == 1 or step % config.training.log_every_steps == 0:
                log_payload = {
                    "train/loss": step_loss / max(1, step_tokens),
                    "train/lr": step_lr,
                    "train/schedule_step": float(schedule_step),
                    "train/global_step": float(step),
                    "train/grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
                    "train/tokens_per_second": step_tokens / elapsed,
                }
                if config.training.uniform_loss_enabled:
                    log_payload.update(
                        {
                            "train/loss_uniform": step_loss_uniform / config.training.gradient_accumulation_steps,
                            "train/loss_uniform_weighted": step_loss_uniform_weighted
                            / config.training.gradient_accumulation_steps,
                            "latents/cosine_mean": step_cosine_mean / max(1, step_cosine_count),
                        }
                    )
                if config.training.isotropy_loss_enabled:
                    log_payload.update(
                        {
                            "train/loss_iso": step_loss_iso / config.training.gradient_accumulation_steps,
                            "train/loss_iso_weighted": step_loss_iso_weighted
                            / config.training.gradient_accumulation_steps,
                        }
                    )
                    if config.training.isotropy_loss_pooled_target and iso_ema_cov is not None:
                        # The actual objective in pooled mode (train/loss_iso is only a
                        # gradient carrier there and can go negative).
                        ema_identity = torch.eye(
                            iso_ema_cov.size(0), device=iso_ema_cov.device, dtype=iso_ema_cov.dtype
                        )
                        log_payload["train/loss_iso_pooled_objective"] = float(
                            ((iso_ema_cov - ema_identity).pow(2).mean() + iso_ema_mean.pow(2).mean()).item()
                        )
                    if iso_sum is not None and iso_count >= 2:
                        # Pooled over the whole optimizer step (N ~ accum * words/batch):
                        # the eigenvalue readout that steers lambda against the
                        # [0.81, 1.19] acceptance band (notes/open/rung3-*.md).
                        pooled_mean = iso_sum / iso_count
                        pooled_cov = iso_sum_outer / iso_count - torch.outer(pooled_mean, pooled_mean)
                        eigenvalues = torch.linalg.eigvalsh(pooled_cov.double())
                        log_payload.update(
                            {
                                "latents/cov_eig_min": float(eigenvalues.min()),
                                "latents/cov_eig_max": float(eigenvalues.max()),
                                "latents/mean_norm": float(pooled_mean.norm()),
                            }
                        )
                if config.training.anchor_loss_enabled:
                    log_payload.update(
                        {
                            "train/loss_clean_ce": step_loss_clean / step_tokens,
                            "train/loss_noisy_ce": step_loss_noisy / step_tokens,
                            "train/loss_noisy_weighted": step_loss_noisy_weighted / step_tokens,
                            "train/noisy_clean_gap": (step_loss_noisy - step_loss_clean) / step_tokens,
                            "train/anchor_lambda": config.training.anchor_loss_lambda,
                            "train/anchor_sigma": config.training.anchor_noise_sigma,
                        }
                    )
                if config.training.predictive_loss_enabled:
                    accum = config.training.gradient_accumulation_steps
                    log_payload.update(
                        {
                            "train/loss_pred": step_loss_pred / accum,
                            "train/loss_pred_weighted": step_loss_pred_weighted / accum,
                            "pred/num_queries": step_pred_queries / accum,
                        }
                    )
                    if step_pred_queries > 0:
                        log_payload["pred/top1_acc"] = step_pred_correct / step_pred_queries
                        # InfoNCE MI lower bound in nats: E[log N] - loss (Oord et al.
                        # 2018; capped at log N, a slight underestimate under
                        # repeated-word false negatives).
                        log_payload["pred/mi_lower_bound"] = (
                            step_pred_log_candidates / max(1, step_pred_batches)
                            - step_loss_pred / accum
                        )
                logger.log(log_payload, step=step)

            if step % config.training.validate_every_steps == 0 or step == config.training.max_steps:
                validation_losses = evaluate_per_slice(
                    model,
                    validation_loaders,
                    device,
                    config.training.validation_batches_per_slice,
                )
                result.validation_losses = validation_losses
                logger.log({f"validation/{name}_ce": value for name, value in validation_losses.items()}, step=step)

                shuffle_metrics = latent_shuffle_check(model, next(latent_batches), device)
                result.latent_shuffle = shuffle_metrics
                logger.log({f"latent_shuffle/{name}": value for name, value in shuffle_metrics.items()}, step=step)

            if step % config.training.checkpoint_every_steps == 0 or step == config.training.max_steps:
                checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    extra={
                        "config": dataclasses.asdict(config),
                        "trainable_parameters": trainable_parameters,
                        "data": data_info,
                        "validation_losses": result.validation_losses,
                        "latent_shuffle": result.latent_shuffle,
                        # Sidecar head state (never in model.state_dict()); its OPTIMIZER
                        # state needs no extra slot -- head params live in the shared
                        # optimizer at stable ModuleDict positions.
                        **(
                            {"predictive_head_state": predictive_head.state_dict()}
                            if predictive_head is not None
                            else {}
                        ),
                    },
                )
                result.checkpoint_paths.append(str(checkpoint_path))
    finally:
        logger.close()

    return result
