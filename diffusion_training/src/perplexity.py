import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from src.generation import GenerationSpec, generate_texts

# Loading gpt2-large is ~3GB and several seconds; the training hook must pay that once per
# process, not once per eval. Keyed by (model_id, device) because the same evaluator may
# legitimately be materialised on two devices in one process.
_EVALUATOR_CACHE: dict[tuple[str, str], "PerplexityEvaluator"] = {}


@dataclass
class PerplexityEvaluator:

    model_id: str
    tokenizer: object
    model: torch.nn.Module
    device: torch.device
    # Hard ceiling from the LM's learned position embeddings (1024 for the GPT-2 family).
    # Scoring past it is an indexing error, not a graceful degradation, so max_eval_tokens
    # is clamped to it rather than trusted.
    max_positions: int = 1024


def load_evaluator(
    model_id: str = "gpt2-large",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    local_files_only: bool = False,
) -> PerplexityEvaluator:

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(device)
    key = (model_id, str(device))
    cached = _EVALUATOR_CACHE.get(key)
    if cached is not None:
        return cached

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"perplexity.evaluator_device={str(device)!r} but CUDA is not available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"perplexity.evaluator_device={str(device)!r} does not exist; this box has "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=local_files_only, torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    max_positions = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", 1024
    )
    evaluator = PerplexityEvaluator(
        model_id=model_id, tokenizer=tokenizer, model=model, device=device, max_positions=int(max_positions)
    )
    _EVALUATOR_CACHE[key] = evaluator
    return evaluator


@torch.no_grad()
def score_perplexity(
    evaluator: PerplexityEvaluator,
    texts: list[str],
    prompts: list[str] | None = None,
    max_tokens: int = 512,
    return_per_sample: bool = False,
):

    if prompts is not None and len(prompts) != len(texts):
        raise ValueError(f"prompts has length {len(prompts)}, expected {len(texts)} to match texts.")

    total_nll = 0.0
    total_tokens = 0
    scored_texts = 0
    per_sample: list[dict[str, float]] = []
    limit = min(max_tokens, evaluator.max_positions)

    for index, text in enumerate(texts):
        record = {"ppl": float("nan"), "nll": float("nan"), "num_tokens": 0.0}
        per_sample.append(record)
        if not text:
            continue
        
        # ids of all tokens in text
        ids = evaluator.tokenizer(text, return_tensors="pt").input_ids[:, :limit]

        # Offset of the first scored token, everything before it is context only.
        # Only applies if we score perplexity with a given prompt
        offset = 0
        if prompts is not None:
            prompt_ids = evaluator.tokenizer(prompts[index], return_tensors="pt").input_ids
            offset = min(prompt_ids.size(1), ids.size(1))

        # Need at least one token to predict from at least one token of context.
        if ids.size(1) - max(offset, 1) < 1:
            continue

        ids = ids.to(evaluator.device)
        logits = evaluator.model(ids).logits.float()
        # Standard causal shift: logits at position i predict the token at position i+1.
        target = ids[:, 1:]
        predictions = logits[:, :-1]
        # Drop the prompt region; offset-1 because predictions are already shifted by one.
        start = max(offset - 1, 0)
        target = target[:, start:]
        predictions = predictions[:, start:]
        if target.numel() == 0:
            continue

        nll = torch.nn.functional.cross_entropy(
            predictions.reshape(-1, predictions.size(-1)), target.reshape(-1), reduction="sum"
        )
        sample_nll = float(nll.item())
        sample_tokens = int(target.numel())
        record["nll"] = sample_nll / sample_tokens
        record["ppl"] = math.exp(record["nll"])
        record["num_tokens"] = float(sample_tokens)
        total_nll += sample_nll
        total_tokens += sample_tokens
        scored_texts += 1

    if total_tokens == 0:
        # Everything was empty or too short. Report it rather than dividing by zero: a
        # silently shrinking sample set is exactly the failure this metric must expose.
        metrics = {
            "ppl": float("nan"),
            "nll": float("nan"),
            "total_nll": 0.0,
            "mean_nll": float("nan"),
            "mean_ppl": float("nan"),
            "median_nll": float("nan"),
            "median_ppl": float("nan"),
            "num_texts": 0.0,
            "num_tokens": 0.0,
        }
        return (metrics, per_sample) if return_per_sample else metrics

    mean_nll = total_nll / total_tokens
    # Equal-weight per-sample statistics over the scored texts only (NaN rows excluded);
    # see the docstring for how they relate to the token-weighted pair.
    sample_nlls = [record["nll"] for record in per_sample if record["num_tokens"] > 0]
    sample_ppls = [record["ppl"] for record in per_sample if record["num_tokens"] > 0]
    metrics = {
        "ppl": math.exp(mean_nll),
        "nll": mean_nll,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "mean_ppl": statistics.fmean(sample_ppls),
        "median_nll": statistics.median(sample_nlls),
        "median_ppl": statistics.median(sample_ppls),
        "num_texts": float(scored_texts),
        "num_tokens": float(total_tokens),
    }
    return (metrics, per_sample) if return_per_sample else metrics


def load_prompts(prompt_file: str | Path) -> list[str]:
    """One prompt per line; blank lines and ``#`` comments ignored."""

    path = Path(prompt_file)
    if not path.is_absolute() and not path.exists():
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / prompt_file
    if not path.exists():
        raise FileNotFoundError(f"perplexity.prompt_file not found: {prompt_file}")
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not prompts:
        raise ValueError(f"perplexity.prompt_file {path} contains no prompts.")
    return prompts


def build_spec(config) -> GenerationSpec:
    """PerplexityConfig -> GenerationSpec, resolving the diffusion.* fallbacks."""

    perplexity = config.perplexity
    return GenerationSpec(
        num_samples=1,  # one text per prompt; see run_perplexity_eval
        num_words=perplexity.num_words,
        num_bytes=perplexity.num_bytes,
        bytes_per_word=perplexity.bytes_per_word,
        sampling_steps=perplexity.sampling_steps or config.diffusion.sampling_steps,
        solver=perplexity.solver,
        decode_mode=perplexity.decode_mode,
        conditioning=perplexity.conditioning,
    )


def describe_spec(config, prompts: list[str] | None) -> str:
    """One-line record of what an eval will actually do, for the startup log."""

    perplexity = config.perplexity
    spec = build_spec(config)
    return (
        f"perplexity: mode={perplexity.mode} num_samples={perplexity.num_samples} "
        f"every_steps={perplexity.every_steps} seed={perplexity.seed} "
        f"num_words={spec.num_words or config.diffusion.max_words} "
        f"sampling_steps={spec.sampling_steps} solver={spec.solver} decode_mode={spec.decode_mode} "
        f"evaluator={perplexity.evaluator_model} on {perplexity.evaluator_device or config.training.device} "
        f"prompts={len(prompts) if prompts else 0}"
    )


@torch.no_grad()
def run_perplexity_eval(
    model,
    autoencoder,
    config,
    device: torch.device,
    dtype: torch.dtype,
    latent_stats: dict | None,
    evaluator: PerplexityEvaluator | None = None,
    length_predictor=None,
    predictor_space: str | None = None,
    prompts: list[str] | None = None,
    return_samples: bool = False,
):

    perplexity = config.perplexity
    if evaluator is None:
        evaluator = load_evaluator(
            perplexity.evaluator_model,
            perplexity.evaluator_device or config.training.device,
            local_files_only=perplexity.evaluator_local_files_only,
        )

    if perplexity.mode not in ("prompt", "unconditional"):
        raise ValueError(f"perplexity.mode must be 'prompt' or 'unconditional', got {perplexity.mode!r}.")
    if perplexity.mode == "prompt" and not prompts:
        raise ValueError("perplexity.mode='prompt' needs prompts; set perplexity.prompt_file.")

    spec = build_spec(config)
    # Fixed seed: a step-to-step PPL curve is only comparable if the starting noise is the
    # same at every evaluation, otherwise sampling variance swamps the training trend.
    generator_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(perplexity.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(perplexity.seed)

    was_training = model.training
    model.eval()
    try:
        samples = []
        for index in range(perplexity.num_samples):
            # One prompt per sample, cycling if there are fewer prompts than samples.
            # Generated one at a time because the byte decode is per-sample anyway (see
            # src/sampling.py generate_bytes) -- only the latent solve would batch.
            prompt_text = prompts[index % len(prompts)] if perplexity.mode == "prompt" else None
            samples.extend(
                generate_texts(
                    model,
                    autoencoder,
                    config,
                    spec,
                    device=device,
                    dtype=dtype,
                    latent_stats=latent_stats,
                    length_predictor=length_predictor,
                    predictor_space=predictor_space,
                    prompt_text=prompt_text,
                )
            )
    finally:
        if was_training:
            model.train()
        torch.random.set_rng_state(generator_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    texts = [sample.text for sample in samples]
    sample_prompts = [sample.prompt for sample in samples] if perplexity.mode == "prompt" else None
    scores, sample_scores = score_perplexity(
        evaluator, texts, sample_prompts, max_tokens=perplexity.max_eval_tokens, return_per_sample=True
    )

    # All three weightings, on purpose: the token-weighted corpus PPL is the stable
    # training curve, the per-sample mean is what much of the literature reports (and is
    # the one a single heavy-tailed sample can move), and the per-sample median is the
    # anti-gaming cross-check -- repetition floods that flatter both means barely move it.
    # scores["mean_nll"] is skipped: it is an exact alias of scores["nll"], and two
    # identical logger series help nobody.
    metrics = {
        "gen_ppl": scores["ppl"],
        "gen_ppl_sample_mean": scores["mean_ppl"],
        "gen_ppl_sample_median": scores["median_ppl"],
        "gen_nll": scores["nll"],
        "gen_nll_sample_median": scores["median_nll"],
        "gen_total_nll": scores["total_nll"],
        "num_texts": scores["num_texts"],
        "num_tokens": scores["num_tokens"],
    }
    return (metrics, samples, sample_scores) if return_samples else metrics


def score_baseline_texts(
    evaluator: PerplexityEvaluator,
    texts: list[str],
    max_bytes: int,
    max_tokens: int = 512,
) -> dict[str, float]:

    truncated = []
    for text in texts:
        encoded = text.encode("utf-8")[:max_bytes]
        decoded = encoded.decode("utf-8", errors="ignore").strip()
        if decoded:
            truncated.append(decoded)
    scores = score_perplexity(evaluator, truncated, max_tokens=max_tokens)
    return {
        "baseline_real_ppl": scores["ppl"],
        "baseline_num_texts": scores["num_texts"],
        "baseline_num_tokens": scores["num_tokens"],
    }
