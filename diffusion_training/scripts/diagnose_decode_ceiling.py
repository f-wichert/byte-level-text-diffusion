import argparse
import difflib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.text_metrics import DROP_FIRST_N_WORDS, format_metrics, load_vocabulary, score_samples
from src.diffusion_train import load_frozen_autoencoder
from src.sampling import decode_latents, encode_prompt, generate_bytes_splitter, resolve_latent_stats
from src.train import _dtype_from_name
from src.utils import load_checkpoint, load_diffusion_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--length-predictor", required=True)
    parser.add_argument(
        "--length-predictor-space", choices=["raw", "standardized"], default="raw",
        help="Latent space the predictor was trained on (raw for the compress256 artifact).",
    )
    parser.add_argument("--num-docs", type=int, default=50)
    parser.add_argument("--max-words", type=int, default=48, help="Cap words/doc to keep free-run decode fast.")
    parser.add_argument("--bytes-per-word", type=int, default=6, help="Fixed grid for condition D.")
    parser.add_argument("--seed-byte", type=int, default=32)
    parser.add_argument(
        "--splitter-max-bytes", type=int, default=None,
        help="Byte cap for condition E. Default: data.sequence_length, matching generate.py. The "
             "splitter decode normally self-terminates when a word past the last latent opens, so "
             "this only bites on pathological output.",
    )
    parser.add_argument(
        "--splitter-seed-words", type=int, default=1,
        help=f"Condition E: teacher-force the source bytes of the first N words instead of the bare "
             f"seed byte. Default 1 -- the splitter cannot open a second word while the prefix is "
             f"all whitespace, so a space seed stalls the decode. N <= {DROP_FIRST_N_WORDS} costs "
             f"nothing, since the scorers already drop the first {DROP_FIRST_N_WORDS} words. "
             f"0 restores the bare seed byte.",
    )
    parser.add_argument("--vocab", default="notes/phase0-artifacts/vocab.json")
    parser.add_argument("--results-dir", default="notes/phase0-artifacts")
    parser.add_argument("--device")
    # Optional reconstruction-PPL analysis. Off by default: it pulls in a ~3GB scoring LM and
    # the rest of the script must stay runnable without one. Every evaluator setting falls back
    # to the config's perplexity: block so the number is comparable to that run's gen_ppl curve.
    perplexity_group = parser.add_argument_group("perplexity (optional second analysis)")
    perplexity_group.add_argument(
        "--perplexity", action="store_true",
        help="Also score every condition's decoded text (and the real source) under the frozen LM.",
    )
    perplexity_group.add_argument(
        "--evaluator-model", default=None,
        help="Scoring LM. Default: config.perplexity.evaluator_model (gpt2-large).",
    )
    perplexity_group.add_argument(
        "--evaluator-device", default=None,
        help="Device for the scoring LM. Default: the decode device. Point it at a spare GPU to "
             "keep ~3GB of fp32 off the card holding the autoencoder.",
    )
    perplexity_group.add_argument(
        "--max-eval-tokens", type=int, default=None,
        help="Token cap per scored text. Default: config.perplexity.max_eval_tokens.",
    )
    perplexity_group.add_argument(
        "--evaluator-local-files-only", action="store_true", default=None,
        help="Force offline load. Default: config.perplexity.evaluator_local_files_only.",
    )
    return parser.parse_args()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _drop2(text: str) -> list[str]:
    return text.split()[DROP_FIRST_N_WORDS:]


def per_word_exact(source: str, decoded: str) -> tuple[int, int]:
    """(matched source words, total source words) via word-level alignment, first 2 dropped.

    Index alignment is unreliable here: the free-run decode's seeded first word can collapse
    to whitespace and vanish under split(), offsetting every later token by one even when the
    decode is otherwise near-verbatim. A difflib alignment (LCS of exact-equal tokens) measures
    the fraction of source words recovered in order, robust to such leading insert/delete.
    """

    s, d = _drop2(source), _drop2(decoded)
    matcher = difflib.SequenceMatcher(a=s, b=d, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched, len(s)


def norm_edit(source: str, decoded: str) -> float:
    s, d = " ".join(_drop2(source)), " ".join(_drop2(decoded))
    if not s and not d:
        return 0.0
    return levenshtein(s, d) / max(1, max(len(s), len(d)))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    device = torch.device(args.device or config.training.device)
    dtype = _dtype_from_name(config.training.dtype)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    autoencoder = load_frozen_autoencoder(config, device)
    # A checkpoint is only needed for latent_stats fallback; the frozen AE + config path
    # already resolve them. Pass an empty checkpoint dict so config fallback is used.
    latent_stats = resolve_latent_stats({}, config, device)

    from src.length_predictor import load_length_predictor

    length_predictor = load_length_predictor(args.length_predictor, device)
    # Artifact-recorded space (v2) wins; fall back to the CLI flag for space-less v1 artifacts.
    predictor_space = length_predictor.space or args.length_predictor_space
    vocab = load_vocabulary(args.vocab)

    # Resolved here, BEFORE the decode loop: free-run byte decode over num_docs x 4 conditions
    # takes minutes, and a typo'd evaluator device should fail now rather than after all of it.
    # Same reasoning as the training hook's step-0 load (src/perplexity.py::load_evaluator).
    evaluator = None
    max_eval_tokens = args.max_eval_tokens or config.perplexity.max_eval_tokens
    if args.perplexity:
        from src.perplexity import load_evaluator

        evaluator_model = args.evaluator_model or config.perplexity.evaluator_model
        evaluator_device = args.evaluator_device or str(device)
        local_files_only = (
            config.perplexity.evaluator_local_files_only
            if args.evaluator_local_files_only is None
            else args.evaluator_local_files_only
        )
        evaluator = load_evaluator(evaluator_model, evaluator_device, local_files_only=local_files_only)
        print(f"perplexity: evaluator={evaluator_model} on {evaluator_device} max_tokens={max_eval_tokens}")

    # Collect real docs' texts from the loader.
    from src.train import build_training_loader

    loader, _ = build_training_loader(config)
    texts: list[str] = []
    for batch in loader:
        texts.extend(batch["texts"])
        if len(texts) >= args.num_docs:
            break
    texts = texts[: args.num_docs]

    conditions = [
        "A_true_prompt", "B_true_freerun", "C_pred_freerun", "D_grid6_freerun", "E_splitter_freerun",
    ]
    splitter_max_bytes = args.splitter_max_bytes or config.data.sequence_length
    if args.splitter_seed_words > DROP_FIRST_N_WORDS:
        print(
            f"WARNING: --splitter-seed-words {args.splitter_seed_words} exceeds the "
            f"{DROP_FIRST_N_WORDS} words the scorers drop, so condition E is handed source text "
            "inside the scored region and its numbers are no longer comparable to B/C/D.",
            file=sys.stderr,
        )
    decoded: dict[str, list[str]] = {c: [] for c in conditions}
    sources: list[str] = []

    exact_num = {c: 0 for c in conditions}
    exact_den = {c: 0 for c in conditions}
    edit_sum = {c: 0.0 for c in conditions}

    n_used = 0
    for text in texts:
        prompt = encode_prompt(autoencoder, text, config.data.sequence_length, device)
        n_words = min(len(prompt.word_lengths), args.max_words)
        if n_words < DROP_FIRST_N_WORDS + 2:  # need words left after dropping 2
            continue
        true_lengths = [int(x) for x in prompt.word_lengths[:n_words]]
        n_true_bytes = sum(true_lengths)
        byte_ids = prompt.byte_ids[:n_true_bytes]
        z_raw = prompt.z_words[:, :n_words, :]  # native (raw) AE latents
        source_text = bytes(byte_ids).decode("utf-8", errors="replace")
        sources.append(source_text)
        n_used += 1

        # Length predictor on the space it was trained on (raw for compress256 artifacts).
        if predictor_space == "raw":
            z_pred = z_raw
        else:
            if latent_stats is None:
                raise SystemExit("standardized predictor space requested but no latent stats available")
            z_pred = (z_raw - latent_stats["mean"]) / latent_stats["std"]
        pred_lengths = length_predictor.predict_lengths(z_pred[0]).tolist()
        pred_lengths = [int(x) for x in pred_lengths[:n_words]]

        # Condition E's seed: the source bytes of the first N words (None = bare seed byte).
        splitter_seed_bytes = None
        if args.splitter_seed_words > 0:
            n_seed = min(args.splitter_seed_words, n_words)
            splitter_seed_bytes = byte_ids[: sum(true_lengths[:n_seed])] or None

        z_decode = z_raw.to(dtype=dtype)
        outputs = {
            "A_true_prompt": decode_latents(
                autoencoder, z_decode, num_bytes=n_true_bytes, bytes_per_word=args.bytes_per_word,
                seed_byte=args.seed_byte, device=device, word_lengths=true_lengths, prompt_byte_ids=byte_ids,
            ),
            "B_true_freerun": decode_latents(
                autoencoder, z_decode, num_bytes=n_true_bytes, bytes_per_word=args.bytes_per_word,
                seed_byte=args.seed_byte, device=device, word_lengths=true_lengths,
            ),
            "C_pred_freerun": decode_latents(
                autoencoder, z_decode, num_bytes=max(1, sum(pred_lengths)), bytes_per_word=args.bytes_per_word,
                seed_byte=args.seed_byte, device=device, word_lengths=pred_lengths,
            ),
            "D_grid6_freerun": decode_latents(
                autoencoder, z_decode, num_bytes=n_words * args.bytes_per_word, bytes_per_word=args.bytes_per_word,
                seed_byte=args.seed_byte, device=device, word_lengths=None,
            ),
            # No length table at all: the splitter re-segments the generated prefix after every
            # byte, so boundaries fall out of decoding. There is no decode_latents wrapper for
            # this path -- it returns raw byte ids.
            #
            # Seeded with the source bytes of the first --splitter-seed-words words rather than
            # the bare seed byte: _splitter_boundaries cannot close a word while the prefix is
            # all whitespace, so a space seed leaves the decoder cross-attending to latent 0 and
            # emitting spaces until it happens to produce a non-space. That stall is a property
            # of the seed, not of splitter decoding, and it distorts every raw-text metric. At
            # the default (1 word <= DROP_FIRST_N_WORDS) the seeded text is dropped by the
            # scorers, so this costs nothing in comparability against B/C/D.
            "E_splitter_freerun": bytes(
                generate_bytes_splitter(
                    autoencoder, z_decode, seed_byte=args.seed_byte, device=device,
                    max_bytes=splitter_max_bytes, prompt_byte_ids=splitter_seed_bytes,
                )
            ).decode("utf-8", errors="replace"),
        }
        for c, out in outputs.items():
            decoded[c].append(out)
            m, den = per_word_exact(source_text, out)
            exact_num[c] += m
            exact_den[c] += den
            edit_sum[c] += norm_edit(source_text, out)

    # Reconstruction PPL: same scorer, same settings, no prompt exclusion anywhere -- the source
    # line and every condition are scored identically, so the columns are directly comparable.
    #
    # Scored on the SAME normalized text word_exact/norm_edit judge (_drop2 + whitespace join),
    # not the raw decode. Two reasons, both load-bearing:
    #   - the free-run seed byte is a space, and splitter mode can stall on it for dozens of
    #     bytes; a long whitespace run is nearly free under GPT-2, which drags PPL toward 1 and
    #     makes the column measure the stall instead of the text (observed: E scored 2.45 raw).
    #   - dropping the first 2 words is already the convention here, because the seeded opening
    #     word is known-unreliable (README: "first 1-2 words of any sample are unreliable").
    # A is teacher-forced verbatim, so A == source survives this normalization -- keep that as
    # the wiring sanity check.
    #
    # These texts are truncated to max_words, so absolute numbers are NOT comparable to a
    # training run's perplexity/baseline_real_ppl (different byte budget, and PPL is strongly
    # length-sensitive below ~32 tokens); the source row here is the reference line.
    ppl: dict[str, dict[str, float]] = {}
    if evaluator is not None:
        from src.perplexity import score_perplexity

        def _for_ppl(texts: list[str]) -> list[str]:
            return [" ".join(_drop2(text)) for text in texts]

        ppl["source"] = score_perplexity(evaluator, _for_ppl(sources), max_tokens=max_eval_tokens)
        for c in conditions:
            ppl[c] = score_perplexity(evaluator, _for_ppl(decoded[c]), max_tokens=max_eval_tokens)

    # Report
    print(f"docs used: {n_used} (requested {args.num_docs}, max_words {args.max_words})")
    print(f"latent_stats available: {latent_stats is not None}  predictor_space: {predictor_space}  "
          f"splitter_max_bytes: {splitter_max_bytes}  splitter_seed_words: {args.splitter_seed_words}\n")
    ppl_header = f" {'gen_ppl':>9}" if evaluator is not None else ""
    header = (
        f"{'condition':<16} {'word_exact':>11} {'norm_edit':>10}{ppl_header} | "
        f"{'valid_word':>11} {'dist1':>7} {'dist2':>7} {'wlen_B':>7}"
    )
    print(header)
    print("-" * len(header))
    if evaluator is not None:
        # Reference line first: real text, unmodified, under the same scorer.
        print(f"{'source (real)':<16} {'-':>11} {'-':>10} {ppl['source']['ppl']:>9.2f} | ")
    rows = []
    for c in conditions:
        we = exact_num[c] / max(1, exact_den[c])
        ne = edit_sum[c] / max(1, n_used)
        m = score_samples(decoded[c], vocab)
        ppl_cell = f" {ppl[c]['ppl']:>9.2f}" if evaluator is not None else ""
        print(f"{c:<16} {we:>11.4f} {ne:>10.4f}{ppl_cell} | {m['valid_word_rate']:>11.4f} "
              f"{m['distinct_1']:>7.4f} {m['distinct_2']:>7.4f} {m['mean_word_len_bytes']:>7.2f}")
        row = {"condition": c, "word_exact": we, "norm_edit": ne, **m}
        if evaluator is not None:
            row["gen_ppl"] = ppl[c]["ppl"]
            row["gen_ppl_num_tokens"] = ppl[c]["num_tokens"]
        rows.append(row)

    # Both real decode paths side by side, since choosing between them is the point of E.
    print("\n=== 5 side-by-side examples (source vs the two real decode paths) ===")
    for i in range(min(5, n_used)):
        print(f"\n[{i}] SOURCE: {sources[i]}")
        print(f"    C len : {decoded['C_pred_freerun'][i]}")
        print(f"    E spl : {decoded['E_splitter_freerun'][i]}")

    out = {
        "config": args.config, "num_docs": n_used, "max_words": args.max_words,
        "predictor_space": args.length_predictor_space,
        "splitter_max_bytes": splitter_max_bytes, "splitter_seed_words": args.splitter_seed_words,
        "rows": rows, "sources": sources, "decoded": decoded,
    }
    if evaluator is not None:
        out["perplexity"] = {
            "evaluator_model": evaluator.model_id,
            "max_eval_tokens": max_eval_tokens,
            "source_gen_ppl": ppl["source"]["ppl"],
            "source_num_tokens": ppl["source"]["num_tokens"],
        }
    (results_dir / "task1_decode_ceiling.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {results_dir / 'task1_decode_ceiling.json'}")


if __name__ == "__main__":
    main()
