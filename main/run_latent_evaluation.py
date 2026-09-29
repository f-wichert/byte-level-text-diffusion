"""Run the latent-space evaluation pipeline for a single encoder.

Invoke from the repo root, e.g.:
    python main/run_latent_evaluation.py -e tfree-hat -f all_tests.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LATENT_EVAL = REPO_ROOT / "latent_evaluation"
EVAL_CONFIGS = REPO_ROOT / "configs" / "evaluation"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the latent-space evaluation pipeline for one encoder."
    )
    p.add_argument("-e", "--encoder", required=True,
                   help="Encoder type, e.g. tfree-hat, bolmo, blt, neobert, gemini.")
    p.add_argument("-f", "--file", required=True,
                   help="Tasks config filename in configs/evaluation/, e.g. all_tests.json.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--default-subsample", type=int, default=5000)
    p.add_argument("--ae-config", default=None,
                   help="Training config for the tfree-hat-finetuned autoencoder. "
                        "Its checkpoint resolves to the latest step_*.pt in the config's "
                        "training.checkpoint_dir unless --ae-checkpoint is given. "
                        "Default: the pinned phase-2 setup.")
    p.add_argument("--ae-checkpoint", default=None,
                   help="Explicit autoencoder checkpoint path (overrides latest-resolution).")
    return p.parse_args()


def _resolve_cli_path(value):
    # Resolve against the invocation cwd BEFORE the chdir below, so relative
    # paths like configs/training/foo.yaml keep meaning what the caller typed.
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def main():
    args = parse_args()
    tasks_path = EVAL_CONFIGS / args.file  # resolved before chdir (absolute)
    ae_config = _resolve_cli_path(args.ae_config)
    ae_checkpoint = _resolve_cli_path(args.ae_checkpoint)

    # Make latent_evaluation importable AND the working dir, so `from pipeline.*`
    # and the pipeline's relative data/output paths resolve exactly as before.
    sys.path.insert(0, str(LATENT_EVAL))
    os.chdir(LATENT_EVAL)
    from pipeline.runner import run_analysis

    config = {
        "encoder_type": args.encoder,
        "device": args.device,
        "data_directory": args.data_dir,
        "output_directory": args.output_dir,
        "default_subsample": args.default_subsample,
        "default_decomposition": None,
        "ae_config": ae_config,
        "ae_checkpoint": ae_checkpoint,
    }
    config.update(json.loads(tasks_path.read_text()))
    run_analysis(config)


if __name__ == "__main__":
    main()
