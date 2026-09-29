#!/usr/bin/env bash
# set -e
# Run the latent evaluation for every encoder, switching conda envs as needed.
# The launcher handles cwd/sys.path, so this can be run from anywhere.

source ~/miniconda3/etc/profile.d/conda.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

conda activate bolmo;      python "$HERE/run_latent_evaluation.py" -e bolmo     -f all_tests.json
conda activate bolmo;      python "$HERE/run_latent_evaluation.py" -e neobert   -f all_tests.json
conda activate bolmo;      python "$HERE/run_latent_evaluation.py" -e gemini    -f all_tests.json
conda activate blt;        python "$HERE/run_latent_evaluation.py" -e blt       -f all_tests.json
conda activate t-free-hat; python "$HERE/run_latent_evaluation.py" -e tfree-hat -f all_tests.json
