"""Synthetic checks for pipeline/analyses/explained_variance_analysis.py.

Run standalone (any env with numpy + pandas):
    python latent_evaluation/test/explained_variance_t.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.analyses.explained_variance_analysis import analyze_explained_variance


def aggregate(embedding, method="mean"):
    if embedding is None:
        return None
    if embedding.ndim == 1:
        return embedding
    if embedding.ndim == 2:
        return embedding.mean(axis=0)
    raise ValueError(f"Unsupported ndim {embedding.ndim}")


def make_df(vectors):
    return pd.DataFrame({"patch_embeddings_array": list(vectors)})


def main():
    rng = np.random.default_rng(0)

    # --- planted spectrum: EV at k must match the planted eigenvalues ---
    n, d = 20000, 16
    lam = np.array([8.0, 4.0, 2.0] + [1.0] * (d - 4) + [0.25])
    X = (rng.standard_normal((n, d)) * np.sqrt(lam)).astype(np.float32)

    r = analyze_explained_variance(
        make_df(X), aggregate_embeddings_fn=aggregate, target_dims=[1, 3, 16]
    )
    lam_sorted = np.sort(lam)[::-1]
    expected = np.cumsum(lam_sorted) / lam.sum()
    assert r["n_samples"] == n and r["embedding_dim"] == d
    assert r["kept_dims"] == [1, 3, 16]
    assert r["discarded_dims"] == []
    for k in r["kept_dims"]:
        assert abs(r["explained_variance_at"][k] - expected[k - 1]) < 0.02, k
    assert abs(r["explained_variance_at"][16] - 1.0) < 1e-6  # full dim explains all
    assert len(r["cumulative_explained_variance"]) == d
    print("planted spectrum: OK")

    # --- discard partition: too-large and invalid dims are dropped with reasons ---
    r_disc = analyze_explained_variance(
        make_df(X), aggregate_embeddings_fn=aggregate, target_dims=[4, 32, 0, -2, 2.5, 8]
    )
    assert r_disc["kept_dims"] == [4, 8]
    assert r_disc["requested_dims"] == [4, 32, 0, -2, 2.5, 8]
    discarded = dict(r_disc["discarded_dims"])
    assert set(discarded) == {32, 0, -2, 2.5}
    assert "exceeds" in discarded[32]
    assert "positive integer" in discarded[0]
    print("discard partition: OK")

    # --- default target dims: powers of 2 up to the embedding dim ---
    r_def = analyze_explained_variance(make_df(X), aggregate_embeddings_fn=aggregate)
    assert r_def["kept_dims"] == [2, 4, 8, 16]
    print("default dims: OK")

    # --- non-power-of-2 dim is appended to the default ladder ---
    X12 = rng.standard_normal((500, 12)).astype(np.float32)
    r12 = analyze_explained_variance(make_df(X12), aggregate_embeddings_fn=aggregate)
    assert r12["kept_dims"] == [2, 4, 8, 12]
    print("non-power-of-2 default: OK")

    # --- degenerate inputs ---
    r_empty = analyze_explained_variance(make_df([]), aggregate_embeddings_fn=aggregate)
    assert r_empty["kept_dims"] == [] and len(r_empty["cumulative_explained_variance"]) == 0
    r_const = analyze_explained_variance(
        make_df([np.ones(4, dtype=np.float32)] * 5), aggregate_embeddings_fn=aggregate
    )
    assert r_const["kept_dims"] == []  # zero variance -> nothing to explain
    print("degenerate inputs: OK")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
