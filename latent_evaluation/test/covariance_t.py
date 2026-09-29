"""Synthetic-spectrum checks for pipeline/analyses/covariance_analysis.py.

Run standalone (any env with numpy + pandas):
    python latent_evaluation/test/covariance_t.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.analyses.covariance_analysis import analyze_covariance


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

    # --- planted spectrum recovery ---
    n, d = 20000, 16
    lam = np.array([8.0, 4.0, 2.0] + [1.0] * (d - 4) + [0.25])
    offset = np.full(d, 0.5)
    X = (rng.standard_normal((n, d)) * np.sqrt(lam) + offset).astype(np.float32)

    r = analyze_covariance(make_df(X), aggregate_embeddings_fn=aggregate, top_k=3)

    lam_sorted = np.sort(lam)[::-1]
    assert r["n_samples"] == n and r["embedding_dim"] == d
    assert r["zero_norm_count"] == 0
    assert np.allclose(r["eigenvalues_raw"], lam_sorted, rtol=0.08), r["eigenvalues_raw"]
    assert abs(r["eigenvalues_normalized"].mean() - 1.0) < 1e-9

    pr_expected = lam.sum() ** 2 / (lam ** 2).sum()
    assert abs(r["participation_ratio"] - pr_expected) / pr_expected < 0.05

    assert r["top_k"] == 3
    tk_expected = lam_sorted[:3].sum() / lam.sum()
    assert abs(r["top_k_explained_variance"] - tk_expected) < 0.02

    ratio_expected = np.linalg.norm(offset) / np.sqrt(np.linalg.norm(offset) ** 2 + lam.sum())
    assert abs(r["mean_norm_ratio"] - ratio_expected) < 0.05, (r["mean_norm_ratio"], ratio_expected)

    assert r["t_window"] is not None and r["t_window"][0] < r["t_window"][1]
    assert r["lambda_min_reliable"] is True and bool(r["n_lt_dim"]) is False
    print("planted spectrum: OK")

    # --- isotropic spectrum: every direction crosses SNR=1 at t = 0.5 ---
    X_iso = rng.standard_normal((5000, 8)).astype(np.float32)
    r_iso = analyze_covariance(make_df(X_iso), aggregate_embeddings_fn=aggregate)
    assert abs(r_iso["t_window"][0] - 0.5) < 0.02 and abs(r_iso["t_window"][1] - 0.5) < 0.02
    assert abs(r_iso["participation_ratio"] - 8) < 0.5
    print("isotropic spectrum: OK")

    # --- None and zero-norm rows are dropped and counted ---
    rows = list(rng.standard_normal((10, 4)).astype(np.float32))
    rows.insert(2, None)
    rows.insert(5, np.zeros(4, dtype=np.float32))
    r_drop = analyze_covariance(make_df(rows), aggregate_embeddings_fn=aggregate)
    assert r_drop["n_samples"] == 10
    assert r_drop["zero_norm_count"] == 1
    print("row filtering: OK")

    # --- degenerate inputs ---
    r_single = analyze_covariance(
        make_df([np.ones(4, dtype=np.float32)]), aggregate_embeddings_fn=aggregate
    )
    assert r_single["n_samples"] == 1 and r_single["t_window"] is None
    r_empty = analyze_covariance(make_df([]), aggregate_embeddings_fn=aggregate)
    assert r_empty["n_samples"] == 0 and len(r_empty["eigenvalues_raw"]) == 0
    r_const = analyze_covariance(
        make_df([np.ones(4, dtype=np.float32)] * 5), aggregate_embeddings_fn=aggregate
    )
    assert r_const["participation_ratio"] == 0.0 and r_const["mean_norm"] > 0
    print("degenerate inputs: OK")

    # --- 2-D per-row embeddings go through the aggregation fn ---
    rows_2d = [rng.standard_normal((3, 6)).astype(np.float32) for _ in range(50)]
    r_2d = analyze_covariance(make_df(rows_2d), aggregate_embeddings_fn=aggregate)
    assert r_2d["n_samples"] == 50 and r_2d["embedding_dim"] == 6
    print("2-D aggregation: OK")

    # --- reliability flags flip with sample size ---
    X_small = rng.standard_normal((10, 16)).astype(np.float32)
    r_small = analyze_covariance(make_df(X_small), aggregate_embeddings_fn=aggregate)
    assert bool(r_small["n_lt_dim"]) is True and r_small["lambda_min_reliable"] is False
    print("reliability flags: OK")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
