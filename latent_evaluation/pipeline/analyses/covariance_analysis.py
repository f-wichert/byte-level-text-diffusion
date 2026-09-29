import numpy as np
import pandas as pd


def _degenerate_result(n_samples=0, embedding_dim=0, zero_norm_count=0, top_k=0):
    return {
        "n_samples": int(n_samples),
        "embedding_dim": int(embedding_dim),
        "zero_norm_count": int(zero_norm_count),
        "mean_norm": 0.0,
        "mean_embedding_norm": 0.0,
        "mean_norm_ratio": 0.0,
        "std_min": 0.0,
        "std_median": 0.0,
        "std_max": 0.0,
        "eigenvalues_raw": np.array([]),
        "eigenvalues_normalized": np.array([]),
        "lambda_max": 0.0,
        "lambda_median": 0.0,
        "lambda_min": 0.0,
        "lambda_max_over_median": 0.0,
        "lambda_min_reliable": False,
        "n_lt_dim": False,
        "top_k": int(top_k),
        "top_k_explained_variance": 0.0,
        "participation_ratio": 0.0,
        "t_window": None,
        "t_window_percentiles": (0.05, 0.95),
    }


def analyze_covariance(
    df: pd.DataFrame,
    *,
    aggregate_embeddings_fn,
    embedding_col: str = "patch_embeddings_array",
    aggregation_method: str = "mean",
    top_k: int = 32,
    compute_t_window: bool = True,
    lambda_min_reliability_factor: int = 4,
) -> dict:
    """
    Analyze the covariance spectrum (latent geometry) of the embedding
    population: mean-vector norm, per-dimension stds, eigenvalue spectrum,
    participation ratio and the cosine-schedule t-window in which the
    variance becomes resolvable (SNR=1 crossings).

    Population-blind: operates on whatever rows the caller provides.
    """
    vocabulary_df = df[df[embedding_col].notna()].copy().reset_index(drop=True)

    if vocabulary_df.empty:
        return _degenerate_result(top_k=0)

    vocabulary_df["embedding_vector"] = vocabulary_df[embedding_col].apply(
        lambda embedding: aggregate_embeddings_fn(embedding, method=aggregation_method)
    )
    vocabulary_df = vocabulary_df[
        vocabulary_df["embedding_vector"].notna()
    ].copy().reset_index(drop=True)

    if vocabulary_df.empty:
        return _degenerate_result(top_k=0)

    vectors = np.vstack(vocabulary_df["embedding_vector"].values).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    non_zero_mask = norms > 0.0
    zero_norm_count = int((~non_zero_mask).sum())

    vectors = vectors[non_zero_mask]
    norms = norms[non_zero_mask]

    n_samples = int(vectors.shape[0])
    embedding_dim = int(vectors.shape[1]) if vectors.ndim == 2 else 0
    if n_samples < 2:
        return _degenerate_result(n_samples, embedding_dim, zero_norm_count, top_k=0)

    mu = vectors.mean(axis=0, dtype=np.float64)
    mean_norm = float(np.linalg.norm(mu))
    mean_embedding_norm = float(norms.mean())
    mean_norm_ratio = mean_norm / mean_embedding_norm if mean_embedding_norm > 0 else 0.0

    # Chunked float64 accumulation: a full float64 copy of the matrix would be
    # several GB for the context populations (~175k x 4096).
    covariance = np.zeros((embedding_dim, embedding_dim), dtype=np.float64)
    block_size = 8192
    for start in range(0, n_samples, block_size):
        centered = vectors[start:start + block_size].astype(np.float64) - mu
        covariance += centered.T @ centered
    covariance /= n_samples - 1

    stds = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)[::-1]  # descending

    trace = float(eigenvalues.sum())
    if trace == 0.0:
        result = _degenerate_result(n_samples, embedding_dim, zero_norm_count, top_k=0)
        result.update({
            "mean_norm": mean_norm,
            "mean_embedding_norm": mean_embedding_norm,
            "mean_norm_ratio": mean_norm_ratio,
        })
        return result

    eigenvalues_normalized = eigenvalues / eigenvalues.mean()

    lambda_max = float(eigenvalues[0])
    lambda_median = float(np.median(eigenvalues))
    lambda_min = float(eigenvalues[-1])
    lambda_max_over_median = (
        lambda_max / lambda_median if lambda_median > 0 else float("inf")
    )

    top_k_effective = int(min(top_k, embedding_dim))
    top_k_explained_variance = float(eigenvalues[:top_k_effective].sum() / trace)
    participation_ratio = float(trace ** 2 / float((eigenvalues ** 2).sum()))

    t_window = None
    if compute_t_window:
        # A direction with data std s reaches SNR=1 at t = (2/pi)*arctan(1/s)
        # on the cosine path; arctan2 handles s == 0 (t = 1, the correct limit).
        s = np.sqrt(eigenvalues_normalized)
        t = (2.0 / np.pi) * np.arctan2(1.0, s)
        cumulative = np.cumsum(eigenvalues) / trace
        i_low = min(int(np.searchsorted(cumulative, 0.05)), embedding_dim - 1)
        i_high = min(int(np.searchsorted(cumulative, 0.95)), embedding_dim - 1)
        t_window = (float(t[i_low]), float(t[i_high]))

    return {
        "n_samples": n_samples,
        "embedding_dim": embedding_dim,
        "zero_norm_count": zero_norm_count,
        "mean_norm": mean_norm,
        "mean_embedding_norm": mean_embedding_norm,
        "mean_norm_ratio": mean_norm_ratio,
        "std_min": float(stds.min()),
        "std_median": float(np.median(stds)),
        "std_max": float(stds.max()),
        "eigenvalues_raw": eigenvalues,
        "eigenvalues_normalized": eigenvalues_normalized,
        "lambda_max": lambda_max,
        "lambda_median": lambda_median,
        "lambda_min": lambda_min,
        "lambda_max_over_median": lambda_max_over_median,
        "lambda_min_reliable": n_samples >= lambda_min_reliability_factor * embedding_dim,
        "n_lt_dim": n_samples <= embedding_dim,
        "top_k": top_k_effective,
        "top_k_explained_variance": top_k_explained_variance,
        "participation_ratio": participation_ratio,
        "t_window": t_window,
        "t_window_percentiles": (0.05, 0.95),
    }
