import numpy as np
import pandas as pd

from pipeline.analyses.covariance_analysis import analyze_covariance


def _default_target_dims(embedding_dim):
    # Powers of 2 up to the embedding dim, plus the dim itself so the curve
    # endpoint (100% explained) always gets a marker.
    dims = []
    k = 2
    while k <= embedding_dim:
        dims.append(k)
        k *= 2
    if embedding_dim not in dims:
        dims.append(embedding_dim)
    return dims


def _partition_target_dims(target_dims, embedding_dim):
    kept, discarded = [], []
    for dim in target_dims:
        if not float(dim).is_integer() or dim < 1:
            discarded.append((dim, "not a positive integer"))
        elif dim > embedding_dim:
            discarded.append((dim, f"exceeds embedding dimension {embedding_dim}"))
        else:
            kept.append(int(dim))
    return sorted(set(kept)), discarded


def _degenerate_result(covariance_result, requested_dims):
    return {
        "n_samples": covariance_result["n_samples"],
        "embedding_dim": covariance_result["embedding_dim"],
        "zero_norm_count": covariance_result["zero_norm_count"],
        "requested_dims": list(requested_dims) if requested_dims is not None else [],
        "kept_dims": [],
        "discarded_dims": [],
        "cumulative_explained_variance": np.array([]),
        "explained_variance_at": {},
    }


def analyze_explained_variance(
    df: pd.DataFrame,
    *,
    aggregate_embeddings_fn,
    embedding_col: str = "patch_embeddings_array",
    aggregation_method: str = "mean",
    target_dims: list = None,
) -> dict:
    """
    Cumulative explained variance (standard PCA: eigenspectrum of the
    mean-centered covariance) evaluated at a list of target dimensionalities.
    Target dims larger than the embedding dim (or otherwise invalid) are
    discarded and reported with a reason.

    Population-blind: operates on whatever rows the caller provides.
    """
    # The covariance analysis already computes the full eigenspectrum.
    covariance = analyze_covariance(
        df,
        aggregate_embeddings_fn=aggregate_embeddings_fn,
        embedding_col=embedding_col,
        aggregation_method=aggregation_method,
        compute_t_window=False,
    )

    eigenvalues = covariance["eigenvalues_raw"]
    trace = float(eigenvalues.sum()) if len(eigenvalues) else 0.0
    if trace == 0.0:
        return _degenerate_result(covariance, target_dims)

    embedding_dim = covariance["embedding_dim"]
    requested = list(target_dims) if target_dims is not None else _default_target_dims(embedding_dim)
    kept, discarded = _partition_target_dims(requested, embedding_dim)

    cumulative = np.cumsum(eigenvalues) / trace
    # cumulative[k-1] = fraction of variance explained by the top k dimensions
    explained_at = {k: float(cumulative[k - 1]) for k in kept}

    return {
        "n_samples": covariance["n_samples"],
        "embedding_dim": embedding_dim,
        "zero_norm_count": covariance["zero_norm_count"],
        "requested_dims": requested,
        "kept_dims": kept,
        "discarded_dims": discarded,
        "cumulative_explained_variance": cumulative,
        "explained_variance_at": explained_at,
    }
