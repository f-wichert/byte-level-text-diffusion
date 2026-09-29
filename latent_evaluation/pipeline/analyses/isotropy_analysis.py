import numpy as np
import pandas as pd


def analyze_isotropy(
    df: pd.DataFrame,
    *,
    aggregate_embeddings_fn,
    embedding_col: str = "patch_embeddings_array",
    aggregation_method: str = "mean",
) -> dict:
    """
    Compute the anisotropy score (ANI) as the mean off-diagonal cosine
    similarity across the embedding vocabulary.
    """
    vocabulary_df = df[df[embedding_col].notna()].copy().reset_index(drop=True)

    if vocabulary_df.empty:
        return {
            "ani_score": 0.0,
            "vocabulary_size": 0,
            "embedding_dim": 0,
            "zero_norm_count": 0,
        }

    vocabulary_df["embedding_vector"] = vocabulary_df[embedding_col].apply(
        lambda embedding: aggregate_embeddings_fn(embedding, method=aggregation_method)
    )
    vocabulary_df = vocabulary_df[
        vocabulary_df["embedding_vector"].notna()
    ].copy().reset_index(drop=True)

    if vocabulary_df.empty:
        return {
            "ani_score": 0.0,
            "vocabulary_size": 0,
            "embedding_dim": 0,
            "zero_norm_count": 0,
        }

    vectors = np.vstack(vocabulary_df["embedding_vector"].values).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    non_zero_mask = norms > 0.0
    zero_norm_count = int((~non_zero_mask).sum())

    vectors = vectors[non_zero_mask]
    if len(vectors) < 2:
        return {
            "ani_score": 0.0,
            "vocabulary_size": int(len(vectors)),
            "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
            "zero_norm_count": zero_norm_count,
        }

    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity_matrix = normalized_vectors @ normalized_vectors.T
    off_diagonal_sum = float(similarity_matrix.sum() - np.trace(similarity_matrix))
    vocabulary_size = int(len(normalized_vectors))
    ani_score = off_diagonal_sum / (vocabulary_size * (vocabulary_size - 1))

    return {
        "ani_score": float(ani_score),
        "vocabulary_size": vocabulary_size,
        "embedding_dim": int(normalized_vectors.shape[1]),
        "zero_norm_count": zero_norm_count,
    }
