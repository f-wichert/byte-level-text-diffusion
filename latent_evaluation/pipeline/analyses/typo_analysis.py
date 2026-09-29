import numpy as np
import pandas as pd


def analyze_typo_robustness(
    df: pd.DataFrame,
    *,
    encode_word_fn,
    aggregate_embeddings_fn,
    prepare_typo_evaluation_space_fn=None,
    embedding_col: str = "patch_embeddings_array",
    label_col: str = "Word",
    aggregation_method: str = "mean",
    max_words: int | None = 500,
    min_word_length: int = 4,
    random_state: int = 42,
) -> dict:
    """
    Simple typo robustness evaluation:
    1. Generate vocabulary from dataframe
    2. Generate typos
    3. Encode typos
    4. Report Recall@1, Recall@5, and MRR
    5. Return data
    """
    # 1. Generate vocabulary from dataframe
    vocabulary_df = df[
        df[label_col].notna() & df[embedding_col].notna()
    ].copy().reset_index(drop=True)
    vocabulary_df[label_col] = vocabulary_df[label_col].astype(str).str.strip()
    vocabulary_df = vocabulary_df[
        vocabulary_df[label_col] != ""
    ].drop_duplicates(subset=[label_col], keep="first").reset_index(drop=True)
    vocabulary_df = vocabulary_df[
        vocabulary_df[label_col].str.len() >= min_word_length
    ].copy().reset_index(drop=True)

    if max_words is not None and len(vocabulary_df) > max_words:
        vocabulary_df = vocabulary_df.sample(
            n=max_words,
            random_state=random_state,
        ).reset_index(drop=True)

    if vocabulary_df.empty:
        return {
            "summary": {"recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0},
            "details": pd.DataFrame(),
            "typo_pairs": pd.DataFrame(),
            "vocabulary": vocabulary_df,
        }

    vocabulary_df["embedding_vector"] = vocabulary_df[embedding_col].apply(
        lambda embedding: aggregate_embeddings_fn(embedding, method=aggregation_method)
    )
    vocabulary_df = vocabulary_df[
        vocabulary_df["embedding_vector"].notna()
    ].copy().reset_index(drop=True)

    if vocabulary_df.empty:
        return {
            "summary": {"recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0},
            "details": pd.DataFrame(),
            "typo_pairs": pd.DataFrame(),
            "vocabulary": vocabulary_df,
        }

    clean_words = vocabulary_df[label_col].tolist()

    # 2. Generate typos
    rng = np.random.default_rng(random_state)
    typo_rows = []
    for clean_word in clean_words:
        for error_type, typo_word in _generate_typos(clean_word, rng).items():
            if typo_word and typo_word != clean_word:
                typo_rows.append(
                    {
                        "clean_word": clean_word,
                        "typo_word": typo_word,
                        "error_type": error_type,
                    }
                )

    typo_pairs_df = pd.DataFrame(typo_rows).drop_duplicates(
        subset=["clean_word", "typo_word"],
        keep="first",
    )

    if typo_pairs_df.empty:
        return {
            "summary": {"recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0},
            "details": pd.DataFrame(),
            "typo_pairs": typo_pairs_df,
            "vocabulary": vocabulary_df.drop(columns=["embedding_vector"]),
        }

    typo_vector_lookup = None
    if prepare_typo_evaluation_space_fn is not None:
        prepared = prepare_typo_evaluation_space_fn(vocabulary_df, typo_pairs_df)
        vocabulary_df = prepared["vocabulary_df"].copy().reset_index(drop=True)
        typo_vector_lookup = dict(prepared["typo_vector_lookup"])

    clean_words = vocabulary_df[label_col].tolist()
    clean_vectors = np.vstack(vocabulary_df["embedding_vector"].values).astype(np.float32)
    clean_vectors = _normalize_rows(clean_vectors)
    clean_index = {word: idx for idx, word in enumerate(clean_words)}

    # 3. Encode typos
    detail_rows = []
    for row in typo_pairs_df.itertuples(index=False):
        if typo_vector_lookup is not None:
            typo_vector = typo_vector_lookup.get(row.typo_word)
            if typo_vector is None:
                continue
            typo_vector = np.asarray(typo_vector, dtype=np.float32)
        else:
            typo_embedding = encode_word_fn(row.typo_word)
            if typo_embedding is None:
                continue

            typo_vector = aggregate_embeddings_fn(typo_embedding, method=aggregation_method)
            if typo_vector is None:
                continue

            typo_vector = np.asarray(typo_vector, dtype=np.float32)

        if typo_vector.ndim != 1 or typo_vector.shape[0] != clean_vectors.shape[1]:
            continue

        typo_vector = _normalize_rows(np.expand_dims(typo_vector, axis=0))[0]
        similarities = clean_vectors @ typo_vector

        ranked_indices = np.argsort(similarities)[::-1]
        target_rank = int(np.where(ranked_indices == clean_index[row.clean_word])[0][0]) + 1

        detail_rows.append(
            {
                "clean_word": row.clean_word,
                "typo_word": row.typo_word,
                "error_type": row.error_type,
                "target_rank": target_rank,
                "recall_at_1": int(target_rank == 1),
                "recall_at_5": int(target_rank <= 5),
                "mrr": 1.0 / target_rank,
                "top_5_words": [clean_words[idx] for idx in ranked_indices[:5]],
            }
        )

    details_df = pd.DataFrame(detail_rows)

    # 4. Report Recall@1, Recall@5 and MRR
    if details_df.empty:
        summary = {"recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0}
    else:
        summary = {
            "recall_at_1": float(details_df["recall_at_1"].mean()),
            "recall_at_5": float(details_df["recall_at_5"].mean()),
            "mrr": float(details_df["mrr"].mean()),
        }

    # 5. Return data
    return {
        "summary": summary,
        "details": details_df,
        "typo_pairs": typo_pairs_df,
        "vocabulary": vocabulary_df.drop(columns=["embedding_vector"]),
    }


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _generate_typos(word: str, rng) -> dict:
    return {
        "deletion": _delete_char(word, rng),
        "insertion": _insert_char(word, rng),
        "substitution": _replace_char(word, rng),
        "transposition": _swap_chars(word, rng),
    }


def _delete_char(word: str, rng) -> str | None:
    if len(word) < 2:
        return None
    index = int(rng.integers(0, len(word)))
    return word[:index] + word[index + 1 :]


def _insert_char(word: str, rng) -> str | None:
    if not word:
        return None
    index = int(rng.integers(0, len(word) + 1))
    char_to_insert = word[index - 1] if index > 0 else word[0]
    return word[:index] + char_to_insert + word[index:]


def _replace_char(word: str, rng) -> str | None:
    if not word:
        return None
    index = int(rng.integers(0, len(word)))
    replacement = "x" if word[index].lower() != "x" else "z"
    if word[index].isupper():
        replacement = replacement.upper()
    return word[:index] + replacement + word[index + 1 :]


def _swap_chars(word: str, rng) -> str | None:
    if len(word) < 2:
        return None
    index = int(rng.integers(0, len(word) - 1))
    return word[:index] + word[index + 1] + word[index] + word[index + 2 :]
