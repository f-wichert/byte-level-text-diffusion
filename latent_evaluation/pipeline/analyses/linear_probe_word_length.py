import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def analyze_linear_probe_word_length(
    df: pd.DataFrame,
    *,
    aggregate_embeddings_fn,
    embedding_col: str = "patch_embeddings_array",
    label_col: str = "Word",
    aggregation_method: str = "mean",
    word_lengths: list[int] | tuple[int, ...] | None = None,
    n_words_per_length: int = 100,
    random_state: int = 42,
    cv_folds: int = 5,
    max_iter: int = 1000,
) -> dict:
    """
    Run a balanced multiclass linear probe for predicting len(word).

    Workflow:
    1. Filter vocabulary to the requested word lengths
    2. Aggregate embeddings into one vector per word
    3. Sample the same number of words for each length
    4. Train/evaluate a logistic-regression probe with stratified CV
    5. Return metrics, confusion matrices, and per-word predictions
    """
    required_columns = {label_col, embedding_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    vocabulary_df = df[
        df[label_col].notna() & df[embedding_col].notna()
    ].copy().reset_index(drop=True)
    vocabulary_df[label_col] = vocabulary_df[label_col].astype(str).str.strip()
    vocabulary_df = vocabulary_df[
        vocabulary_df[label_col] != ""
    ].drop_duplicates(subset=[label_col], keep="first").reset_index(drop=True)
    vocabulary_df["word_length"] = vocabulary_df[label_col].apply(len)

    if word_lengths is None:
        requested_word_lengths = sorted(
            vocabulary_df["word_length"].dropna().astype(int).unique().tolist()
        )
    else:
        requested_word_lengths = sorted({int(length) for length in word_lengths})

    if not requested_word_lengths:
        return _empty_results()

    candidate_df = vocabulary_df[
        vocabulary_df["word_length"].isin(requested_word_lengths)
    ].copy().reset_index(drop=True)

    available_counts = (
        candidate_df["word_length"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    available_word_lengths = sorted(available_counts.keys())
    unavailable_word_lengths = [
        length for length in requested_word_lengths if length not in available_counts
    ]

    if len(available_word_lengths) < 2:
        results = _empty_results()
        results.update(
            {
                "requested_word_lengths": requested_word_lengths,
                "available_word_lengths": available_word_lengths,
                "unavailable_word_lengths": unavailable_word_lengths,
                "available_counts_by_length": available_counts,
            }
        )
        return results

    effective_n_words_per_length = min(
        int(n_words_per_length),
        min(int(count) for count in available_counts.values()),
    )
    if effective_n_words_per_length < 2:
        results = _empty_results()
        results.update(
            {
                "requested_word_lengths": requested_word_lengths,
                "available_word_lengths": available_word_lengths,
                "unavailable_word_lengths": unavailable_word_lengths,
                "available_counts_by_length": available_counts,
                "effective_n_words_per_length": effective_n_words_per_length,
            }
        )
        return results

    sampled_frames = []
    for offset, word_length in enumerate(available_word_lengths):
        class_df = candidate_df[
            candidate_df["word_length"] == word_length
        ].copy().reset_index(drop=True)
        sampled_frames.append(
            class_df.sample(
                n=effective_n_words_per_length,
                random_state=random_state + offset,
                replace=False,
            )
        )

    sampled_df = (
        pd.concat(sampled_frames, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    sampled_df["embedding_vector"] = sampled_df[embedding_col].apply(
        lambda embedding: aggregate_embeddings_fn(embedding, method=aggregation_method)
    )
    sampled_df = sampled_df[
        sampled_df["embedding_vector"].notna()
    ].copy().reset_index(drop=True)

    sampled_counts = sampled_df["word_length"].value_counts().sort_index()
    if len(sampled_counts) < 2:
        results = _empty_results()
        results.update(
            {
                "requested_word_lengths": requested_word_lengths,
                "available_word_lengths": available_word_lengths,
                "unavailable_word_lengths": unavailable_word_lengths,
                "available_counts_by_length": available_counts,
                "effective_n_words_per_length": effective_n_words_per_length,
                "sampled_counts_by_length": sampled_counts.to_dict(),
                "sampled_dataframe": sampled_df.drop(columns=["embedding_vector"]),
            }
        )
        return results

    X = np.vstack(sampled_df["embedding_vector"].values).astype(np.float32)
    y = sampled_df["word_length"].to_numpy(dtype=np.int64)

    class_counts = sampled_counts
    effective_cv_folds = min(int(cv_folds), int(class_counts.min()))
    if effective_cv_folds < 2:
        results = _empty_results()
        results.update(
            {
                "requested_word_lengths": requested_word_lengths,
                "available_word_lengths": available_word_lengths,
                "unavailable_word_lengths": unavailable_word_lengths,
                "available_counts_by_length": available_counts,
                "effective_n_words_per_length": effective_n_words_per_length,
                "sampled_counts_by_length": class_counts.to_dict(),
                "sampled_dataframe": sampled_df.drop(columns=["embedding_vector"]),
            }
        )
        return results

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
        ),
    )
    cv = StratifiedKFold(
        n_splits=effective_cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    accuracy_scores = cross_val_score(
        probe,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )
    balanced_accuracy_scores = cross_val_score(
        probe,
        X,
        y,
        cv=cv,
        scoring="balanced_accuracy",
    )
    y_pred = cross_val_predict(
        probe,
        X,
        y,
        cv=cv,
    )

    labels = np.array(available_word_lengths, dtype=np.int64)
    confusion_counts = confusion_matrix(y, y_pred, labels=labels)
    confusion_normalized = confusion_matrix(
        y,
        y_pred,
        labels=labels,
        normalize="true",
    )

    predictions_df = sampled_df[[label_col, "word_length"]].copy()
    predictions_df["predicted_length"] = y_pred
    predictions_df["is_correct"] = (
        predictions_df["word_length"] == predictions_df["predicted_length"]
    )
    predictions_df["absolute_error"] = (
        predictions_df["word_length"] - predictions_df["predicted_length"]
    ).abs()

    per_length_accuracy_df = (
        predictions_df.groupby("word_length", as_index=False)
        .agg(
            n_words=(label_col, "count"),
            accuracy=("is_correct", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
        )
        .sort_values("word_length")
        .reset_index(drop=True)
    )

    baseline_accuracy = 1.0 / len(available_word_lengths)

    return {
        "requested_word_lengths": requested_word_lengths,
        "available_word_lengths": available_word_lengths,
        "unavailable_word_lengths": unavailable_word_lengths,
        "available_counts_by_length": available_counts,
        "effective_n_words_per_length": int(effective_n_words_per_length),
        "sampled_counts_by_length": class_counts.to_dict(),
        "n_classes": int(len(available_word_lengths)),
        "n_samples": int(len(sampled_df)),
        "embedding_dim": int(X.shape[1]),
        "cv_folds": int(effective_cv_folds),
        "baseline_accuracy": float(baseline_accuracy),
        "accuracy_mean": float(np.mean(accuracy_scores)),
        "accuracy_std": float(np.std(accuracy_scores)),
        "balanced_accuracy_mean": float(np.mean(balanced_accuracy_scores)),
        "balanced_accuracy_std": float(np.std(balanced_accuracy_scores)),
        "overall_balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "confusion_matrix": confusion_counts,
        "confusion_matrix_normalized": confusion_normalized,
        "predictions": predictions_df,
        "per_length_accuracy": per_length_accuracy_df,
        "sampled_dataframe": sampled_df.drop(columns=["embedding_vector"]),
    }


def _empty_results() -> dict:
    return {
        "requested_word_lengths": [],
        "available_word_lengths": [],
        "unavailable_word_lengths": [],
        "available_counts_by_length": {},
        "effective_n_words_per_length": 0,
        "sampled_counts_by_length": {},
        "n_classes": 0,
        "n_samples": 0,
        "embedding_dim": 0,
        "cv_folds": 0,
        "baseline_accuracy": 0.0,
        "accuracy_mean": 0.0,
        "accuracy_std": 0.0,
        "balanced_accuracy_mean": 0.0,
        "balanced_accuracy_std": 0.0,
        "overall_balanced_accuracy": 0.0,
        "macro_f1": 0.0,
        "confusion_matrix": np.zeros((0, 0), dtype=np.int64),
        "confusion_matrix_normalized": np.zeros((0, 0), dtype=np.float32),
        "predictions": pd.DataFrame(),
        "per_length_accuracy": pd.DataFrame(),
        "sampled_dataframe": pd.DataFrame(),
    }
