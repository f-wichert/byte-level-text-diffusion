import re
from typing import Callable, Sequence

import pandas as pd


def analyze_neighbour_to_synonym_relationship(
    df: pd.DataFrame,
    *,
    find_nearest_neighbors_fn: Callable,
    encode_word_fn: Callable,
    prepare_neighbor_space_fn: Callable | None = None,
    embedding_col: str = "patch_embeddings_array",
    aggregation_method: str = "mean",
    label_col: str = "Word",
    synonym_col: str = "synonyms",
    n_neighbors: int = 10,
    min_synonyms: int = 2,
    synonym_delimiters: Sequence[str] = (";", "|"),
    lowercase_matching: bool = True,
):
    """
    Simple analysis flow:
    1. filter anchor words with enough synonyms
    2. parse synonym lists
    3. encode missing synonyms
    4. find nearest neighbours for the original anchor words
    5. compare neighbours against the anchor synonym lists
    """
    # Validate the minimum dataframe shape we need before doing any work.
    required_columns = {label_col, embedding_col, synonym_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    # Keep only rows that can participate in the neighbour space.
    vocabulary_df = df[
        df[embedding_col].notna() & df[label_col].notna()
    ].copy().reset_index(drop=True)

    if vocabulary_df.empty:
        return {
            "anchor_count": 0,
            "neighbor_space_size": 0,
            "encoded_missing_synonym_count": 0,
            "mean_overlap_count": 0.0,
            "mean_synonym_recall": 0.0,
            "hit_rate": 0.0,
            "details": pd.DataFrame(),
            "anchor_dataframe": pd.DataFrame(),
            "neighbor_space_dataframe": pd.DataFrame(),
            "neighbors_dataframe": pd.DataFrame(),
        }

    vocabulary_df[label_col] = vocabulary_df[label_col].astype(str).str.strip()
    vocabulary_df = vocabulary_df[
        vocabulary_df[label_col] != ""
    ].drop_duplicates(subset=[label_col], keep="first").reset_index(drop=True)

    # Build the set of original anchor words we want to evaluate.
    # At this stage we only parse and filter by synonym-list size.
    vocabulary_words = vocabulary_df[label_col].tolist()
    resolver = _build_resolver(vocabulary_words, lowercase_matching=lowercase_matching)

    anchor_df = vocabulary_df.copy()
    anchor_df["parsed_synonyms"] = anchor_df.apply(
        lambda row: _parse_synonyms(
            row[synonym_col],
            synonym_delimiters=synonym_delimiters,
            anchor_word=row[label_col],
        ),
        axis=1,
    )
    anchor_df = anchor_df[
        anchor_df["parsed_synonyms"].apply(len) >= min_synonyms
    ].copy().reset_index(drop=True)

    if anchor_df.empty:
        return {
            "anchor_count": 0,
            "neighbor_space_size": int(len(vocabulary_df)),
            "encoded_missing_synonym_count": 0,
            "mean_overlap_count": 0.0,
            "mean_synonym_recall": 0.0,
            "hit_rate": 0.0,
            "details": pd.DataFrame(),
            "anchor_dataframe": anchor_df,
            "neighbor_space_dataframe": vocabulary_df,
            "neighbors_dataframe": pd.DataFrame(),
        }

    # Some synonyms are not part of the original dataframe.
    # Those get encoded so the neighbour space contains both anchors and their synonyms.
    missing_synonyms = sorted(
        {
            synonym
            for synonyms in anchor_df["parsed_synonyms"]
            for synonym in synonyms
            if resolver(synonym) is None
        }
    )

    new_rows = []
    for synonym in missing_synonyms:
        patch_embedding = encode_word_fn(synonym)
        new_rows.append(
            {
                label_col: synonym,
                synonym_col: "",
                embedding_col: patch_embedding,
            }
        )

    # The neighbour search runs over the original vocabulary plus newly encoded synonyms.
    neighbor_space_df = vocabulary_df.copy()
    if new_rows:
        neighbor_space_df = pd.concat(
            [neighbor_space_df, pd.DataFrame(new_rows)],
            ignore_index=True,
        )

    if prepare_neighbor_space_fn is not None:
        neighbor_space_df = prepare_neighbor_space_fn(neighbor_space_df)

    neighbor_space_words = neighbor_space_df[label_col].astype(str).tolist()
    neighbor_resolver = _build_resolver(
        neighbor_space_words,
        lowercase_matching=lowercase_matching,
    )

    # Resolve synonym strings against the final neighbour space so later comparisons
    # use exactly the same surface forms that appear in the neighbour results.
    anchor_df["synonyms_in_neighbor_space"] = anchor_df["parsed_synonyms"].apply(
        lambda synonyms: [
            resolved
            for resolved in (
                neighbor_resolver(synonym)
                for synonym in synonyms
            )
            if resolved is not None
        ]
    )
    anchor_df = anchor_df[
        anchor_df["synonyms_in_neighbor_space"].apply(len) > 0
    ].copy().reset_index(drop=True)

    if anchor_df.empty:
        return {
            "anchor_count": 0,
            "neighbor_space_size": int(len(neighbor_space_df)),
            "encoded_missing_synonym_count": int(len(new_rows)),
            "mean_overlap_count": 0.0,
            "mean_synonym_recall": 0.0,
            "hit_rate": 0.0,
            "details": pd.DataFrame(),
            "anchor_dataframe": anchor_df,
            "neighbor_space_dataframe": neighbor_space_df,
            "neighbors_dataframe": pd.DataFrame(),
        }

    # Find neighbours in the full comparison space, then keep only the rows that
    # correspond to the original anchor words.
    neighbors_df = find_nearest_neighbors_fn(
        df=neighbor_space_df,
        embedding_col=embedding_col,
        aggregation_method=aggregation_method,
        n_neighbors=n_neighbors,
        label_col=label_col,
        include_similarity_scores=True,
    ).reset_index(drop=True)

    anchor_neighbors_df = neighbors_df[
        neighbors_df[label_col].isin(anchor_df[label_col])
    ].copy().reset_index(drop=True)

    anchor_lookup = anchor_df.set_index(label_col)
    detail_rows = []

    # Compare each anchor's nearest neighbours against its synonym set and store
    # per-word overlap statistics for later inspection.
    for _, row in anchor_neighbors_df.iterrows():
        word = row[label_col]
        synonyms = list(anchor_lookup.at[word, "synonyms_in_neighbor_space"])
        synonym_set = set(synonyms)
        neighbors = list(row["neighbors_list"])
        similarities = (
            list(row["neighbors_similarity_list"])
            if "neighbors_similarity_list" in row and isinstance(row["neighbors_similarity_list"], list)
            else []
        )

        matched_neighbors = [neighbor for neighbor in neighbors if neighbor in synonym_set]
        matched_synonyms = [synonym for synonym in synonyms if synonym in set(matched_neighbors)]
        first_match_rank = next(
            (rank for rank, neighbor in enumerate(neighbors, start=1) if neighbor in synonym_set),
            None,
        )

        detail_rows.append(
            {
                "word": word,
                "parsed_synonyms": list(anchor_lookup.at[word, "parsed_synonyms"]),
                "synonyms_in_neighbor_space": synonyms,
                "neighbors": neighbors,
                "neighbor_similarities": similarities,
                "matched_neighbors": matched_neighbors,
                "matched_synonyms": matched_synonyms,
                "overlap_count": len(matched_neighbors),
                "synonym_recall": len(matched_synonyms) / len(synonyms) if synonyms else 0.0,
                "neighbor_precision": len(matched_neighbors) / len(neighbors) if neighbors else 0.0,
                "has_match": len(matched_neighbors) > 0,
                "first_match_rank": first_match_rank,
            }
        )

    details_df = pd.DataFrame(detail_rows)

    # Return both aggregate scores and the intermediate tables so the caller can
    # inspect anchors, the expanded neighbour space, and per-word matches.
    return {
        "anchor_count": int(len(details_df)),
        "neighbor_space_size": int(len(neighbor_space_df)),
        "encoded_missing_synonym_count": int(len(new_rows)),
        "mean_overlap_count": float(details_df["overlap_count"].mean()) if not details_df.empty else 0.0,
        "mean_synonym_recall": float(details_df["synonym_recall"].mean()) if not details_df.empty else 0.0,
        "mean_neighbor_precision": float(details_df["neighbor_precision"].mean()) if not details_df.empty else 0.0,
        "hit_rate": float(details_df["has_match"].mean()) if not details_df.empty else 0.0,
        "details": details_df,
        "anchor_dataframe": anchor_df,
        "neighbor_space_dataframe": neighbor_space_df,
        "neighbors_dataframe": anchor_neighbors_df,
    }


def _parse_synonyms(
    synonym_value,
    *,
    synonym_delimiters: Sequence[str],
    anchor_word: str | None,
) -> list[str]:
    if pd.isna(synonym_value) or synonym_value == "":
        return []

    if isinstance(synonym_value, list):
        raw_synonyms = [str(item).strip() for item in synonym_value if str(item).strip()]
    else:
        pattern = "|".join(re.escape(delimiter) for delimiter in synonym_delimiters)
        raw_synonyms = [
            item.strip()
            for item in re.split(pattern, str(synonym_value))
            if item.strip()
        ]

    seen = set()
    parsed = []
    anchor_key = anchor_word.casefold() if anchor_word is not None else None

    for synonym in raw_synonyms:
        synonym_key = synonym.casefold()
        if anchor_key is not None and synonym_key == anchor_key:
            continue
        if synonym_key in seen:
            continue
        seen.add(synonym_key)
        parsed.append(synonym)

    return parsed


def _build_resolver(words: Sequence[str], *, lowercase_matching: bool):
    exact_words = set(words)
    lowercase_map = {}

    for word in words:
        lowercase_map.setdefault(word.casefold(), set()).add(word)

    def resolve(term: str):
        if term in exact_words:
            return term

        if not lowercase_matching:
            return None

        candidates = lowercase_map.get(term.casefold(), set())
        if len(candidates) == 1:
            return next(iter(candidates))

        return None

    return resolve
