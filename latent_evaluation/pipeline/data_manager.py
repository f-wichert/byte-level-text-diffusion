import os
import torch

import pandas as pd
import numpy as np

from tqdm import tqdm
from typing import Optional, Literal, List

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.kernel_approximation import RBFSampler

SYNONYMS_DATASET = "synonyms"
CONTEXT_DATASET = "context"
CONTEXT_ALLOWED_ENCODERS = ("tfree-hat", "tfree-hat-finetuned")


class DataManager():

    data_directory: str = None
    output_directory: str = None
    base_dataset_name: str = "output_c"
    context_corpus_name: str = "context_corpus"

    current_model_type: str = None
    # Name used in cache filenames; tagged with the AE variant when a
    # non-default checkpoint is loaded (set by Pipeline.load_encoder_model).
    cache_model_name: str = None
    encoder_model = None
    pipeline_device: str = "cuda:0"

    current_subsample: int = None
    current_decomposition: str = None
    current_dataset_type: str = SYNONYMS_DATASET
    current_key = None

    dataframes = {}

    def __init__(
            self, 
            data_directory,
            output_directory,
            data_folder_path="NOT IMPLEMENTED",
        ):
        self.data_directory = data_directory
        self.output_directory = output_directory
        self.dataframes = {
            "dataframes": {},
            "neighbours": {},
        }

        print("DataManager initialized.")


    @property
    def cache_name(self):
        return self.cache_model_name or self.current_model_type

    @staticmethod
    def _normalize_subsample(subsample):
        # Unify int and numeric-string subsamples so they share one cache key
        # and one parquet filename suffix.
        if isinstance(subsample, str) and subsample.isdigit():
            return int(subsample)
        return subsample

    def prepare_data(
            self,
            model_type: str,
            subsample: int | str = None,
            decomposition: str = None,
            compute_nn: bool = True,
            dataset: str = SYNONYMS_DATASET,
        ):
        if dataset not in (SYNONYMS_DATASET, CONTEXT_DATASET):
            raise ValueError(f"Unknown dataset type: '{dataset}'")

        if dataset == CONTEXT_DATASET:
            if decomposition is not None:
                raise ValueError("Decompositions are not supported for the context dataset.")
            if compute_nn:
                raise ValueError(
                    "compute_nn is not supported for the context dataset (the similarity "
                    "matrix over ~170k positions would need ~60 GB). "
                    'Set "compute_nn": false on the task.'
                )
            subsample = None  # the context corpus has no subsampling semantics

        self.current_model_type = model_type
        self.current_subsample = self._normalize_subsample(subsample)
        self.current_decomposition = decomposition
        self.current_dataset_type = dataset
        self.current_key = (
            dataset, self.cache_name, self.current_subsample, decomposition,
        )

        df_cache = self.dataframes["dataframes"]
        nn_cache = self.dataframes["neighbours"]

        if dataset == CONTEXT_DATASET:
            if df_cache.get(self.current_key) is None:
                df_cache[self.current_key] = self._load_or_encode_context_parquet()
            nn_cache.setdefault(self.current_key, None)
            return

        if df_cache.get(self.current_key) is None:
            df_parquet = self.load_or_encode_parquet()
            df_cache[self.current_key] = self.apply_decomposition_if_required(df_parquet)

        if compute_nn:
            if nn_cache.get(self.current_key) is None:
                nn_cache[self.current_key] = self.load_or_find_nearest_neighbours(
                    df_cache[self.current_key]
                )
        else:
            print("NO NEIGHBOURS COMPUTED")
            nn_cache.setdefault(self.current_key, None)

    def get_current_dataframe(self):
        return self.dataframes["dataframes"][self.current_key]

    def get_current_dataframe_with_neighbours(self):
        return self.dataframes["neighbours"][self.current_key]


    def load_or_encode_parquet(self):
        print("LOG: load_or_encode_parquet")
        self.process_csv_to_parquet(
            csv_name=f'{self.base_dataset_name}.csv', 
            column_to_encode="Word", 
            random_subsample=self.current_subsample if not self.current_subsample == "all" else None, 
            encode_if_parquet_file_present=False,
        )

        print("To load: " + f'{self.base_dataset_name}_{self.cache_name}{"_" + str(self.current_subsample) if self.current_subsample and not self.current_subsample == "all" else ""}.parquet')

        return self.load_file(
            file_name=f'{self.base_dataset_name}_{self.cache_name}{"_" + str(self.current_subsample) if self.current_subsample and not self.current_subsample == "all" else ""}.parquet'
        )

    def apply_decomposition_if_required(self, df):
        if self.current_decomposition is None:
            return df
        
        if self.current_decomposition == "pca":
            df_transformed = self.apply_kernel_pca(
                df=df,
                embedding_col="patch_embeddings_array",
                n_components=200,
                kernel="rbf",
                gamma=0.5,
                # output_col="kernel_embeddings"  
            )
        elif self.current_decomposition == "rbf":
            df_transformed = self.apply_rbf(
                df=df,
                embedding_col="patch_embeddings_array",
                n_components=200,
                gamma=0.5,
                # output_col="rbf_embeddings"
            )
        
        return df_transformed

    def load_or_find_nearest_neighbours(self, df):
        return self.find_nearest_neighbors_and_save(
            df=df,
            embedding_col="patch_embeddings_array",
            aggregation_method="mean",
            n_neighbors=10,
            include_similarity_scores=True,
            find_if_csv_file_present=True,
            save=False
        )

    # ========== Context dataset ==========
    def _load_or_encode_context_parquet(self):
        if self.current_model_type not in CONTEXT_ALLOWED_ENCODERS:
            raise ValueError(
                f"The context dataset requires a word-aligned sequence encoder "
                f"{CONTEXT_ALLOWED_ENCODERS}; got '{self.current_model_type}'."
            )

        parquet_name = f"{self.context_corpus_name}_{self.cache_name}.parquet"
        if not os.path.exists(os.path.join(self.data_directory, parquet_name)):
            self._encode_context_corpus_to_parquet(parquet_name)
        else:
            print(f"Aborted encoding because file was already present: {parquet_name}")

        return self.load_file(file_name=parquet_name)

    def _encode_context_corpus_to_parquet(self, parquet_name):
        if self.encoder_model is None:
            raise ValueError("No encoder model loaded; cannot encode the context corpus.")

        corpus_path = os.path.join(self.data_directory, f"{self.context_corpus_name}.txt")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Context corpus not found at {corpus_path}")

        with open(corpus_path, encoding="utf-8") as f:
            documents = [line for line in f.read().splitlines() if line.strip()]

        doc_ids = []
        positions = []
        patch_embeddings_list = []
        patch_embedding_shapes = []

        with torch.no_grad():
            for doc_id, document in enumerate(tqdm(documents, desc="Encoding context documents")):
                result = self.encoder_model.encode_text(document, device=self.pipeline_device)
                emb = result["patch_embeddings"].detach().to(device="cpu", dtype=torch.float32).numpy()
                if emb.ndim == 3 and emb.shape[0] == 1:
                    emb = emb[0]  # tfree-hat-finetuned keeps a leading batch dim on z_words
                if emb.ndim != 2:
                    raise ValueError(
                        f"Expected word-aligned sequence latents [num_words, dim], "
                        f"got shape {emb.shape} from '{self.current_model_type}'."
                    )
                for position in range(emb.shape[0]):
                    vector = np.ascontiguousarray(emb[position])
                    doc_ids.append(doc_id)
                    positions.append(position)
                    patch_embeddings_list.append(vector.tobytes())
                    patch_embedding_shapes.append(vector.shape)

        df = pd.DataFrame({
            "doc_id": doc_ids,
            "position": positions,
            "patch_embeddings": patch_embeddings_list,
            "patch_embedding_shape": patch_embedding_shapes,
        })

        output_path = os.path.join(self.data_directory, parquet_name)
        df.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")



    def load_file(self, file_name: str, filter_out_duplicate_subset: Optional[List[str]] = None):
        if file_name.endswith(".csv"):
            return self.load_csv(file_name, filter_out_duplicate_subset=filter_out_duplicate_subset)
        elif file_name.endswith(".parquet"):
            return self.load_parquet_with_embeddings(file_name, filter_out_duplicate_subset=filter_out_duplicate_subset)
        
        raise ValueError(f'The ending of the file ${file_name} is not associated with an opening method.')

    def load_csv(self, csv_name: str, filter_out_duplicate_subset: Optional[List[str]]):
        df = pd.read_csv(os.path.join(self.data_directory, csv_name))
        if filter_out_duplicate_subset is not None:
            df = df.drop_duplicates(subset=filter_out_duplicate_subset)
        if set_current_dataframe:
            self.current_dataframe = df
        return df

    def load_parquet_with_embeddings(self, parquet_name, filter_out_duplicate_subset: Optional[List[str]], hard_data_path: str = None):
        """
        Also deserializes embeddings back to numpy arrays.
        """
        if hard_data_path is not None:
            df = pd.read_parquet(os.path.join(hard_data_path, parquet_name))
        else:
            df = pd.read_parquet(os.path.join(self.data_directory, parquet_name))

        def deserialize_embedding(row, emb_col, shape_col):
            if row[emb_col] is None or row[shape_col] is None:
                return None
            return np.frombuffer(row[emb_col], dtype=np.float32).reshape(row[shape_col])
        
        if "byte_embeddings_array" in df:
            df["byte_embeddings_array"] = df.apply(
                lambda row: deserialize_embedding(row, "byte_embeddings", "byte_embedding_shape"), 
                axis=1
            )
        df["patch_embeddings_array"] = df.apply(
            lambda row: deserialize_embedding(row, "patch_embeddings", "patch_embedding_shape"), 
            axis=1
        )

        if filter_out_duplicate_subset is not None:
            df = df.drop_duplicates(subset=filter_out_duplicate_subset)

        return df

    def process_csv_to_parquet(
        self, 
        csv_name, 
        column_to_encode,
        encode_if_parquet_file_present=False,
        random_subsample: int = None,
    ):
        """
        Read CSV and **encode all words**!!!
        """
        print("debug")
        print(self.data_directory)
        print(csv_name)
        print(self.current_model_type)
        print(random_subsample)

        # Check if output should be overwritten
        if random_subsample is not None:
            if os.path.exists(os.path.join(self.data_directory, csv_name[:-4] + f"_{self.cache_name}" + f"_{random_subsample}" + ".parquet")) and not encode_if_parquet_file_present: # data is already encoded
                print("Aborted encoding because file was already present: " + self.data_directory + "/" + csv_name[:-4] + f"_{self.cache_name}" + f"_{random_subsample}" + ".parquet")
                return
        else:
            if os.path.exists(os.path.join(self.data_directory, csv_name[:-4] + f"_{self.cache_name}" + ".parquet")) and not encode_if_parquet_file_present: # data is already encoded
                print("Aborted encoding because file was already present: " + self.data_directory + "/" + csv_name[:-4] + f"_{self.cache_name}" + ".parquet")
                return

        if self.encoder_model is None:
            self.encoder_model

        df = pd.read_csv(os.path.join(self.data_directory, csv_name))
        if random_subsample is not None:
            np.random.seed(random_subsample)
            df = df.sample(n=random_subsample)


        byte_embeddings_list = []
        patch_embeddings_list = []
        byte_embedding_shapes = []
        patch_embedding_shapes = []
        

        with torch.no_grad():
            for lemma in tqdm(df[column_to_encode], desc="Encoding words"):
                result = self.encoder_model.encode_text(str(lemma), device=self.pipeline_device)

                def to_numpy_float32(tensor: torch.Tensor) -> np.ndarray:
                    # Parquet serialization/deserialization in this pipeline assumes float32 bytes.
                    return tensor.detach().to(device="cpu", dtype=torch.float32).numpy()

                if "byte_embeddings" in result:
                    byte_emb = to_numpy_float32(result["byte_embeddings"])
                    byte_embeddings_list.append(byte_emb.tobytes())
                    byte_embedding_shapes.append(byte_emb.shape)

                patch_emb = to_numpy_float32(result["patch_embeddings"])
                patch_embeddings_list.append(patch_emb.tobytes())
                patch_embedding_shapes.append(patch_emb.shape)
                    
        if "byte_embeddings" in result:
            df["byte_embeddings"] = byte_embeddings_list
            df["byte_embedding_shape"] = byte_embedding_shapes
        df["patch_embeddings"] = patch_embeddings_list
        df["patch_embedding_shape"] = patch_embedding_shapes
        
        if random_subsample is not None:
            output_path = os.path.join(self.data_directory, csv_name[:-4] + f"_{self.cache_name}" + f"_{random_subsample}" + ".parquet")
        else:
            output_path = os.path.join(self.data_directory, csv_name[:-4] + f"_{self.cache_name}" + ".parquet")
        df.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")
        
        return df
 


    # ========== Decomposition ==========
    def aggregate_embeddings(
        self, 
        embedding: np.ndarray, 
        method: str = "mean"
    ) -> np.ndarray:
        if embedding is None:
            return None
        
        if embedding.ndim == 1:
            return embedding
        
        elif embedding.ndim == 2:
            if method == "mean":
                return np.mean(embedding, axis=0)
            elif method == "max":
                return np.max(embedding, axis=0)
            elif method == "first":
                return embedding[0]
            elif method == "last":
                return embedding[-1]
        
        elif embedding.ndim == 3:
            total_patches = embedding.shape[0] * embedding.shape[1]
            dim = embedding.shape[2]
            embedding_2d = embedding.reshape(total_patches, dim)

            if method == "mean":
                return np.mean(embedding_2d, axis=0)
            elif method == "max":
                return np.max(embedding_2d, axis=0)
            elif method == "first":
                return embedding_2d[0]
            elif method == "last":
                return embedding_2d[-1]
        
        raise ValueError(f"Unknown aggregation method: {method}")

    def apply_kernel_pca(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        n_components: int = 50,
        kernel: str = "rbf",
        gamma: float = 0.1,
        output_col: str = None,
    ) -> pd.DataFrame:
        """
        Apply Kernel PCA to embeddings in a dataframe column.
        
        Args:
            df: DataFrame with embeddings
            embedding_col: Column containing embeddings
            n_components: Number of components to keep
            kernel: Kernel type ('rbf', 'poly', 'cosine', 'linear')
            gamma: Kernel coefficient (for rbf/poly)
            output_col: Output column name (default: overwrites embedding_col)
        
        Returns:
            DataFrame with transformed embeddings
        """
        print(f"Kernel PCA")
        if df is None:
            assert self.current_dataframe is not None
            df = self.current_dataframe
        
        df = df.copy()
        
        if output_col is None:
            output_col = embedding_col
        
        print(f"Aggregating embeddings...")
        aggregated = []
        for emb in df[embedding_col].values:
            agg = self.aggregate_embeddings(emb)
            aggregated.append(agg)
        
        embeddings = np.vstack(aggregated)
        print(f"Input shape: {embeddings.shape}")
        
        # Apply Kernel PCA
        print(f"Fitting Kernel PCA (kernel={kernel}, gamma={gamma}, n_components={n_components})...")
        kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
        transformed = kpca.fit_transform(embeddings)
        print(f"Output shape: {transformed.shape}")
        
        df[output_col] = list(transformed)

        if output_col is not None:
            df[output_col] = list(transformed)
        else:
            df[embedding_col] = list(transformed)
        
        self.current_dataframe = df

        return df

    def apply_rbf(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        n_components: int = 50,
        gamma: float = 0.1,
        random_state: int = 42,
        output_col: str = None,
    ) -> pd.DataFrame:
        """
        Apply an RBF feature mapping to embeddings in a dataframe column.
        
        Args:
            df: DataFrame with embeddings
            embedding_col: Column containing embeddings
            n_components: Number of components to keep
            gamma: Kernel coefficient for the RBF mapping
            output_col: Output column name (default: overwrites embedding_col)
        
        Returns:
            DataFrame with transformed embeddings
        """
        
        if df is None:
            assert self.current_dataframe is not None
            df = self.current_dataframe
        
        df = df.copy()
        
        if output_col is None:
            output_col = embedding_col
        
        print(f"Aggregating embeddings...")
        aggregated = []
        for emb in df[embedding_col].values:
            agg = self.aggregate_embeddings(emb, method=aggregation_method)
            aggregated.append(agg)
        
        embeddings = np.vstack(aggregated)
        print(f"Input shape: {embeddings.shape}")
        
        # Apply RBF mapping
        print(f"Fitting RBF mapping (gamma={gamma}, n_components={n_components}, random_state={random_state})...")
        rbf_map = RBFSampler(
            gamma=gamma,
            n_components=n_components,
            random_state=random_state,
        )
        transformed = rbf_map.fit_transform(embeddings)
        print(f"Output shape: {transformed.shape}")
        
        if output_col is not None:
            df[output_col] = list(transformed)
        else:
            df[embedding_col] = list(transformed)

        self.current_dataframe = df
        
        return df

    # =========== Neihgbours ===========
    def find_nearest_neighbors(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        n_neighbors: int = 10,
        label_col: str = "Word",
        include_similarity_scores: bool = True,
        pos_col: str = "part_of_speech", 
    ):
        """
        Find the k nearest neighbors for each word using cosine similarity.
        
        Args:
            df: DataFrame with embeddings from load_parquet_with_embeddings
                If this is none, use the current dataframe of the pipeline
            embedding_col: Column containing embeddings
            aggregation_method: How to aggregate variable-length embeddings
            n_neighbors: Number of nearest neighbors to find
            label_col: Column containing word labels
            include_similarity_scores: Whether to include similarity scores in output
        
        Returns:
            DataFrame with nearest neighbor columns added
        """
        if df is None:
            assert self.current_dataframe is not None
            df = self.current_dataframe

        # Filter rows
        df_valid = df[df[embedding_col].notna()].copy().reset_index(drop=True)
        if len(df_valid) == 0:
            raise ValueError("No valid embeddings found in the dataframe")
        
        print("Aggregating embeddings...")
        embeddings_aggregated = df_valid[embedding_col].apply(
            lambda x: self.aggregate_embeddings(x, method=aggregation_method)
        )
        
        X = np.vstack(embeddings_aggregated.values)
        print(f"Embedding matrix shape: {X.shape}")
        
        print("Computing cosine similarity matrix...")
        similarity_matrix = cosine_similarity(X)
        
        # Don't match self
        np.fill_diagonal(similarity_matrix, -np.inf)
        

        labels = df_valid[label_col].values
        pos_tags = df_valid[pos_col].values 

        effective_n_neighbors = min(n_neighbors, len(labels) - 1)
        if effective_n_neighbors != n_neighbors:
            print(f"Adjusted n_neighbors from {n_neighbors} to {effective_n_neighbors}")
        
        # Find neighbours
        print(f"Finding {effective_n_neighbors} nearest neighbors for each word...")
        neighbor_labels = []
        neighbor_scores = []
        neighbor_pos = [] 
        for i in tqdm(range(len(labels)), desc="Finding neighbors"):
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1][:effective_n_neighbors]
            
            top_labels = [labels[idx] for idx in top_indices]
            top_scores = [similarities[idx] for idx in top_indices]
            top_pos = [pos_tags[idx] for idx in top_indices]
            
            neighbor_labels.append(top_labels)
            neighbor_scores.append(top_scores)
            neighbor_pos.append(top_pos)
        
        # Add to dataframe
        for k in range(effective_n_neighbors):
            df_valid[f"neighbor_{k+1}"] = [
                neighbors[k] if k < len(neighbors) else None 
                for neighbors in neighbor_labels
            ]
            df_valid[f"neighbor_{k+1}_pos"] = [                       
                pos[k] if k < len(pos) else None
                for pos in neighbor_pos
            ]
            if include_similarity_scores:
                df_valid[f"neighbor_{k+1}_similarity"] = [
                    scores[k] if k < len(scores) else None 
                    for scores in neighbor_scores
                ]
        
        df_valid["neighbors_list"] = neighbor_labels
        df_valid["neighbors_pos_list"] = neighbor_pos
        if include_similarity_scores:
            df_valid["neighbors_similarity_list"] = neighbor_scores
        
        return df_valid

    def find_nearest_neighbors_and_save(
        self,
        df: pd.DataFrame = None,
        output_path: str = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        n_neighbors: int = 10,
        label_col: str = "Word",
        include_similarity_scores: bool = True,
        drop_embedding_columns: bool = True,
        find_if_csv_file_present: bool = False,
        save: bool = True
    ):    
        output_path = os.path.join(self.data_directory, f'{self.base_dataset_name}_{self.cache_name}{"_" + str(self.current_subsample) if self.current_subsample and not self.current_subsample == "all" else ""}_with_neighbours.csv')
        if find_if_csv_file_present is False and os.path.exists(output_path):
            self.current_dataframe_with_neighbors = pd.read_csv(output_path)
            return

        if df is None:
            assert self.get_current_dataframe() is not None
            df = self.get_current_dataframe().copy()


        df_result = self.find_nearest_neighbors(
            df=df,
            embedding_col=embedding_col,
            aggregation_method=aggregation_method,
            n_neighbors=n_neighbors,
            label_col=label_col,
            include_similarity_scores=include_similarity_scores,
        )
        df_export = df_result.copy()
        

        df_export["neighbors_list"] = df_export["neighbors_list"].apply(
            lambda x: ", ".join([str(item) for item in x if item is not None]) if x else ""
        )
        if include_similarity_scores:
            df_export["neighbors_similarity_list"] = df_export["neighbors_similarity_list"].apply(
                lambda x: ", ".join([f"{s:.4f}" for s in x if s is not None]) if x else ""
            )
        
        # Remove embedding column
        if drop_embedding_columns:
            cols_to_drop = [
                col for col in df_export.columns 
                if "embedding" in col.lower() and col not in ["neighbors_list", "neighbors_similarity_list"]
            ]
            df_export = df_export.drop(columns=cols_to_drop, errors="ignore")
        
        
        # Save to CSV
        if save:
            df_export.to_csv(output_path, index=False)
            print(f"Saved to {output_path}")
        print(f"Columns: {list(df_export.columns)}")
        
        # self.current_dataframe_with_neighbors = df_result
        return df_export
