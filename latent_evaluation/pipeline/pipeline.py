import logging
import os
import shutil
import torch
from datetime import datetime

try:
    import umap
except ImportError:
    umap = None

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from collections import defaultdict, Counter
from functools import partial
from itertools import combinations
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from typing import Optional, Literal, List

from pipeline.analyses.neighbour_to_synonym_analysis import (
    analyze_neighbour_to_synonym_relationship as run_neighbour_to_synonym_analysis,
)
from pipeline.analyses.isotropy_analysis import (
    analyze_isotropy as run_isotropy_analysis,
)
from pipeline.analyses.typo_analysis import (
    analyze_typo_robustness as run_typo_robustness_analysis,
)
from pipeline.analyses.linear_probe_word_length import (
    analyze_linear_probe_word_length as run_linear_probe_word_length_analysis,
)
from pipeline.analyses.covariance_analysis import (
    analyze_covariance as run_covariance_analysis,
)
from pipeline.analyses.explained_variance_analysis import (
    analyze_explained_variance as run_explained_variance_analysis,
)
from pipeline.encoder import BLTEncoder, BolmoEncoder, GeminiEncoder, LangFlowEncoder, NeoBERTEncoder, TFreeHatEncoder, TFreeHatFineTunedEncoder

from pipeline.data_manager import DataManager

loggger = logging.getLogger(__name__)

class Pipeline():

    # ======================================
    # ========== Class Parameters ==========
    # ======================================
    steps = []
    
    encoder_type: str = None
    encoder_model = None

    data_directory = None
    dataset_name: str = None
    current_subsample: int = None

    output_directory = None
    output_image_directory = None
    pipeline_device = None

    current_dataframe = None
    current_dataframe_with_neighbors = None

    verbose = True

    # Report class to handle plotpy stuff
    report = None

    @staticmethod
    def _tensor_to_numpy_float32(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().to(device="cpu", dtype=torch.float32).numpy()


    # ======================================
    # ==========  Pipeline Usage  ==========
    # ======================================
    def __init__(
            self,
            data_directory: str,
            output_directory: str,
        ):
        self.data_manager = DataManager(data_directory=data_directory, output_directory=output_directory)
        self.set_data_directory(data_directory)
        self.set_output_directory(output_directory)


        self.outpout_image_directory = os.path.join(self.output_directory, f"{datetime.today().strftime('%Y-%m-%d')}_images")
        try:
            os.mkdir(self.outpout_image_directory)
            print(f"Directory '{directory_name}' created successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")


        print("Pipeline initialized.")

    def start_report(self, report_name, initial_heading=None, overwrite=True):
        if self.report is None:
            self.report = self.Report(output_directory=self.output_directory)

        self.report.start_report(report_name, overwrite)
        
        if initial_heading:
            self.report.add_heading(initial_heading, level=1)

        self.report.add_heading("Parameters", level=2)
        self.report.print(f"Encoder Model: {self.encoder_type}")

    def report_current_dataset(self):
        self.report.print(f"Current Dataset: {self.data_manager.current_dataset_type}")
        self.report.print(f"Current Subsample: {self.data_manager.current_subsample}")
        self.report.print(f"Current Decomposition: {self.data_manager.current_decomposition}")

    # ======================================
    # ========== Task Definition  ==========
    # ======================================
    def load_encoder_model(self, model_type=None, ae_config=None, ae_checkpoint=None):
        if model_type is None:
            model_type = self.encoder_type
        if (ae_config is not None or ae_checkpoint is not None) and model_type != "tfree-hat-finetuned":
            raise ValueError(
                "--ae-config/--ae-checkpoint are only supported for encoder 'tfree-hat-finetuned'."
            )
        if model_type == "BLT" or model_type == "blt":
            self.encoder_type = "blt"
            self.encoder_model = BLTEncoder()
        elif model_type == "BOLMO" or model_type == "bolmo":
            self.encoder_type = "bolmo"
            self.encoder_model = BolmoEncoder()
        elif model_type == "NeoBERT" or model_type == "neobert":
            self.encoder_type = "neobert"
            self.encoder_model = NeoBERTEncoder()
        elif model_type == "Gemini" or model_type == "gemini":
            self.encoder_type = "gemini"
            self.encoder_model = GeminiEncoder()
        elif model_type == "LangFlow" or model_type == "langflow":
            self.encoder_type = "langflow"
            self.encoder_model = LangFlowEncoder()
        elif model_type == "tfree-hat":
            self.encoder_type = "tfree-hat"
            self.encoder_model = TFreeHatEncoder()
        elif model_type == "tfree-hat-finetuned":
            self.encoder_type = "tfree-hat-finetuned"
            self.encoder_model = TFreeHatFineTunedEncoder(
                ae_config_path=ae_config, ae_checkpoint_path=ae_checkpoint
            )
        else:
            raise ValueError("Unknown Encoder Type")

        self.encoder_model.load_model(device=self.pipeline_device)
        self.data_manager.encoder_model = self.encoder_model

        # Non-default AE variants tag the encoder name so caches, reports and
        # images never mix checkpoints; dispatch above stays on the plain name.
        variant_tag = getattr(self.encoder_model, "variant_tag", None)
        if variant_tag is not None:
            self.encoder_type = f"{self.encoder_type}-{variant_tag}"
        self.data_manager.cache_model_name = self.encoder_type





    # ======================================
    # =========== Visualizations ===========
    # ======================================

    # ================ PCA =================
    def visualize_embeddings_pca(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        n_components: int = 2,
        color_by_column: str = "part_of_speech",
        label_col: str = "Word",
        show_labels: bool = False,
        max_labels: int = 50,
        figsize: tuple = (12, 8),
        title: str = None,
        save_path: str = None,
        save_name: str = None,
        save: bool = False,
        show_plot: bool = True,
    ):
        # Prepare dataframe
        # Use pipeline frame if no parameter is given
        # Filter invalid embeddings and throw error if there are no valid embeddings
        # Probably messed up encoding then
        # Randomly sample if applicable
        print(f"PCA")
        if df is None:
            assert self.data_manager.get_current_dataframe() is not None
            df = self.data_manager.get_current_dataframe()

        df_valid = df[df[embedding_col].notna()].copy().reset_index(drop=True)

        if len(df_valid) == 0:
            raise ValueError("No valid embeddings found")

        print(f"Analyzing {len(df_valid)} words with valid embeddings")
        
        embeddings_aggregated = df_valid[embedding_col].apply(
            lambda x: self.aggregate_embeddings(x, method=aggregation_method)
        )
        X = np.vstack(embeddings_aggregated.values)
        
        print(f"Embedding matrix shape: {X.shape}")
        
        # make mean=0 and std=1
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        

        # Prepare figure stuff
        hover_texts = []
        if label_col in df_valid.columns:
            for idx, row in df_valid.iterrows():
                hover_text = f"{label_col}: {row[label_col]}"
                if color_by_column and color_by_column in df_valid.columns:
                    hover_text += f"<br>{color_by_column}: {row[color_by_column]}"
                hover_texts.append(hover_text)
        else:
            hover_texts = [f"Point {i}" for i in range(len(df_valid))]

        if color_by_column and color_by_column in df_valid.columns:
            categories = df_valid[color_by_column].fillna("unknown").astype(str)
            color_discrete_map = None
        else:
            categories = None
            color_discrete_map = None

        if title is None:
            title = f"PCA Visualization of {embedding_col}"

        if n_components == 2:
            fig = go.Figure()
            if color_by_column and color_by_column in df_valid.columns:
                unique_categories = categories.unique()
                colors_palette = px.colors.qualitative.Plotly if len(unique_categories) <= 10 else px.colors.qualitative.Light24
                
                for idx, cat in enumerate(unique_categories):
                    mask = categories == cat
                    
                    category_labels = df_valid[mask][label_col].values if label_col in df_valid.columns else None
                    
                    fig.add_trace(go.Scatter(
                        x=X_pca[mask, 0],
                        y=X_pca[mask, 1],
                        mode='markers+text' if (show_labels and label_col in df_valid.columns) else 'markers',
                        name=cat,
                        text=category_labels if (show_labels and label_col in df_valid.columns) else None,
                        textposition="top center",
                        textfont=dict(size=8),
                        marker=dict(
                            size=8,
                            color=colors_palette[idx % len(colors_palette)],
                            opacity=0.7,
                            line=dict(width=0.5, color='white')
                        ),
                        hovertext=hover_texts,
                        hoverinfo='text',
                    ))
            else:
                # Plot without color categories
                fig.add_trace(go.Scatter(
                    x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    mode='markers+text' if (show_labels and label_col in df_valid.columns) else 'markers',
                    text=df_valid[label_col].values if (show_labels and label_col in df_valid.columns) else None,
                    textposition="top center",
                    textfont=dict(size=8),
                    marker=dict(
                        size=8,
                        color='blue',
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    hovertext=hover_texts,
                    hoverinfo='text',
                    showlegend=False
                ))
            
            # Add to report
            fig.update_layout(
                title=title,
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
                # width=figsize[0] * 100,
                width=self.report.default_width,
                # height=figsize[1] * 100,
                height=self.report.default_height,
                hovermode='closest',
                template='plotly_white'
            )
        
        elif n_components == 3:
            # TODO: think about whether this is interesting
            raise ValueError("n_components for 3 dimension is not implemented yet")
        else:
            raise ValueError("n_components must be 2 or 3")
        

        # Save if requested
        if save:
            # path = os.path.join(self.output_directory, self.report.report_name.replace(".html", ""), ("pca" + (f'_{self.data_manager.current_decomposition}' if self.data_manager.current_decomposition is not None else "") + ".png"))
            path = os.path.join(self.outpout_image_directory, (f"{self.encoder_type}_pca" + (f'_{self.data_manager.current_decomposition}' if self.data_manager.current_decomposition is not None else "") + ".png"))
            fig.write_image(path)
            print(f"Interactive figure saved to {path}")

        # if show_plot:
        #     fig.show()

        self.report.add_figure(fig)

        return {
            "pca": pca,
            "X_pca": X_pca,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "scaler": scaler,
            "df_valid": df_valid,
            "fig": fig,  
        }

    # =============== T-SNE ================
    def visualize_embeddings_tsne(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        n_components: int = 2,
        color_by: str = "part_of_speech",
        label_col: str = "Word",
        show_labels: bool = False,
        max_labels: int = 50,
        figsize: tuple = (12, 8),
        title: str = None,
        save_path: str = None,
        # t-SNE specific parameters
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: int = 42,
        metric: str = "cosine",
        show_plot: bool = True,
        save: bool = False,
    ):
        print(f"T-SNE")
        if df is None:
            assert self.data_manager.get_current_dataframe() is not None
            df = self.data_manager.get_current_dataframe()

        # Check embeddings
        df_valid = df[df[embedding_col].notna()].copy().reset_index(drop=True)
        if len(df_valid) == 0:
            raise ValueError("No valid embeddings found in the dataframe")

        embeddings_aggregated = df_valid[embedding_col].apply(
            lambda x: self.aggregate_embeddings(x, method=aggregation_method)
        )

        X = np.vstack(embeddings_aggregated.values)
        print(f"Embedding matrix shape: {X.shape}")

        # make mean=0 and std=1
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Adjust perplexity if needed (must be less than n_samples)
        effective_perplexity = min(perplexity, len(X_scaled) - 1)
        if effective_perplexity != perplexity:
            print(f"Something most likely went wrong. Adjusted perplexity from {perplexity} to {effective_perplexity} (must be < n_samples)")

        # Apply t-SNE
        print(f"Running t-SNE with perplexity={effective_perplexity}, n_iter={n_iter}...")
        tsne = TSNE(
            n_components=n_components,
            perplexity=effective_perplexity,
            learning_rate=learning_rate,
            max_iter=n_iter,
            random_state=random_state,
            metric=metric,
            init="pca",
        )
        X_tsne = tsne.fit_transform(X_scaled)
        print(f"t-SNE complete. Final KL divergence: {tsne.kl_divergence_:.4f}")

        # Prepare labels and colors
        labels = df_valid[label_col].values if label_col in df_valid.columns else None

        color_values = None
        if color_by == "word_length":
            length_source_col = label_col if label_col in df_valid.columns else None
            if length_source_col is None and "Word" in df_valid.columns:
                length_source_col = "Word"
            if length_source_col is not None:
                color_values = df_valid[length_source_col].astype(str).str.len()
        elif color_by and color_by in df_valid.columns:
            color_values = df_valid[color_by]

        has_color = color_values is not None
        is_continuous_color = has_color and pd.api.types.is_numeric_dtype(color_values)

        hover_texts = []
        if label_col in df_valid.columns:
            for idx, row in df_valid.iterrows():
                hover_text = f"{label_col}: {row[label_col]}"
                if has_color:
                    hover_text += f"<br>{color_by}: {color_values.iloc[idx]}"
                hover_texts.append(hover_text)
        else:
            hover_texts = [f"Point {i}" for i in range(len(df_valid))]

        if n_components == 2:
            fig = go.Figure()

            if has_color:
                if is_continuous_color:
                    fig.add_trace(go.Scatter(
                        x=X_tsne[:, 0],
                        y=X_tsne[:, 1],
                        mode="markers",
                        hovertext=hover_texts,
                        hoverinfo='text',
                        marker=dict(
                            size=8,
                            opacity=0.8,
                            color=color_values.to_numpy(),
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title=color_by),
                            line=dict(width=0.5, color="white"),
                        ),
                        showlegend=False,
                    ))
                else:
                    # pandas 3 infers this column as `str` dtype, so astype(str) is a
                    # no-op and missing values stay float NaN -- which breaks the sort
                    # below and silently empties their mask. Name them instead.
                    categories = color_values.fillna("unknown").astype(str).values
                    unique_categories = sorted(set(categories))

                    for cat in unique_categories:
                        mask = categories == cat

                        fig.add_trace(go.Scatter(
                            x=X_tsne[mask, 0],
                            y=X_tsne[mask, 1],
                            mode="markers",
                            name=cat,
                            hovertext=np.array(hover_texts)[mask],
                            hoverinfo='text',
                            marker=dict(
                                size=8,
                                opacity=0.7,
                                line=dict(width=0.5, color="white"),
                            ),
                        ))
            else:
                fig.add_trace(go.Scatter(
                    x=X_tsne[:, 0],
                    y=X_tsne[:, 1],
                    mode="markers",
                    hovertext=hover_texts,
                    hoverinfo='text',
                    marker=dict(
                        size=8,
                        opacity=0.7,
                        line=dict(width=0.5, color="white"),
                    ),
                    showlegend=False,
                ))

            # Add text labels for a random subset
            # if show_labels and labels is not None:
            #     indices = np.random.choice(
            #         len(labels),
            #         size=min(max_labels, len(labels)),
            #         replace=False,
            #     )
            #     fig.add_trace(go.Scatter(
            #         x=X_tsne[indices, 0],
            #         y=X_tsne[indices, 1],
            #         mode="text",
            #         text=labels[indices],
            #         textposition="top center",
            #         textfont=dict(size=10),
            #         showlegend=False,
            #         hoverinfo="skip",
            #     ))

        elif n_components == 3:
            raise ValueError("n_components = 3 is not implemented yet")
        else:
            raise ValueError("n_components must be 2 or 3")

        # Title
        if title is None:
            title = f"t-SNE Visualization of {embedding_col} (perplexity={effective_perplexity})"

        fig.update_layout(
            title=title,
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            legend_title_text=color_by if (has_color and not is_continuous_color) else None,
            width=self.report.default_width,
            height=self.report.default_height,
            template="plotly_white"
        )

        # Save if requested
        self.report.add_figure(fig)
        if save:
            path = os.path.join(self.outpout_image_directory, (f"{self.encoder_type}_tsne" + (f'_{self.data_manager.current_decomposition}' if self.data_manager.current_decomposition is not None else "") + ".png"))
            fig.write_image(path)
            print(f"Interactive figure saved to {path}")
        # if save_path:
        #     fig.write_image(save_path, scale=2)
        #     print(f"Figure saved to {save_path}")

        # if show_plot:
        #     fig.show()

        return {
            "tsne": tsne,
            "X_tsne": X_tsne,
            "kl_divergence": tsne.kl_divergence_,
            "scaler": scaler,
            "df_valid": df_valid,
            "perplexity": effective_perplexity,
            "fig": fig,
        }

    # ================ UMAP ================
    def visualize_embeddings_umap(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        n_components: int = 2,
        color_by: str = "part_of_speech",
        label_col: str = "lemma",
        show_labels: bool = False,
        max_labels: int = 50,
        figsize: tuple = (12, 8),
        title: str = None,
        save_path: str = None,
        # UMAP specific parameters
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
        show_plot: bool = True,
    ):
        if umap is None:
            raise ImportError(
                "UMAP support requires the 'umap-learn' package. "
                "Install it with `pip install umap-learn`."
            )

        if df is None:
            assert self.current_dataframe is not None
            df = self.current_dataframe

        # Filter out rows with missing embeddings
        df_valid = df[df[embedding_col].notna()].copy()

        if len(df_valid) == 0:
            raise ValueError("No valid embeddings found in the dataframe")

        # Aggregate embeddings
        embeddings_aggregated = df_valid[embedding_col].apply(
            lambda x: self.aggregate_embeddings(x, method=aggregation_method)
        )

        # Stack into matrix
        X = np.vstack(embeddings_aggregated.values)
        print(f"Embedding matrix shape: {X.shape}")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Adjust n_neighbors if needed
        effective_n_neighbors = min(n_neighbors, len(X_scaled) - 1)
        if effective_n_neighbors != n_neighbors:
            print(f"Adjusted n_neighbors from {n_neighbors} to {effective_n_neighbors} (must be < n_samples)")

        # Apply UMAP
        print(f"Running UMAP with n_neighbors={effective_n_neighbors}, min_dist={min_dist}...")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=effective_n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        X_umap = reducer.fit_transform(X_scaled)
        print("UMAP complete.")

        # Prepare labels and categories
        labels = df_valid[label_col].values if label_col in df_valid.columns else None
        has_color = color_by and color_by in df_valid.columns

        if n_components == 2:
            fig = go.Figure()

            if has_color:
                # pandas 3 infers this column as `str` dtype, so astype(str) is a
                # no-op and missing values stay float NaN -- which breaks the sort
                # below and silently empties their mask. Name them instead.
                categories = df_valid[color_by].fillna("unknown").astype(str).values
                unique_categories = sorted(set(categories))

                for cat in unique_categories:
                    mask = categories == cat
                    hover_text = labels[mask] if labels is not None else None

                    fig.add_trace(go.Scatter(
                        x=X_umap[mask, 0],
                        y=X_umap[mask, 1],
                        mode="markers",
                        name=cat,
                        text=hover_text,
                        hovertemplate=(
                            f"<b>%{{text}}</b><br>"
                            f"{color_by}: {cat}<br>"
                            "UMAP 1: %{x:.2f}<br>"
                            "UMAP 2: %{y:.2f}"
                            "<extra></extra>"
                        ),
                        marker=dict(
                            size=8,
                            opacity=0.7,
                            line=dict(width=0.5, color="white"),
                        ),
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=X_umap[:, 0],
                    y=X_umap[:, 1],
                    mode="markers",
                    text=labels,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "UMAP 1: %{x:.2f}<br>"
                        "UMAP 2: %{y:.2f}"
                        "<extra></extra>"
                    ),
                    marker=dict(
                        size=8,
                        opacity=0.7,
                        line=dict(width=0.5, color="white"),
                    ),
                    showlegend=False,
                ))

            # Add text labels for a random subset
            if show_labels and labels is not None:
                indices = np.random.choice(
                    len(labels),
                    size=min(max_labels, len(labels)),
                    replace=False,
                )
                fig.add_trace(go.Scatter(
                    x=X_umap[indices, 0],
                    y=X_umap[indices, 1],
                    mode="text",
                    text=labels[indices],
                    textposition="top center",
                    textfont=dict(size=10),
                    showlegend=False,
                    hoverinfo="skip",
                ))

        elif n_components == 3:
            raise ValueError("3 not implemented yet")
        else:
            raise ValueError("n_components must be 2 or 3")

        # Title
        if title is None:
            title = f"UMAP Visualization of {embedding_col} (n_neighbors={effective_n_neighbors}, min_dist={min_dist})"

        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            legend_title_text=color_by if has_color else None,
            template="plotly_white",
            width=figsize[0] * 80,
            height=figsize[1] * 80,
        )

        # Save if requested
        self.report.add_figure(fig)
        # if save_path:
            # fig.write_image(save_path, scale=2)
            # print(f"Figure saved to {save_path}")

        if show_plot:
            fig.show()

        return {
            "umap": reducer,
            "X_umap": X_umap,
            "scaler": scaler,
            "df_valid": df_valid,
            "n_neighbors": effective_n_neighbors,
            "min_dist": min_dist,
            "fig": fig,
        }



    # ======================================
    # ========= Neighbour Analysis =========
    # ======================================
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
        find_if_csv_file_present: bool = False
    ):
        output_path = os.path.join(self.data_directory, f'{self.dataset_name}_{self.encoder_type}{"_" + str(self.current_subsample) if self.current_subsample else ""}_with_neighbours.csv')
        if find_if_csv_file_present is False and os.path.exists(output_path):
            self.current_dataframe_with_neighbors = pd.read_csv(output_path)
            return

        if df is None:
            assert self.current_dataframe is not None
            df = self.current_dataframe.copy()


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
        df_export.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        print(f"Columns: {list(df_export.columns)}")
        
        self.current_dataframe_with_neighbors = df_result
        return df_result

    def analyze_neighbors(self, df_with_neighbors: pd.DataFrame = None, label_col: str = "Word"):
        """
        Analyze the nearest neighbor results.
        """
        if df_with_neighbors is None:
            assert self.current_dataframe_with_neighbors is not None
            df_with_neighbors = self.current_dataframe_with_neighbors

        # Get neighbor columns
        neighbor_cols = [col for col in df_with_neighbors.columns if col.startswith("neighbor_") and "similarity" not in col and "list" not in col]
        similarity_cols = [col for col in df_with_neighbors.columns if "similarity" in col and "list" not in col]
        
        print(f"Number of words: {len(df_with_neighbors)}")
        self.report.print(f"Number of words: {len(df_with_neighbors)}")
        print(f"Number of neighbors per word: {len(neighbor_cols)}")
        self.report.print(f"Number of neighbors per word: {len(neighbor_cols)}")
        
        if similarity_cols:
            # Average similarity statistics
            print("\nSimilarity Statistics:")
            for col in similarity_cols:
                values = df_with_neighbors[col].dropna()
                print(f"  {col}: mean={values.mean():.4f}, std={values.std():.4f}, min={values.min():.4f}, max={values.max():.4f}")
        
        # Check for reciprocal neighbors (A is neighbor of B and B is neighbor of A)
        print("\nChecking for reciprocal neighbors...")
        reciprocal_count = 0
        reciprocal_pairs = []
        
        labels = df_with_neighbors[label_col].values
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        for idx, row in df_with_neighbors.iterrows():
            word = row[label_col]
            neighbors = [row[col] for col in neighbor_cols if pd.notna(row[col])]
            
            for neighbor in neighbors:
                if neighbor in label_to_idx:
                    neighbor_idx = label_to_idx[neighbor]
                    neighbor_row = df_with_neighbors.iloc[neighbor_idx]
                    neighbor_neighbors = [neighbor_row[col] for col in neighbor_cols if pd.notna(neighbor_row[col])]
                    
                    if word in neighbor_neighbors:
                        if (neighbor, word) not in reciprocal_pairs:
                            reciprocal_pairs.append((word, neighbor))
                            reciprocal_count += 1
        
        print(f"  Found {len(reciprocal_pairs)} reciprocal neighbor pairs")
        if reciprocal_pairs[:5]:
            print(f"  Examples: {reciprocal_pairs[:5]}")
        
        return {
            "n_words": len(df_with_neighbors),
            "n_neighbors": len(neighbor_cols),
            "reciprocal_pairs": reciprocal_pairs,
        }




    # ======================================
    # ======= Morphological Analysis =======
    # ======================================
    # Possible Delete
    def analyze_morphology(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        label_col: str = "Word",
        min_prefix_length: int = 3,
        min_suffix_length: int = 3,
        min_words_per_affix: int = 5,
    ):
        if df is None:
            assert self.current_dataframe is not None
            df = self.current_dataframe

        df_valid = df[df[embedding_col].notna()].copy()
        embeddings = df_valid[embedding_col].apply(
            lambda x: self.aggregate_embeddings(x, method=aggregation_method)
        )
        X = np.vstack(embeddings.values)
        labels = df_valid[label_col].values
        
        print("=" * 60)
        print("MORPHOLOGICAL ANALYSIS")
        print("=" * 60)
        
        # Group words by prefix
        prefix_groups = defaultdict(list)
        suffix_groups = defaultdict(list)
        
        for idx, word in enumerate(labels):
            word_str = str(word)
            if len(word_str) >= min_prefix_length:
                prefix = word_str[:min_prefix_length]
                prefix_groups[prefix].append(idx)
            if len(word_str) >= min_suffix_length:
                suffix = word_str[-min_suffix_length:]
                suffix_groups[suffix].append(idx)
        
        # Filter groups with enough words
        prefix_groups = {k: v for k, v in prefix_groups.items() if len(v) >= min_words_per_affix}
        suffix_groups = {k: v for k, v in suffix_groups.items() if len(v) >= min_words_per_affix}
        
        def analyze_affix_groups(groups, affix_type):
            within_sims = []
            between_sims = []
            
            group_list = list(groups.items())
            
            for affix, indices in group_list:
                if len(indices) < 2:
                    continue
                
                # Within-group similarity
                group_embeddings = X[indices]
                sim_matrix = cosine_similarity(group_embeddings)
                np.fill_diagonal(sim_matrix, np.nan)
                within_sims.extend(sim_matrix[~np.isnan(sim_matrix)].tolist())
            
            # Between-group similarity (sample)
            all_indices = [idx for indices in groups.values() for idx in indices]
            sample_size = min(1000, len(all_indices))
            sample_indices = np.random.choice(all_indices, size=sample_size, replace=False)
            
            for i in range(len(sample_indices)):
                for j in range(i + 1, min(i + 10, len(sample_indices))):
                    idx_i, idx_j = sample_indices[i], sample_indices[j]
                    # Check if from different groups
                    affix_i = None
                    affix_j = None
                    for affix, indices in groups.items():
                        if idx_i in indices:
                            affix_i = affix
                        if idx_j in indices:
                            affix_j = affix
                    if affix_i != affix_j:
                        sim = cosine_similarity([X[idx_i]], [X[idx_j]])[0][0]
                        between_sims.append(sim)
            
            return within_sims, between_sims
        
        # Analyze prefixes
        self.report.print(f"\nPrefix Analysis ({len(prefix_groups)} groups with {min_words_per_affix}+ words):")
        if prefix_groups:
            within_prefix, between_prefix = analyze_affix_groups(prefix_groups, "prefix")
            self.report.print(f"- Within-group similarity:  mean={np.mean(within_prefix):.4f}")
            self.report.print(f"- Between-group similarity: mean={np.mean(between_prefix):.4f}")
            self.report.print(f"- Difference: {np.mean(within_prefix) - np.mean(between_prefix):+.4f}")
            
            # Top prefixes
            self.report.print(f"\n  Top prefixes by group size:")
            for prefix, indices in sorted(prefix_groups.items(), key=lambda x: -len(x[1]))[:10]:
                words = [labels[i] for i in indices[:5]]
                self.report.print(f"    '{prefix}': {len(indices)} words (e.g., {words})")
        
        # Analyze suffixes
        self.report.print(f"\nSuffix Analysis ({len(suffix_groups)} groups with {min_words_per_affix}+ words):")
        if suffix_groups:
            within_suffix, between_suffix = analyze_affix_groups(suffix_groups, "suffix")
            self.report.print(f"  Within-group similarity:  mean={np.mean(within_suffix):.4f}")
            self.report.print(f"  Between-group similarity: mean={np.mean(between_suffix):.4f}")
            self.report.print(f"  Difference: {np.mean(within_suffix) - np.mean(between_suffix):+.4f}")
            
            # Top suffixes
            self.report.print(f"\n  Top suffixes by group size:")
            for suffix, indices in sorted(suffix_groups.items(), key=lambda x: -len(x[1]))[:10]:
                words = [labels[i] for i in indices[:5]]
                self.report.print(f"    '{suffix}': {len(indices)} words (e.g., {words})")
        
        return {
            "prefix_groups": prefix_groups,
            "suffix_groups": suffix_groups,
        }

    # Required
    def analyze_morphological_neighbors(
        self,
        df_with_neighbors: pd.DataFrame = None,
        label_col: str = "Word",
        prefix_lengths: list = [2, 3, 4],
        suffix_lengths: list = [2, 3, 4],
        verbose: bool = True,
    ):
        if df_with_neighbors is None:
            assert self.data_manager.get_current_dataframe_with_neighbours() is not None
            df_with_neighbors = self.data_manager.get_current_dataframe_with_neighbours()

        # df_with_neighbors = df_with_neighbors[df_with_neighbors[embedding_col].notna()].copy().reset_index(drop=True)

        # Get neighbor columns
        neighbor_cols = [
            col for col in df_with_neighbors.columns 
            if col.startswith("neighbor_") and "similarity" not in col and "list" not in col
        ]
        
        if not neighbor_cols:
            raise ValueError("No neighbor columns found in dataframe")
        
        all_words = df_with_neighbors[label_col].astype(str).values
        
        if verbose:
            print("=" * 70)
            print("MORPHOLOGICAL NEIGHBOR ANALYSIS")
            print("=" * 70)
        
        results = {
            "prefix": {},
            "suffix": {},
        }
        
        # ==========================================================================
        # PREFIX ANALYSIS
        # ==========================================================================
        if verbose:
            print("\n" + "=" * 70)
            print("PREFIX ANALYSIS")
            print("=" * 70)
        
        for prefix_len in prefix_lengths:
            if verbose:
                print(f"\n{'-' * 70}")
                print(f"PREFIX LENGTH: {prefix_len}")
                print(f"{'-' * 70}")
            
            # Calculate baseline: for each prefix, what % of all words have it?
            prefix_counts = Counter()
            for word in all_words:
                try:
                    if len(word) >= prefix_len:
                        prefix_counts[word[:prefix_len]] += 1
                except:
                    print("LEN ERROR")
                    print(word)

            # For each word, calculate % of neighbors with same prefix
            same_prefix_percentages = []
            word_details = []
            
            for idx, row in df_with_neighbors.iterrows():
                word = str(row[label_col])
                
                if len(word) < prefix_len:
                    continue
                
                word_prefix = word[:prefix_len]
                
                # Get neighbors
                neighbors = [str(row[col]) for col in neighbor_cols if pd.notna(row[col])]
                
                if not neighbors:
                    continue
                
                # Count neighbors with same prefix
                neighbors_same_prefix = [
                    n for n in neighbors 
                    if len(n) >= prefix_len and n[:prefix_len] == word_prefix
                ]
                
                percentage = 100 * len(neighbors_same_prefix) / len(neighbors)
                same_prefix_percentages.append(percentage)
                
                # Calculate expected percentage (baseline)
                expected_pct = 100 * prefix_counts[word_prefix] / len(all_words)
                
                word_details.append({
                    "word": word,
                    "prefix": word_prefix,
                    "n_neighbors": len(neighbors),
                    "n_same_prefix": len(neighbors_same_prefix),
                    "percentage": percentage,
                    "expected_percentage": expected_pct,
                    "neighbors_same_prefix": neighbors_same_prefix,
                })
            
            # Calculate statistics
            avg_same_prefix_pct = np.mean(same_prefix_percentages) if same_prefix_percentages else 0
            std_same_prefix_pct = np.std(same_prefix_percentages) if same_prefix_percentages else 0
            
            # Calculate expected baseline (average expected percentage)
            expected_percentages = [d["expected_percentage"] for d in word_details]
            avg_expected_pct = np.mean(expected_percentages) if expected_percentages else 0
            
            # Lift: how much more likely than random?
            lift = avg_same_prefix_pct / avg_expected_pct if avg_expected_pct > 0 else 0
            
            if verbose:
                print(f"\nWords analyzed: {len(same_prefix_percentages)}")
                print(f"Unique prefixes: {len(prefix_counts)}")
                print(f"\nResults:")
                print(f"  Avg % neighbors with same prefix: {avg_same_prefix_pct:.2f}% ± {std_same_prefix_pct:.2f}%")
                print(f"  Expected % (random baseline):     {avg_expected_pct:.2f}%")
                print(f"  Lift over baseline:               {lift:.2f}x")
                
                if lift > 1.5:
                    print(f"\n  ✓ Words are {lift:.1f}x MORE likely to have neighbors with the same prefix")
                elif lift < 0.7:
                    print(f"\n  ✗ Words are {1/lift:.1f}x LESS likely to have neighbors with the same prefix")
                else:
                    print(f"\n  ≈ No strong preference for neighbors with the same prefix")
            
            # Breakdown by prefix
            prefix_stats = defaultdict(list)
            for detail in word_details:
                prefix_stats[detail["prefix"]].append(detail["percentage"])
            
            # Find prefixes with highest neighbor retention
            prefix_summary = []
            for prefix, percentages in prefix_stats.items():
                if len(percentages) >= 5:  # Only prefixes with enough samples
                    prefix_summary.append({
                        "prefix": prefix,
                        "n_words": len(percentages),
                        "avg_same_prefix_neighbors": np.mean(percentages),
                        "std": np.std(percentages),
                    })
            
            prefix_summary.sort(key=lambda x: -x["avg_same_prefix_neighbors"])
            
            if verbose and prefix_summary:
                print(f"\n  Top 10 prefixes (highest % of same-prefix neighbors):")
                for item in prefix_summary[:10]:
                    print(f"    '{item['prefix']}': {item['avg_same_prefix_neighbors']:.1f}% ± {item['std']:.1f}% "
                        f"(n={item['n_words']})")
                
                print(f"\n  Bottom 10 prefixes (lowest % of same-prefix neighbors):")
                for item in prefix_summary[-10:]:
                    print(f"    '{item['prefix']}': {item['avg_same_prefix_neighbors']:.1f}% ± {item['std']:.1f}% "
                        f"(n={item['n_words']})")
            
            # Examples
            if verbose and word_details:
                sorted_details = sorted(word_details, key=lambda x: -x["percentage"])
                
                print(f"\n  Examples - highest same-prefix neighbor %:")
                for detail in sorted_details[:5]:
                    print(f"    '{detail['word']}' (prefix='{detail['prefix']}'): "
                        f"{detail['percentage']:.0f}% ({detail['n_same_prefix']}/{detail['n_neighbors']})")
                    if detail['neighbors_same_prefix'][:3]:
                        print(f"      → {detail['neighbors_same_prefix'][:3]}")
            
            results["prefix"][prefix_len] = {
                "avg_same_prefix_pct": avg_same_prefix_pct,
                "std_same_prefix_pct": std_same_prefix_pct,
                "avg_expected_pct": avg_expected_pct,
                "lift": lift,
                "n_words_analyzed": len(same_prefix_percentages),
                "prefix_summary": prefix_summary,
                "word_details": word_details,
            }
        
        # ==========================================================================
        # SUFFIX ANALYSIS
        # ==========================================================================
        if verbose:
            print("\n\n" + "=" * 70)
            print("SUFFIX ANALYSIS")
            print("=" * 70)
        
        for suffix_len in suffix_lengths:
            if verbose:
                print(f"\n{'-' * 70}")
                print(f"SUFFIX LENGTH: {suffix_len}")
                print(f"{'-' * 70}")
            
            # Calculate baseline
            suffix_counts = Counter()
            for word in all_words:
                if len(word) >= suffix_len:
                    suffix_counts[word[-suffix_len:]] += 1
            
            # For each word, calculate % of neighbors with same suffix
            same_suffix_percentages = []
            word_details = []
            
            for idx, row in df_with_neighbors.iterrows():
                word = str(row[label_col])
                
                if len(word) < suffix_len:
                    continue
                
                word_suffix = word[-suffix_len:]
                
                neighbors = [str(row[col]) for col in neighbor_cols if pd.notna(row[col])]
                
                if not neighbors:
                    continue
                
                neighbors_same_suffix = [
                    n for n in neighbors 
                    if len(n) >= suffix_len and n[-suffix_len:] == word_suffix
                ]
                
                percentage = 100 * len(neighbors_same_suffix) / len(neighbors)
                same_suffix_percentages.append(percentage)
                
                expected_pct = 100 * suffix_counts[word_suffix] / len(all_words)
                
                word_details.append({
                    "word": word,
                    "suffix": word_suffix,
                    "n_neighbors": len(neighbors),
                    "n_same_suffix": len(neighbors_same_suffix),
                    "percentage": percentage,
                    "expected_percentage": expected_pct,
                    "neighbors_same_suffix": neighbors_same_suffix,
                })
            
            avg_same_suffix_pct = np.mean(same_suffix_percentages) if same_suffix_percentages else 0
            std_same_suffix_pct = np.std(same_suffix_percentages) if same_suffix_percentages else 0
            
            expected_percentages = [d["expected_percentage"] for d in word_details]
            avg_expected_pct = np.mean(expected_percentages) if expected_percentages else 0
            
            lift = avg_same_suffix_pct / avg_expected_pct if avg_expected_pct > 0 else 0
            
            if verbose:
                print(f"\nWords analyzed: {len(same_suffix_percentages)}")
                print(f"Unique suffixes: {len(suffix_counts)}")
                print(f"\nResults:")
                print(f"  Avg % neighbors with same suffix: {avg_same_suffix_pct:.2f}% ± {std_same_suffix_pct:.2f}%")
                print(f"  Expected % (random baseline):     {avg_expected_pct:.2f}%")
                print(f"  Lift over baseline:               {lift:.2f}x")
                
                if lift > 1.5:
                    print(f"\n  ✓ Words are {lift:.1f}x MORE likely to have neighbors with the same suffix")
                elif lift < 0.7:
                    print(f"\n  ✗ Words are {1/lift:.1f}x LESS likely to have neighbors with the same suffix")
                else:
                    print(f"\n  ≈ No strong preference for neighbors with the same suffix")
            
            # Breakdown by suffix
            suffix_stats = defaultdict(list)
            for detail in word_details:
                suffix_stats[detail["suffix"]].append(detail["percentage"])
            
            suffix_summary = []
            for suffix, percentages in suffix_stats.items():
                if len(percentages) >= 5:
                    suffix_summary.append({
                        "suffix": suffix,
                        "n_words": len(percentages),
                        "avg_same_suffix_neighbors": np.mean(percentages),
                        "std": np.std(percentages),
                    })
            
            suffix_summary.sort(key=lambda x: -x["avg_same_suffix_neighbors"])
            
            if verbose and suffix_summary:
                print(f"\n  Top 10 suffixes (highest % of same-suffix neighbors):")
                for item in suffix_summary[:10]:
                    print(f"    '{item['suffix']}': {item['avg_same_suffix_neighbors']:.1f}% ± {item['std']:.1f}% "
                        f"(n={item['n_words']})")
                
                print(f"\n  Bottom 10 suffixes (lowest % of same-suffix neighbors):")
                for item in suffix_summary[-10:]:
                    print(f"    '{item['suffix']}': {item['avg_same_suffix_neighbors']:.1f}% ± {item['std']:.1f}% "
                        f"(n={item['n_words']})")
            
            if verbose and word_details:
                sorted_details = sorted(word_details, key=lambda x: -x["percentage"])
                
                print(f"\n  Examples - highest same-suffix neighbor %:")
                for detail in sorted_details[:5]:
                    print(f"    '{detail['word']}' (suffix='{detail['suffix']}'): "
                        f"{detail['percentage']:.0f}% ({detail['n_same_suffix']}/{detail['n_neighbors']})")
                    if detail['neighbors_same_suffix'][:3]:
                        print(f"      → {detail['neighbors_same_suffix'][:3]}")
            
            results["suffix"][suffix_len] = {
                "avg_same_suffix_pct": avg_same_suffix_pct,
                "std_same_suffix_pct": std_same_suffix_pct,
                "avg_expected_pct": avg_expected_pct,
                "lift": lift,
                "n_words_analyzed": len(same_suffix_percentages),
                "suffix_summary": suffix_summary,
                "word_details": word_details,
            }
        
        # ==========================================================================
        # SUMMARY COMPARISON
        # ==========================================================================
        if verbose:
            print("\n\n" + "=" * 70)
            print("SUMMARY: PREFIX VS SUFFIX INFLUENCE")
            print("=" * 70)
            
            print(f"\n{'Length':<10} {'Prefix Lift':<15} {'Suffix Lift':<15} {'Stronger?':<15}")
            print("-" * 55)
            
            for length in sorted(set(prefix_lengths) & set(suffix_lengths)):
                prefix_lift = results["prefix"].get(length, {}).get("lift", 0)
                suffix_lift = results["suffix"].get(length, {}).get("lift", 0)
                
                if prefix_lift > suffix_lift * 1.2:
                    stronger = "PREFIX"
                elif suffix_lift > prefix_lift * 1.2:
                    stronger = "SUFFIX"
                else:
                    stronger = "EQUAL"
                
                print(f"{length:<10} {prefix_lift:<15.2f} {suffix_lift:<15.2f} {stronger:<15}")
            
            print("\nInterpretation:")
            print("  Lift > 1.0: Words prefer neighbors with same affix more than random")
            print("  Lift > 2.0: Strong morphological clustering in embedding space")
            print("  Lift < 1.0: Words avoid neighbors with same affix")
        
        return results

    # Required
    def plot_morphological_neighbor_percentages(
        self,
        results: dict,
        save: bool = True
    ):
        lengths = sorted(set(list(results["prefix"].keys()) + list(results["suffix"].keys())))

        prefix_pcts = [results["prefix"][l]["avg_same_prefix_pct"] for l in lengths if l in results["prefix"]]
        prefix_stds = [results["prefix"][l]["std_same_prefix_pct"] for l in lengths if l in results["prefix"]]
        suffix_pcts = [results["suffix"][l]["avg_same_suffix_pct"] for l in lengths if l in results["suffix"]]
        suffix_stds = [results["suffix"][l]["std_same_suffix_pct"] for l in lengths if l in results["suffix"]]

        x_labels = [f"{l}-char" for l in lengths]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Prefix",
            x=x_labels,
            y=prefix_pcts,
            # error_y=dict(type="data", array=prefix_stds, visible=True),
            marker_color="#378ADD",
        ))

        fig.add_trace(go.Bar(
            name="Suffix",
            x=x_labels,
            y=suffix_pcts,
            # error_y=dict(type="data", array=suffix_stds, visible=True),
            marker_color="#1D9E75",
        ))

        fig.update_layout(
            barmode="group",
            title="% of neighbors sharing the same prefix / suffix",
            xaxis_title="Affix length",
            yaxis_title="Avg % of neighbors",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="sans-serif", size=13),
            height=self.report.default_height,
            width=self.report.default_width,
            template="plotly_white",
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#E5E5E5")

        # Save if requested
        if save:
            path = os.path.join(self.outpout_image_directory, (f"{self.encoder_type}_morphological_neighbor_percentage" + (f'_{self.data_manager.current_decomposition}' if self.data_manager.current_decomposition is not None else "") + ".png"))
            fig.write_image(path)
            print(f"Interactive figure saved to {path}")
        self.report.add_figure(fig)

        return fig

    # Required
    def analyze_morphological_neighbors_by_pos(
        self,
        df_with_neighbors: pd.DataFrame = None,
        label_col: str = "Word",
        pos_col: str = "part_of_speech",
        prefix_lengths: list = [2, 3, 4],
        suffix_lengths: list = [2, 3, 4],
        verbose: bool = True,
    ):
        if df_with_neighbors is None:
            assert self.data_manager.get_current_dataframe_with_neighbours() is not None
            df_with_neighbors = self.data_manager.get_current_dataframe_with_neighbours()
 
        neighbor_cols = [
            col for col in df_with_neighbors.columns
            if col.startswith("neighbor_") and "similarity" not in col and "list" not in col
        ]
 
        if not neighbor_cols:
            raise ValueError("No neighbor columns found in dataframe")
 
        if pos_col not in df_with_neighbors.columns:
            raise ValueError(f"POS column '{pos_col}' not found in dataframe")
 
        all_words = df_with_neighbors[label_col].astype(str).values
        pos_tags = sorted(df_with_neighbors[pos_col].dropna().astype(str).unique())
 
        if verbose:
            print("=" * 70)
            print("MORPHOLOGICAL NEIGHBOR ANALYSIS BY PART OF SPEECH")
            print("=" * 70)
            print(f"\nPOS tags found: {pos_tags}")
 
        results = {
            "prefix": {},
            "suffix": {},
        }
 
        # ==========================================================================
        # PREFIX ANALYSIS
        # ==========================================================================
        if verbose:
            print("\n" + "=" * 70)
            print("PREFIX ANALYSIS")
            print("=" * 70)
 
        for prefix_len in prefix_lengths:
            if verbose:
                print(f"\n{'-' * 70}")
                print(f"PREFIX LENGTH: {prefix_len}")
                print(f"{'-' * 70}")
 
            # Global baseline
            prefix_counts = Counter()
            for word in all_words:
                if len(word) >= prefix_len:
                    prefix_counts[word[:prefix_len]] += 1
 
            pos_results = {}
 
            for pos_tag in pos_tags:
                df_pos = df_with_neighbors[
                    df_with_neighbors[pos_col].astype(str) == pos_tag
                ]
 
                if len(df_pos) == 0:
                    continue
 
                same_prefix_percentages = []
                word_details = []
 
                for idx, row in df_pos.iterrows():
                    word = str(row[label_col])
 
                    if len(word) < prefix_len:
                        continue
 
                    word_prefix = word[:prefix_len]
                    neighbors = [str(row[col]) for col in neighbor_cols if pd.notna(row[col])]
 
                    if not neighbors:
                        continue
 
                    neighbors_same_prefix = [
                        n for n in neighbors
                        if len(n) >= prefix_len and n[:prefix_len] == word_prefix
                    ]
 
                    percentage = 100 * len(neighbors_same_prefix) / len(neighbors)
                    same_prefix_percentages.append(percentage)
 
                    expected_pct = 100 * prefix_counts[word_prefix] / len(all_words)
 
                    word_details.append({
                        "word": word,
                        "prefix": word_prefix,
                        "n_neighbors": len(neighbors),
                        "n_same_prefix": len(neighbors_same_prefix),
                        "percentage": percentage,
                        "expected_percentage": expected_pct,
                        "neighbors_same_prefix": neighbors_same_prefix,
                    })
 
                if not same_prefix_percentages:
                    continue
 
                avg_same_prefix_pct = np.mean(same_prefix_percentages)
                std_same_prefix_pct = np.std(same_prefix_percentages)
 
                expected_percentages = [d["expected_percentage"] for d in word_details]
                avg_expected_pct = np.mean(expected_percentages) if expected_percentages else 0
 
                lift = avg_same_prefix_pct / avg_expected_pct if avg_expected_pct > 0 else 0
 
                pos_results[pos_tag] = {
                    "avg_same_prefix_pct": avg_same_prefix_pct,
                    "std_same_prefix_pct": std_same_prefix_pct,
                    "avg_expected_pct": avg_expected_pct,
                    "lift": lift,
                    "n_words_analyzed": len(same_prefix_percentages),
                    "word_details": word_details,
                }
 
                if verbose:
                    print(f"\n  POS: {pos_tag}  (n={len(same_prefix_percentages)})")
                    print(f"    Avg % neighbors with same prefix: {avg_same_prefix_pct:.2f}% ± {std_same_prefix_pct:.2f}%")
                    print(f"    Expected % (random baseline):     {avg_expected_pct:.2f}%")
                    print(f"    Lift over baseline:               {lift:.2f}x")
 
                    if lift > 1.5:
                        print(f"    ✓ Words are {lift:.1f}x MORE likely to have neighbors with the same prefix")
                    elif lift < 0.7:
                        print(f"    ✗ Words are {1/lift:.1f}x LESS likely to have neighbors with the same prefix")
                    else:
                        print(f"    ≈ No strong preference for neighbors with the same prefix")
 
            results["prefix"][prefix_len] = pos_results
 
            if verbose and pos_results:
                print(f"\n  {'POS':<15} {'Avg %':<12} {'Expected %':<14} {'Lift':<10} {'N words':<10}")
                print(f"  {'-' * 61}")
                for pos_tag, pr in sorted(pos_results.items(), key=lambda x: -x[1]["lift"]):
                    print(
                        f"  {pos_tag:<15} {pr['avg_same_prefix_pct']:<12.2f} "
                        f"{pr['avg_expected_pct']:<14.2f} {pr['lift']:<10.2f} "
                        f"{pr['n_words_analyzed']:<10}"
                    )
 
        # ==========================================================================
        # SUFFIX ANALYSIS
        # ==========================================================================
        if verbose:
            print("\n\n" + "=" * 70)
            print("SUFFIX ANALYSIS")
            print("=" * 70)
 
        for suffix_len in suffix_lengths:
            if verbose:
                print(f"\n{'-' * 70}")
                print(f"SUFFIX LENGTH: {suffix_len}")
                print(f"{'-' * 70}")
 
            suffix_counts = Counter()
            for word in all_words:
                if len(word) >= suffix_len:
                    suffix_counts[word[-suffix_len:]] += 1
 
            pos_results = {}
 
            for pos_tag in pos_tags:
                df_pos = df_with_neighbors[
                    df_with_neighbors[pos_col].astype(str) == pos_tag
                ]
 
                if len(df_pos) == 0:
                    continue
 
                same_suffix_percentages = []
                word_details = []
 
                for idx, row in df_pos.iterrows():
                    word = str(row[label_col])
 
                    if len(word) < suffix_len:
                        continue
 
                    word_suffix = word[-suffix_len:]
                    neighbors = [str(row[col]) for col in neighbor_cols if pd.notna(row[col])]
 
                    if not neighbors:
                        continue
 
                    neighbors_same_suffix = [
                        n for n in neighbors
                        if len(n) >= suffix_len and n[-suffix_len:] == word_suffix
                    ]
 
                    percentage = 100 * len(neighbors_same_suffix) / len(neighbors)
                    same_suffix_percentages.append(percentage)
 
                    expected_pct = 100 * suffix_counts[word_suffix] / len(all_words)
 
                    word_details.append({
                        "word": word,
                        "suffix": word_suffix,
                        "n_neighbors": len(neighbors),
                        "n_same_suffix": len(neighbors_same_suffix),
                        "percentage": percentage,
                        "expected_percentage": expected_pct,
                        "neighbors_same_suffix": neighbors_same_suffix,
                    })
 
                if not same_suffix_percentages:
                    continue
 
                avg_same_suffix_pct = np.mean(same_suffix_percentages)
                std_same_suffix_pct = np.std(same_suffix_percentages)
 
                expected_percentages = [d["expected_percentage"] for d in word_details]
                avg_expected_pct = np.mean(expected_percentages) if expected_percentages else 0
 
                lift = avg_same_suffix_pct / avg_expected_pct if avg_expected_pct > 0 else 0
 
                pos_results[pos_tag] = {
                    "avg_same_suffix_pct": avg_same_suffix_pct,
                    "std_same_suffix_pct": std_same_suffix_pct,
                    "avg_expected_pct": avg_expected_pct,
                    "lift": lift,
                    "n_words_analyzed": len(same_suffix_percentages),
                    "word_details": word_details,
                }
 
                if verbose:
                    print(f"\n  POS: {pos_tag}  (n={len(same_suffix_percentages)})")
                    print(f"    Avg % neighbors with same suffix: {avg_same_suffix_pct:.2f}% ± {std_same_suffix_pct:.2f}%")
                    print(f"    Expected % (random baseline):     {avg_expected_pct:.2f}%")
                    print(f"    Lift over baseline:               {lift:.2f}x")
 
                    if lift > 1.5:
                        print(f"    ✓ Words are {lift:.1f}x MORE likely to have neighbors with the same suffix")
                    elif lift < 0.7:
                        print(f"    ✗ Words are {1/lift:.1f}x LESS likely to have neighbors with the same suffix")
                    else:
                        print(f"    ≈ No strong preference for neighbors with the same suffix")
 
            results["suffix"][suffix_len] = pos_results
 
            if verbose and pos_results:
                print(f"\n  {'POS':<15} {'Avg %':<12} {'Expected %':<14} {'Lift':<10} {'N words':<10}")
                print(f"  {'-' * 61}")
                for pos_tag, pr in sorted(pos_results.items(), key=lambda x: -x[1]["lift"]):
                    print(
                        f"  {pos_tag:<15} {pr['avg_same_suffix_pct']:<12.2f} "
                        f"{pr['avg_expected_pct']:<14.2f} {pr['lift']:<10.2f} "
                        f"{pr['n_words_analyzed']:<10}"
                    )
 
        # ==========================================================================
        # SUMMARY COMPARISON
        # ==========================================================================
        if verbose:
            print("\n\n" + "=" * 70)
            print("SUMMARY: PREFIX VS SUFFIX LIFT BY POS")
            print("=" * 70)
 
            for length in sorted(set(prefix_lengths) & set(suffix_lengths)):
                print(f"\n  Affix length: {length}")
                print(f"  {'POS':<15} {'Prefix Lift':<15} {'Suffix Lift':<15} {'Stronger?':<15}")
                print(f"  {'-' * 60}")
 
                all_pos_in_results = set(results["prefix"].get(length, {}).keys()) | set(
                    results["suffix"].get(length, {}).keys()
                )
                for pos_tag in sorted(all_pos_in_results):
                    prefix_lift = results["prefix"].get(length, {}).get(pos_tag, {}).get("lift", 0)
                    suffix_lift = results["suffix"].get(length, {}).get(pos_tag, {}).get("lift", 0)
 
                    if prefix_lift > suffix_lift * 1.2:
                        stronger = "PREFIX"
                    elif suffix_lift > prefix_lift * 1.2:
                        stronger = "SUFFIX"
                    else:
                        stronger = "EQUAL"
 
                    print(f"  {pos_tag:<15} {prefix_lift:<15.2f} {suffix_lift:<15.2f} {stronger:<15}")
 
            print("\nInterpretation:")
            print("  Lift > 1.0: This POS clusters more by affix than random chance")
            print("  Lift > 2.0: Strong morphological clustering for this POS")
            print("  Lift < 1.0: This POS avoids neighbors with the same affix")
 
        return results

    # Required
    def plot_morphological_neighbors_by_pos(
        self,
        results: dict,
        save: bool = True
    ):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
 
        prefix_lengths = sorted(results["prefix"].keys())
        suffix_lengths = sorted(results["suffix"].keys())
        lengths = sorted(set(prefix_lengths) & set(suffix_lengths))
 
        n_cols = len(lengths)
        fig = make_subplots(
            rows=1,
            cols=n_cols,
            subplot_titles=[f"{l}-char affix" for l in lengths],
            shared_yaxes=True,
        )
 
        # Collect all POS tags across all lengths for consistent ordering
        all_pos = sorted(set(
            pos
            for length in lengths
            for affix_type in ("prefix", "suffix")
            for pos in results[affix_type].get(length, {}).keys()
        ))
 
        prefix_color = "#378ADD"
        suffix_color = "#1D9E75"
        show_legend = True
 
        for col_idx, length in enumerate(lengths, start=1):
            prefix_data = results["prefix"].get(length, {})
            suffix_data = results["suffix"].get(length, {})
 
            pos_tags    = [p for p in all_pos if p in prefix_data or p in suffix_data]
            prefix_avgs = [prefix_data.get(p, {}).get("avg_same_prefix_pct", 0) for p in pos_tags]
            prefix_stds = [prefix_data.get(p, {}).get("std_same_prefix_pct", 0) for p in pos_tags]
            suffix_avgs = [suffix_data.get(p, {}).get("avg_same_suffix_pct", 0) for p in pos_tags]
            suffix_stds = [suffix_data.get(p, {}).get("std_same_suffix_pct", 0) for p in pos_tags]
 
            fig.add_trace(
                go.Bar(
                    name="Prefix",
                    x=pos_tags,
                    y=prefix_avgs,
                    # error_y=dict(type="data", array=prefix_stds, visible=True),
                    marker_color=prefix_color,
                    legendgroup="prefix",
                    showlegend=show_legend,
                ),
                row=1, col=col_idx,
            )
            fig.add_trace(
                go.Bar(
                    name="Suffix",
                    x=pos_tags,
                    y=suffix_avgs,
                    # error_y=dict(type="data", array=suffix_stds, visible=True),
                    marker_color=suffix_color,
                    legendgroup="suffix",
                    showlegend=show_legend,
                ),
                row=1, col=col_idx,
            )
            show_legend = False  # Only show legend entries once
 
        fig.update_layout(
            barmode="group",
            title="Avg % of neighbors sharing same affix — by POS and affix length",
            yaxis_title="Avg % of neighbors",
            yaxis=dict(range=[0, 40]),
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="sans-serif", size=13),
            height=self.report.default_height,
            width=self.report.default_width + 200,
            template="plotly_white",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#E5E5E5")

        if save:
            path = os.path.join(self.outpout_image_directory, (f"{self.encoder_type}_morphological_neighbor_by_pos" + (f'_{self.data_manager.current_decomposition}' if self.data_manager.current_decomposition is not None else "") + ".png"))
            fig.write_image(path)
            print(f"Interactive figure saved to {path}")
        self.report.add_figure(fig)
 
        return fig
  
    # Possible Delete
    def plot_affix_pos_signature(
        self,
        results: dict,
    ):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
 
        prefix_lengths = sorted(results["prefix"].keys())
        suffix_lengths = sorted(results["suffix"].keys())
        lengths = sorted(set(prefix_lengths) & set(suffix_lengths))
 
        # One row per affix type (prefix / suffix), one col per length
        n_cols = len(lengths)
        fig = make_subplots(
            rows=2,
            cols=n_cols,
            subplot_titles=(
                [f"Prefix {l}-char" for l in lengths] +
                [f"Suffix {l}-char" for l in lengths]
            ),
            horizontal_spacing=0.08,
            vertical_spacing=0.15,
        )
 
        # Build a stable color map over all dominant POS tags seen in the data
        all_pos = sorted(set(
            item["dominant_word_pos"]
            for length in lengths
            for affix_type, summary_key in [("prefix", "prefix_summary"), ("suffix", "suffix_summary")]
            for item in results[affix_type].get(length, {}).get(summary_key, [])
        ))
 
        palette = [
            "#378ADD", "#1D9E75", "#D85A30", "#BA7517",
            "#D4537E", "#7F77DD", "#639922", "#E24B4A",
            "#888780",
        ]
        pos_color_map = {pos: palette[i % len(palette)] for i, pos in enumerate(all_pos)}
 
        for col_idx, length in enumerate(lengths, start=1):
            for row_idx, (affix_type, summary_key) in enumerate(
                [("prefix", "prefix_summary"), ("suffix", "suffix_summary")], start=1
            ):
                summary = results[affix_type].get(length, {}).get(summary_key, [])
 
                if not summary:
                    continue
 
                # Group scatter points by dominant POS so each POS gets one legend entry
                pos_groups = defaultdict(lambda: {"x": [], "y": [], "text": [], "sizes": []})
                for item in summary:
                    affix_label = item.get("prefix") or item.get("suffix")
                    dominant_pos = item["dominant_word_pos"]
                    pos_groups[dominant_pos]["x"].append(item["dominant_word_pct"])
                    pos_groups[dominant_pos]["y"].append(item["neighbor_dominant_pct"])
                    pos_groups[dominant_pos]["sizes"].append(
                        6 + min(item["n_words"] / 3, 20)  # scale by word count, capped
                    )
                    pos_groups[dominant_pos]["text"].append(
                        f"'{affix_label}'<br>"
                        f"Dominant POS: {dominant_pos}<br>"
                        f"Word purity: {item['dominant_word_pct']:.1f}%<br>"
                        f"Neighbor purity: {item['neighbor_dominant_pct']:.1f}%<br>"
                        f"N words: {item['n_words']}"
                    )
 
                # Only emit legend entries from the first subplot
                show_legend = (col_idx == 1 and row_idx == 1)
 
                for pos_tag, group in pos_groups.items():
                    fig.add_trace(
                        go.Scatter(
                            x=group["x"],
                            y=group["y"],
                            mode="markers",
                            name=pos_tag,
                            marker=dict(
                                color=pos_color_map.get(pos_tag, "#888780"),
                                size=group["sizes"],
                                opacity=0.75,
                                line=dict(width=0.5, color="white"),
                            ),
                            text=group["text"],
                            hoverinfo="text",
                            legendgroup=pos_tag,
                            showlegend=show_legend,
                        ),
                        row=row_idx, col=col_idx,
                    )
 
                # Diagonal reference line: word purity == neighbor purity
                fig.add_trace(
                    go.Scatter(
                        x=[0, 100],
                        y=[0, 100],
                        mode="lines",
                        line=dict(color="#CCCCCC", width=1, dash="dash"),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row_idx, col=col_idx,
                )
 
                fig.update_xaxes(range=[0, 105], showgrid=True, gridcolor="#E5E5E5", row=row_idx, col=col_idx)
                fig.update_yaxes(range=[0, 105], showgrid=True, gridcolor="#E5E5E5", row=row_idx, col=col_idx)
 
        fig.update_layout(
            title="Affix POS signature — word purity % vs neighbor purity %",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="sans-serif", size=12),
            legend=dict(
                title="Dominant POS",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
            height=560,
        )
 
        # Shared axis labels via annotations
        fig.add_annotation(
            text="Word purity %",
            xref="paper", yref="paper",
            x=0.5, y=-0.07,
            showarrow=False,
            font=dict(size=12),
        )
        fig.add_annotation(
            text="Neighbor purity %",
            xref="paper", yref="paper",
            x=-0.055, y=0.5,
            showarrow=False,
            font=dict(size=12),
            textangle=-90,
        )
 
        return fig




    # ======================================
    # ========== Synonym Analysis ==========
    # ======================================

    # ===== Synonym Closeness Analysis =====
    # Required
    def analyze_synonyms(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        label_col: str = "Word",
        synonym_col: str = "synonyms",
        save: bool = True,
    ):
        """
        Basic check of how close synonyms are in latent space
        """
        # Load df from pipeline
        if df is None:
            assert self.data_manager.get_current_dataframe() is not None
            df = self.data_manager.get_current_dataframe()

        # Sanity checks
        df_valid = df[df[embedding_col].notna()].copy()
        
        if synonym_col not in df_valid.columns:
            print(f"Synonym column '{synonym_col}' not found")
            return None
        
        # Prepare embeddings
        embeddings = df_valid[embedding_col].apply(
            lambda x: self.aggregate_embeddings(x, method=aggregation_method)
        )
        X = np.vstack(embeddings.values)
        labels = df_valid[label_col].values
        
        word_to_idx = {word: idx for idx, word in enumerate(labels)}
        
        print("=" * 60)
        print("SYNONYM VALIDATION")
        print("=" * 60)
        
        synonym_similarities = []
        random_similarities = []
        
        for idx, row in df_valid.iterrows():
            word = row[label_col]
            word_idx = word_to_idx.get(word)
            word_emb = X[word_idx]
            
            synonyms_raw = row[synonym_col]
        
            if word_idx is None or pd.isna(synonyms_raw) or not synonyms_raw:
                continue
            
            # Parse synonyms (assuming comma-separated or list)
            if isinstance(synonyms_raw, str):
                synonyms = [s.strip() for s in synonyms_raw.split(",")]
            elif isinstance(synonyms_raw, list):
                synonyms = synonyms_raw
            else:
                continue
                
            
            # similarity with synonyms
            for syn in synonyms:
                syn = syn.strip()
                if syn in word_to_idx and syn != word:
                    syn_idx = word_to_idx[syn]
                    sim = cosine_similarity([word_emb], [X[syn_idx]])[0][0]
                    synonym_similarities.append(sim)
            
            # Calculate similarity with random words (for comparison)
            random_indices = np.random.choice(len(X), size=min(5, len(synonyms)), replace=False)
            for rand_idx in random_indices:
                if rand_idx != word_idx:
                    sim = cosine_similarity([word_emb], [X[rand_idx]])[0][0]
                    random_similarities.append(sim)
        
        if not synonym_similarities:
            print("No valid synonym pairs found in vocabulary")
            return None
        
        # Statistics
        syn_mean = np.mean(synonym_similarities)
        syn_std = np.std(synonym_similarities)
        rand_mean = np.mean(random_similarities)
        rand_std = np.std(random_similarities)
        
        print(f"\nSynonym pairs found: {len(synonym_similarities)}")
        print(f"\nCosine Similarity Statistics:")
        print(f"  Synonyms: mean={syn_mean:.4f}, std={syn_std:.4f}")
        print(f"  Random:   mean={rand_mean:.4f}, std={rand_std:.4f}")
        print(f"  Difference: {syn_mean - rand_mean:+.4f}")
        self.report.print(f"\nSynonym pairs found: {len(synonym_similarities)}")
        self.report.print(f"\nCosine Similarity Statistics:")
        self.report.print(f"  Synonyms: mean={syn_mean:.4f}, std={syn_std:.4f}")
        self.report.print(f"  Random:   mean={rand_mean:.4f}, std={rand_std:.4f}")
        self.report.print(f"  Difference: {syn_mean - rand_mean:+.4f}")

        # Statistical test
        t_stat, p_value = stats.ttest_ind(synonym_similarities, random_similarities)
        self.report.start_paragraph()
        print(f"\nT-test: t={t_stat:.4f}, p={p_value:.2e}")
        self.report.print(f"\nT-test: t={t_stat:.4f}, p={p_value:.2e}")
        if p_value < 0.05:
            print("→ Synonyms are significantly more similar than random pairs (p < 0.05)")
            self.report.print("→ Synonyms are significantly more similar than random pairs (p < 0.05)")
        self.report.end_paragraph()

        # Plotly
        bins_edges = np.linspace(
            min(min(synonym_similarities), min(random_similarities)),
            max(max(synonym_similarities), max(random_similarities)),
            120,
        )


        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=synonym_similarities,
            xbins=dict(start=bins_edges[0], end=bins_edges[-1], size=bins_edges[1] - bins_edges[0]),
            histnorm='probability density',
            opacity=0.7,
            name=f'Synonyms (μ={syn_mean:.3f})',
            hoverinfo='skip'
        ))

        fig.add_trace(go.Histogram(
            x=random_similarities,
            xbins=dict(start=bins_edges[0], end=bins_edges[-1], size=bins_edges[1] - bins_edges[0]),
            histnorm='probability density',
            opacity=0.7,
            name=f'Random (μ={rand_mean:.3f})',
            hoverinfo='skip'
        ))

        fig.update_layout(
            template='plotly_white',
            title='Synonym vs Random Word Similarity',
            xaxis_title='Cosine Similarity',
            yaxis_title='Density',
            barmode='overlay',
            width=1000,
            height=500,
        )

        # fig.write_image("synonym_validation.png", scale=2)
        if save:
            path = os.path.join(self.outpout_image_directory, (f"{self.encoder_type}_synonym_basic" + (f'_{self.data_manager.current_decomposition}' if self.data_manager.current_decomposition is not None else "") + ".png"))
            fig.write_image(path)
            print(f"Interactive figure saved to {path}")
        self.report.add_figure(fig)

        return {
            "synonym_similarities": synonym_similarities,
            "random_similarities": random_similarities,
            "t_stat": t_stat,
            "p_value": p_value,
        }

    # Required
    def analyze_neighbour_to_synonym_relationship(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        label_col: str = "Word",
        synonym_col: str = "synonyms",
        n_neighbors: int = 10,
        min_synonyms: int = 2,
        synonym_delimiters: tuple[str, ...] = (";", "|"),
        lowercase_matching: bool = True,
        n_examples: int = 5,
    ):
        """
        Simple synonym-neighbour comparison:
        filter anchor words by synonym count, encode missing synonyms, find
        neighbours for the original words, and compare those neighbours to the
        synonym list of each original word.
        """
        if df is None:
            assert self.data_manager.get_current_dataframe() is not None
            df = self.data_manager.get_current_dataframe()

        self.assert_model_loaded()

        current_model_type = self.data_manager.current_model_type
        current_subsample = self.data_manager.current_subsample
        current_decomposition = self.data_manager.current_decomposition
        analysis_df = df
        prepare_neighbor_space_fn = None

        if current_decomposition == "rbf":
            try:
                self.data_manager.prepare_data(
                    current_model_type,
                    subsample=current_subsample,
                    decomposition=None,
                    compute_nn=False,
                )
                analysis_df = self.data_manager.get_current_dataframe().copy()
            finally:
                self.data_manager.current_model_type = current_model_type
                self.data_manager.current_subsample = current_subsample
                self.data_manager.current_decomposition = current_decomposition

            def prepare_neighbor_space(df_to_prepare: pd.DataFrame):
                return self.data_manager.apply_rbf(
                    df=df_to_prepare,
                    embedding_col=embedding_col,
                    aggregation_method=aggregation_method,
                    n_components=200,
                    gamma=0.5,
                    random_state=42,
                )

            prepare_neighbor_space_fn = prepare_neighbor_space
        elif current_decomposition is not None:
            raise NotImplementedError(
                "Dynamic synonym encoding is currently only supported for the "
                f"'rbf' decomposition, not '{current_decomposition}'."
            )

        def encode_word(word: str):
            with torch.no_grad():
                result = self.encoder_model.encode_text(word, device=self.pipeline_device)
            return self._tensor_to_numpy_float32(result["patch_embeddings"])

        results = run_neighbour_to_synonym_analysis(
            df=analysis_df,
            find_nearest_neighbors_fn=self.find_nearest_neighbors,
            encode_word_fn=encode_word,
            prepare_neighbor_space_fn=prepare_neighbor_space_fn,
            embedding_col=embedding_col,
            aggregation_method=aggregation_method,
            label_col=label_col,
            synonym_col=synonym_col,
            n_neighbors=n_neighbors,
            min_synonyms=min_synonyms,
            synonym_delimiters=synonym_delimiters,
            lowercase_matching=lowercase_matching,
        )

        def emit(line: str):
            print(line)
            self.report.print(line)

        emit("=" * 60)
        emit("NEIGHBOUR TO SYNONYM ANALYSIS")
        emit("=" * 60)
        emit(f"Anchor words evaluated: {results['anchor_count']:,}")
        emit(f"Neighbour space size: {results['neighbor_space_size']:,}")
        emit(
            f"Encoded missing synonyms: {results['encoded_missing_synonym_count']:,}"
        )

        if results["anchor_count"] == 0:
            emit("")
            emit("No valid anchors remained after filtering.")
            return results

        emit("")
        emit(f"Mean overlap count: {results['mean_overlap_count']:.3f}")
        emit(f"Mean synonym recall: {results['mean_synonym_recall']:.2%}")
        emit(f"Mean neighbour precision: {results['mean_neighbor_precision']:.2%}")
        emit(f"Hit rate: {results['hit_rate']:.2%}")

        details = results["details"]
        if not details.empty:
            hit_examples = details[details["has_match"]].head(n_examples)
            if not hit_examples.empty:
                emit("")
                emit("Example synonym hits:")
                for _, row in hit_examples.iterrows():
                    emit(
                        f"  {row['word']}: matched {row['matched_synonyms']} "
                        f"at rank {row['first_match_rank']}"
                    )

            miss_examples = (
                details[~details["has_match"]]
                .sort_values("synonym_recall", ascending=True)
                .head(n_examples)
            )
            if not miss_examples.empty:
                emit("")
                emit("Example misses:")
                for _, row in miss_examples.iterrows():
                    emit(
                        f"  {row['word']}: no synonym hit; "
                        f"synonyms={row['synonyms_in_neighbor_space'][:6]}"
                    )

        return results


    def analyze_typo_robustness(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        label_col: str = "Word",
        max_words: int | None = 500,
        min_word_length: int = 4,
        random_state: int = 42,
        n_examples: int = 5,
    ):
        """
        Simple typo robustness evaluation over the current vocabulary.
        """
        if df is None:
            assert self.data_manager.get_current_dataframe() is not None
            df = self.data_manager.get_current_dataframe()

        self.assert_model_loaded()

        current_model_type = self.data_manager.current_model_type
        current_subsample = self.data_manager.current_subsample
        current_decomposition = self.data_manager.current_decomposition
        analysis_df = df
        prepare_typo_evaluation_space_fn = None

        if current_decomposition == "rbf":
            try:
                self.data_manager.prepare_data(
                    current_model_type,
                    subsample=current_subsample,
                    decomposition=None,
                    compute_nn=False,
                )
                analysis_df = self.data_manager.get_current_dataframe().copy()
            finally:
                self.data_manager.current_model_type = current_model_type
                self.data_manager.current_subsample = current_subsample
                self.data_manager.current_decomposition = current_decomposition

            def prepare_typo_evaluation_space(
                vocabulary_df: pd.DataFrame,
                typo_pairs_df: pd.DataFrame,
            ):
                vocabulary_df = vocabulary_df.copy().reset_index(drop=True)
                clean_words = set(vocabulary_df[label_col].astype(str))
                typo_words = (
                    typo_pairs_df["typo_word"]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                )

                new_rows = []
                for typo_word in typo_words:
                    if typo_word in clean_words:
                        continue

                    with torch.no_grad():
                        result = self.encoder_model.encode_text(
                            typo_word,
                            device=self.pipeline_device,
                        )

                    new_rows.append(
                        {
                            label_col: typo_word,
                            embedding_col: self._tensor_to_numpy_float32(
                                result["patch_embeddings"]
                            ),
                        }
                    )

                combined_df = vocabulary_df[[col for col in vocabulary_df.columns]].copy()
                if new_rows:
                    combined_df = pd.concat(
                        [combined_df, pd.DataFrame(new_rows)],
                        ignore_index=True,
                    )

                combined_df = self.data_manager.apply_rbf(
                    df=combined_df,
                    embedding_col=embedding_col,
                    aggregation_method=aggregation_method,
                    n_components=200,
                    gamma=0.5,
                    random_state=42,
                )

                transformed_lookup = (
                    combined_df[[label_col, embedding_col]]
                    .drop_duplicates(subset=[label_col], keep="first")
                    .set_index(label_col)[embedding_col]
                    .to_dict()
                )

                prepared_vocabulary_df = vocabulary_df.copy()
                prepared_vocabulary_df["embedding_vector"] = prepared_vocabulary_df[
                    label_col
                ].map(transformed_lookup)

                typo_vector_lookup = {
                    typo_word: transformed_lookup.get(typo_word)
                    for typo_word in typo_words
                }

                return {
                    "vocabulary_df": prepared_vocabulary_df,
                    "typo_vector_lookup": typo_vector_lookup,
                }

            prepare_typo_evaluation_space_fn = prepare_typo_evaluation_space
        elif current_decomposition is not None:
            raise NotImplementedError(
                "Dynamic typo encoding is currently only supported for the "
                f"'rbf' decomposition, not '{current_decomposition}'."
            )

        def encode_word(word: str):
            with torch.no_grad():
                result = self.encoder_model.encode_text(word, device=self.pipeline_device)
            return self._tensor_to_numpy_float32(result["patch_embeddings"])

        results = run_typo_robustness_analysis(
            df=analysis_df,
            encode_word_fn=encode_word,
            aggregate_embeddings_fn=self.aggregate_embeddings,
            prepare_typo_evaluation_space_fn=prepare_typo_evaluation_space_fn,
            embedding_col=embedding_col,
            label_col=label_col,
            aggregation_method=aggregation_method,
            max_words=max_words,
            min_word_length=min_word_length,
            random_state=random_state,
        )

        def emit(line: str):
            print(line)
            self.report.print(line)

        summary = results["summary"]
        details = results["details"]

        emit("=" * 60)
        emit("TYPO ROBUSTNESS ANALYSIS")
        emit("=" * 60)
        emit(f"Vocabulary size: {len(results['vocabulary']):,}")
        emit(f"Generated typo variants: {len(results['typo_pairs']):,}")
        emit(f"Evaluated typo queries: {len(details):,}")

        if details.empty:
            emit("")
            emit("No valid typo queries were available after filtering.")
            return results

        emit("")
        emit(f"Recall@1: {summary['recall_at_1']:.2%}")
        emit(f"Recall@5: {summary['recall_at_5']:.2%}")
        emit(f"MRR: {summary['mrr']:.4f}")

        best_examples = details.sort_values(by="target_rank", ascending=True).head(n_examples)
        if not best_examples.empty:
            emit("")
            emit("Example matches:")
            for row in best_examples.itertuples(index=False):
                emit(f"  {row.typo_word} -> {row.clean_word} (rank={row.target_rank})")

        difficult_examples = details.sort_values(by="target_rank", ascending=False).head(n_examples)
        if not difficult_examples.empty:
            emit("")
            emit("Example difficult cases:")
            for row in difficult_examples.itertuples(index=False):
                predicted = row.top_5_words[0] if row.top_5_words else "N/A"
                emit(
                    f"  {row.typo_word} -> target {row.clean_word} "
                    f"(rank={row.target_rank}, top1={predicted})"
                )

        return results

    def analyze_isotropy(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
    ):
        """
        Compute the mean off-diagonal cosine similarity (ANI) over the current
        embedding vocabulary.
        """
        if df is None:
            assert self.data_manager.get_current_dataframe() is not None
            df = self.data_manager.get_current_dataframe()

        results = run_isotropy_analysis(
            df=df,
            aggregate_embeddings_fn=self.aggregate_embeddings,
            embedding_col=embedding_col,
            aggregation_method=aggregation_method,
        )

        def emit(line: str):
            print(line)
            self.report.print(line)

        emit("=" * 60)
        emit("ISOTROPY ANALYSIS")
        emit("=" * 60)
        emit(f"Vocabulary size: {results['vocabulary_size']:,}")
        emit(f"Embedding dimension: {results['embedding_dim']:,}")
        emit(f"Zero-norm embeddings removed: {results['zero_norm_count']:,}")

        if results["vocabulary_size"] < 2:
            emit("")
            emit("Not enough valid embeddings remained to compute ANI.")
            return results

        emit("")
        emit(f"ANI score: {results['ani_score']:.6f}")

        return results

    def analyze_covariance(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        top_k: int = 32,
        min_position: int = 8,
        diffusion_t_window: bool = True,
    ):
        """
        Analyze the covariance spectrum (latent geometry) of the current
        embedding population: mean-vector norm, per-dimension stds,
        eigenvalue spectrum, participation ratio and the cosine-schedule
        t-window in which the variance becomes resolvable.
        """
        if df is None:
            assert self.data_manager.get_current_dataframe() is not None
            df = self.data_manager.get_current_dataframe()

        def emit(line: str):
            print(line)
            self.report.print(line)

        emit("=" * 60)
        emit("COVARIANCE / LATENT GEOMETRY ANALYSIS")
        emit("=" * 60)

        dataset_tag = "context" if "position" in df.columns else "synonyms"

        if dataset_tag == "context":
            positions_per_doc = df.groupby("doc_id")["position"].size()
            emit(
                f"Documents: {df['doc_id'].nunique():,}; positions per doc: "
                f"min {int(positions_per_doc.min())}, mean {positions_per_doc.mean():.1f}, "
                f"max {int(positions_per_doc.max())}"
            )

            if min_position > 0:
                aggregated_norms = df[embedding_col].apply(
                    lambda emb: None if emb is None else float(np.linalg.norm(
                        self.aggregate_embeddings(emb, method=aggregation_method)
                    ))
                )
                emit(f"Position filter: keeping position >= {min_position}")
                for position in range(min_position):
                    position_norms = aggregated_norms[df["position"] == position].dropna()
                    if position_norms.empty:
                        continue
                    emit(
                        f"  position {position}: count {len(position_norms):,}, "
                        f"mean norm {position_norms.mean():.4f}, "
                        f"median norm {position_norms.median():.4f}"
                    )
                baseline_norms = aggregated_norms[df["position"] >= min_position].dropna()
                if not baseline_norms.empty:
                    emit(
                        f"  position >= {min_position}: count {len(baseline_norms):,}, "
                        f"mean norm {baseline_norms.mean():.4f}, "
                        f"median norm {baseline_norms.median():.4f}"
                    )
                df = df[df["position"] >= min_position]

        results = run_covariance_analysis(
            df=df,
            aggregate_embeddings_fn=self.aggregate_embeddings,
            embedding_col=embedding_col,
            aggregation_method=aggregation_method,
            top_k=top_k,
            compute_t_window=diffusion_t_window,
        )

        emit("")
        emit(f"Number of samples: {results['n_samples']:,}")
        emit(f"Embedding dimension: {results['embedding_dim']:,}")
        emit(f"Zero-norm embeddings removed: {results['zero_norm_count']:,}")

        if results["n_samples"] < 2:
            emit("")
            emit("Not enough valid embeddings remained to compute the covariance spectrum.")
            return results

        if results["n_lt_dim"]:
            emit("WARNING: n_samples <= embedding_dim — the spectrum estimate is rank-deficient.")

        emit("")
        emit(f"Mean-vector norm: {results['mean_norm']:.4f}")
        emit(f"Mean embedding norm: {results['mean_embedding_norm']:.4f}")
        emit(f"Mean-norm ratio (||mu|| / E||z||): {results['mean_norm_ratio']:.4f}")
        emit("")
        emit(
            f"Per-dimension std: min {results['std_min']:.4f}, "
            f"median {results['std_median']:.4f}, max {results['std_max']:.4f}"
        )
        emit("")
        emit(f"Top eigenvalue (normalized, mean eigenvalue = 1): {results['eigenvalues_normalized'][0]:.4f}")
        emit(f"lambda_max: {results['lambda_max']:.6g}")
        emit(f"lambda_median: {results['lambda_median']:.6g}")
        emit(f"lambda_max / lambda_median: {results['lambda_max_over_median']:.2f}")
        lambda_min_suffix = "" if results["lambda_min_reliable"] else " (UNRELIABLE: n_samples not >> dim)"
        emit(f"lambda_min: {results['lambda_min']:.6g}{lambda_min_suffix}")
        emit("")
        emit(f"Top-{results['top_k']} explained variance: {results['top_k_explained_variance'] * 100:.2f}%")
        emit(
            f"Participation ratio (effective dims): {results['participation_ratio']:.1f} "
            f"of {results['embedding_dim']} "
            f"({results['participation_ratio'] / results['embedding_dim'] * 100:.1f}%)"
        )
        if results["t_window"] is not None:
            t_low, t_high = results["t_window"]
            emit("")
            emit(
                f"Diffusion t-window (cosine schedule, SNR=1 crossings): "
                f"5%..95% of variance resolvable in t in [{t_low:.3f}, {t_high:.3f}]"
            )

        eigenvalues_normalized = results["eigenvalues_normalized"]
        ranks = np.arange(1, len(eigenvalues_normalized) + 1)
        cumulative_variance = np.cumsum(results["eigenvalues_raw"]) / results["eigenvalues_raw"].sum()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=ranks, y=eigenvalues_normalized, mode="lines", name="Normalized eigenvalue"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=ranks, y=cumulative_variance, mode="lines", name="Cumulative explained variance"),
            secondary_y=True,
        )
        fig.update_layout(
            title=f"Covariance Spectrum ({dataset_tag}) - {self.encoder_type}",
            template="plotly_white",
            width=self.report.default_width,
            height=self.report.default_height,
        )
        fig.update_xaxes(title_text="Eigenvalue rank")
        fig.update_yaxes(title_text="Normalized eigenvalue (log)", type="log", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative explained variance", range=[0, 1], secondary_y=True)

        image_name = f"{self.encoder_type}_covariance_{dataset_tag}" + (
            f"_{self.data_manager.current_decomposition}"
            if self.data_manager.current_decomposition is not None else ""
        ) + ".png"
        try:
            fig.write_image(os.path.join(self.outpout_image_directory, image_name))
        except Exception as e:
            print(f"WARNING: could not export covariance figure to PNG: {e}")

        self.report.add_figure(fig)

        return results

    def analyze_explained_variance(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        target_dims: list = None,
        min_position: int = 8,
    ):
        """
        Plot the cumulative explained-variance curve (PCA eigenspectrum) of the
        current embedding population, with markers at the requested target
        dimensionalities.
        """
        if df is None:
            assert self.data_manager.get_current_dataframe() is not None
            df = self.data_manager.get_current_dataframe()

        def emit(line: str):
            print(line)
            self.report.print(line)

        emit("=" * 60)
        emit("EXPLAINED VARIANCE ANALYSIS")
        emit("=" * 60)

        dataset_tag = "context" if "position" in df.columns else "synonyms"

        # Same population filter as the covariance analysis: early positions
        # are unrepresentative warm-up states.
        if dataset_tag == "context" and min_position > 0:
            emit(f"Position filter: keeping position >= {min_position}")
            df = df[df["position"] >= min_position]

        results = run_explained_variance_analysis(
            df=df,
            aggregate_embeddings_fn=self.aggregate_embeddings,
            embedding_col=embedding_col,
            aggregation_method=aggregation_method,
            target_dims=target_dims,
        )

        emit("")
        emit(f"Number of samples: {results['n_samples']:,}")
        emit(f"Embedding dimension: {results['embedding_dim']:,}")
        emit(f"Zero-norm embeddings removed: {results['zero_norm_count']:,}")

        emit("")
        emit(f"Requested target dimensionalities: {results['requested_dims']}")
        if results["discarded_dims"]:
            for dim, reason in results["discarded_dims"]:
                emit(f"  DISCARDED {dim}: {reason}")
        else:
            emit("  (none discarded)")

        if not results["kept_dims"]:
            emit("")
            emit("No usable target dimensionalities or empty spectrum — nothing to plot.")
            return results

        emit("")
        for k in results["kept_dims"]:
            emit(f"Explained variance @ {k} dims: {results['explained_variance_at'][k] * 100:.2f}%")

        cumulative = results["cumulative_explained_variance"]
        ranks = np.arange(1, len(cumulative) + 1)
        kept = results["kept_dims"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ranks, y=cumulative * 100, mode="lines", name="Cumulative explained variance",
        ))
        fig.add_trace(go.Scatter(
            x=kept,
            y=[results["explained_variance_at"][k] * 100 for k in kept],
            mode="markers+text",
            text=[str(k) for k in kept],
            textposition="bottom right",
            name="Power of two dimensions",
        ))
        fig.update_layout(
            title=f"Explained Variance ({dataset_tag}) - {self.encoder_type}",
            template="plotly_white",
            width=self.report.default_width,
            height=self.report.default_height,
        )
        # Explicit power-of-two ticks with plain number labels: the default
        # log-axis labeling only marks decades (1, 10, 100, ...).
        tick_dims = [1]
        while tick_dims[-1] * 2 <= len(cumulative):
            tick_dims.append(tick_dims[-1] * 2)
        fig.update_xaxes(
            title_text="Number of dimensions", type="log",
            tickvals=tick_dims, exponentformat="none",
        )
        fig.update_yaxes(
            title_text="Cumulative explained variance (%)", range=[0, 105],
            dtick=10, exponentformat="none",
        )

        image_name = f"{self.encoder_type}_explained_variance_{dataset_tag}" + (
            f"_{self.data_manager.current_decomposition}"
            if self.data_manager.current_decomposition is not None else ""
        ) + ".png"
        # try:
        fig.write_image(os.path.join(self.outpout_image_directory, image_name))
        # except Exception as e:
            # print(f"WARNING: could not export explained-variance figure to PNG: {e}")

        self.report.add_figure(fig)

        return results

    def analyze_linear_probe_word_length(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        label_col: str = "Word",
        word_lengths: list[int] | tuple[int, ...] = (3, 4, 5, 6, 7, 8),
        n_words_per_length: int = 100,
        random_state: int = 42,
        cv_folds: int = 5,
        max_iter: int = 1000,
        n_examples: int = 5,
    ):
        """
        Balanced multiclass linear probe for predicting len(word) from
        aggregated embeddings.
        """
        if df is None:
            assert self.data_manager.get_current_dataframe() is not None
            df = self.data_manager.get_current_dataframe()

        results = run_linear_probe_word_length_analysis(
            df=df,
            aggregate_embeddings_fn=self.aggregate_embeddings,
            embedding_col=embedding_col,
            label_col=label_col,
            aggregation_method=aggregation_method,
            word_lengths=word_lengths,
            n_words_per_length=n_words_per_length,
            random_state=random_state,
            cv_folds=cv_folds,
            max_iter=max_iter,
        )

        def emit(line: str):
            print(line)
            self.report.print(line)

        emit("=" * 60)
        emit("WORD LENGTH LINEAR PROBE")
        emit("=" * 60)
        emit(f"Requested word lengths: {results['requested_word_lengths']}")
        emit(f"Available word lengths: {results['available_word_lengths']}")
        if results["unavailable_word_lengths"]:
            emit(f"Unavailable word lengths: {results['unavailable_word_lengths']}")

        if results["available_counts_by_length"]:
            emit(f"Candidate counts by length: {results['available_counts_by_length']}")

        if results["n_classes"] < 2 or results["n_samples"] == 0:
            emit("")
            emit("Not enough valid classes remained to fit the probe.")
            return results

        emit(f"Balanced sample size per length: {results['effective_n_words_per_length']:,}")
        emit(f"Total sampled words: {results['n_samples']:,}")
        emit(f"Embedding dimension: {results['embedding_dim']:,}")
        emit(f"Cross-validation folds: {results['cv_folds']}")

        emit("")
        emit(f"Accuracy: {results['accuracy_mean']:.3f} +- {results['accuracy_std']:.3f}")
        emit(
            f"Balanced accuracy: {results['balanced_accuracy_mean']:.3f} +- "
            f"{results['balanced_accuracy_std']:.3f}"
        )
        emit(f"Macro F1: {results['macro_f1']:.3f}")
        emit(f"Random baseline: {results['baseline_accuracy']:.3f}")
        emit(f"Lift over baseline: {results['accuracy_mean'] - results['baseline_accuracy']:+.3f}")

        predictions = results["predictions"]
        if not predictions.empty:
            best_examples = predictions[predictions["is_correct"]].head(n_examples)
            if not best_examples.empty:
                emit("")
                emit("Example correct predictions:")
                for _, row in best_examples.iterrows():
                    emit(
                        f"  {row[label_col]} -> true={row['word_length']}, "
                        f"predicted={row['predicted_length']}"
                    )

            difficult_examples = (
                predictions[~predictions["is_correct"]]
                .sort_values(["absolute_error", "word_length"], ascending=[False, True])
                .head(n_examples)
            )
            if not difficult_examples.empty:
                emit("")
                emit("Example difficult predictions:")
                for _, row in difficult_examples.iterrows():
                    emit(
                        f"  {row[label_col]} -> true={row['word_length']}, "
                        f"predicted={row['predicted_length']}, abs_error={row['absolute_error']}"
                    )

        length_labels = [str(length) for length in results["available_word_lengths"]]
        per_length_accuracy_df = results["per_length_accuracy"]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Normalized Confusion Matrix", "Per-Length Accuracy"),
            horizontal_spacing=0.15,
        )
        fig.add_trace(
            go.Heatmap(
                z=results["confusion_matrix_normalized"],
                x=length_labels,
                y=length_labels,
                colorscale="Blues",
                zmin=0.0,
                zmax=1.0,
                text=results["confusion_matrix"],
                texttemplate="%{text}",
                colorbar=dict(title="Recall"),
                hovertemplate=(
                    "True length: %{y}<br>"
                    "Predicted length: %{x}<br>"
                    "Row-normalized score: %{z:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=per_length_accuracy_df["word_length"].astype(str),
                y=per_length_accuracy_df["accuracy"],
                text=[f"{value:.2%}" for value in per_length_accuracy_df["accuracy"]],
                textposition="outside",
                marker_color="#1f77b4",
                hovertemplate=(
                    "Word length: %{x}<br>"
                    "Accuracy: %{y:.2%}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Predicted Length", row=1, col=1)
        fig.update_yaxes(title_text="True Length", row=1, col=1)
        fig.update_xaxes(title_text="Word Length", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", range=[0.0, 1.05], row=1, col=2)
        fig.update_layout(
            title=(
                "Word Length Linear Probe"
                f"<br>{results['n_classes']} classes, "
                f"{results['effective_n_words_per_length']} words/class"
            ),
            template="plotly_white",
            width=1100,
            height=450,
            showlegend=False,
        )
        self.report.add_figure(fig)

        return results



    # ====== Synonym Cluster Analysis ======
    def build_synonym_classes(
        self,
        df: pd.DataFrame,
        label_col: str = "Word",
        synonym_col: str = "synonyms",
        embedding_col: str = "patch_embeddings_array",
        delimiter: str = ";",
        strategy: Literal["direct", "transitive"] = "direct"
    ) -> dict[str, int]:
        return self.build_clusters_simple_two(df=df, label_col=label_col, synonym_col=synonym_col, embedding_col=embedding_col, delimiter=delimiter)

    def build_clusters(
        self,
        df: pd.DataFrame,
        label_col: str = "Word",
        synonym_col: str = "synonyms",
        delimiter: str = ";"
    ) -> dict[str, list[int]]:
        words_in_df = set(df[label_col].values)
        
        clusters = []
        word_to_clusters = {}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building clusters"):
            word = row[label_col] # joyful
            synonyms = self.parse_synonyms(row[synonym_col], delimiter) # happy, nice, great
            synonyms_in_df = {s for s in synonyms if s in words_in_df} # happy, nice
            
            words_to_group = {word} | synonyms_in_df # joyful, happy, nice
            
            # Find all clusters that contain any of these words
            found_indices = set()
            # for w in words_to_group:
            #     if w in word_to_clusters:
            #         found_indices.update(word_to_clusters[w])
            
            if word in word_to_clusters:
                found_indices.update(word_to_clusters[word])

            if not found_indices:
                # Create new cluster
                idx = len(clusters)
                clusters.append(words_to_group.copy())
                for w in words_to_group:
                    word_to_clusters[w] = {idx}
            else:
                # Add current word to ALL found clusters (bridge word behavior)
                for idx in found_indices:
                    clusters[idx].add(word)
                word_to_clusters[word] = found_indices.copy()
                
                # Add unassigned synonyms to first found cluster only
                first_idx = min(found_indices)
                for syn in synonyms_in_df:
                    if syn not in word_to_clusters:
                        clusters[first_idx].add(syn)
                        word_to_clusters[syn] = {first_idx}
        

        idx_to_class = {}
        class_id = 0
        for idx, cluster in enumerate(clusters):
            if cluster:
                idx_to_class[idx] = class_id
                class_id += 1
        
        word_to_class = {}
        for word, indices in word_to_clusters.items():
            word_to_class[word] = sorted(idx_to_class[i] for i in indices if i in idx_to_class)
        
        return word_to_class

    def build_clusters_simple(
        self,
        df: pd.DataFrame,
        label_col: str = "Word",
        synonym_col: str = "synonyms",
        delimiter: str = ";"
    ) -> dict[str, list[int]]:
        words_in_df = set(df[label_col].values)
        
        clusters = []
        word_to_clusters = {}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building clusters"):
            word = row[label_col] # joyful
            synonyms = self.parse_synonyms(row[synonym_col], delimiter) # happy, nice, great
            synonyms_in_df = {s for s in synonyms if s in words_in_df} # happy, nice
            
            words_to_group = {word} | synonyms_in_df # joyful, happy, nice
            
            if word in word_to_clusters:
                continue
            else:
                idx = len(clusters)
                clusters.append(words_to_group.copy())
                for w in words_to_group:
                    word_to_clusters[w] = {idx}
        
        print("CHECK")
        print(len(clusters))
        print(clusters[0])
        n = 0
        for c in clusters:
            n += len(c)
        
        print(n / len(clusters))

        idx_to_class = {}
        class_id = 0
        for idx, cluster in enumerate(clusters):
            if cluster:
                idx_to_class[idx] = class_id
                class_id += 1
        
        word_to_class = {}
        for word, indices in word_to_clusters.items():
            word_to_class[word] = sorted(idx_to_class[i] for i in indices if i in idx_to_class)
        
        return word_to_class

    def build_clusters_simple_two(
        self,
        df: pd.DataFrame,
        label_col: str = "Word",
        synonym_col: str = "synonyms",
        embedding_col: str = "patch_embeddings_array",
        delimiter: str = ";",
        encode_missing: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, list[int]]]:
        """Build clusters, encoding missing synonyms on the fly."""
        
        def is_valid_embedding(emb):
            if emb is None:
                return False
            if isinstance(emb, float):  # NaN shows up as float
                return False
            if not isinstance(emb, np.ndarray):
                return False
            if np.isnan(emb).any():
                return False
            return True

        df = df.copy()
        words_in_df = set(df[label_col].values)


        print(f'COL: {embedding_col}')
        invalid_count = (~df[embedding_col].apply(is_valid_embedding)).sum()
        if invalid_count > 0:
            print(f"Removing {invalid_count} rows with invalid embeddings")

        # Step 1: Find and encode missing synonyms
        if encode_missing:
            missing_synonyms = set()
            
            for _, row in df.iterrows():
                if not is_valid_embedding(row['patch_embeddings_array']):
                    print("PREVIOUS ERROR IN ENCODING")
                synonyms = self.parse_synonyms(row[synonym_col], delimiter)
                for syn in synonyms:
                    if syn not in words_in_df:
                        missing_synonyms.add(syn)
            
            print(f"Found {len(missing_synonyms)} synonyms not in dataframe")

            if missing_synonyms:
                print(f"Encoding {len(missing_synonyms)} missing synonyms...")
                new_rows = []
                
                nanc = 0

                for syn in tqdm(missing_synonyms, desc="Encoding missing synonyms"):
                    # try:
                    with torch.no_grad():
                        result = self.encoder_model.encode_text(syn, self.pipeline_device)
                    patch_emb = self._tensor_to_numpy_float32(result["patch_embeddings"])
                    
                    if not is_valid_embedding(patch_emb):
                        raise Exception

                    new_rows.append({
                        label_col: syn,
                        synonym_col: "",
                        embedding_col: patch_emb,
                    })
                    words_in_df.add(syn)
                
                print("ENC STAT")
                print(nanc)
                print(len(new_rows))

                if new_rows:
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    print(f"Added {len(new_rows)} new rows to dataframe")
        
        # Step 2: Build clusters
        clusters = []
        word_to_clusters = {}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building clusters"):
            word = row[label_col]
            synonyms = self.parse_synonyms(row[synonym_col], delimiter)
            if len(synonyms) == 0:
                continue
            synonyms_in_df = {s for s in synonyms if s in words_in_df}
            words_to_group = {word} | synonyms_in_df
            
            if word in word_to_clusters:
                continue
            
            idx = len(clusters)
            clusters.append(words_to_group.copy())
            
            for w in words_to_group:
                word_to_clusters[w] = {idx}
        
        print(f"Created {len(clusters)} clusters")
        
        # Convert to final class IDs
        idx_to_class = {idx: cid for cid, idx in enumerate(i for i, c in enumerate(clusters) if c)}
        
        word_to_class = {}
        for word, indices in word_to_clusters.items():
            word_to_class[word] = sorted(idx_to_class[i] for i in indices if i in idx_to_class)
        
        return df, word_to_class

    def compute_cluster_metrics(
        self,
        embeddings: np.ndarray,
        word_to_class: dict[str, list[int]],
        words: list[str],
        random_state: int = 42
    ) -> dict:
        """
        Compute intra/inter-cluster similarity.
        
        Handles words belonging to multiple clusters:
        - Intra-cluster: word contributes to ALL its clusters
        - Inter-cluster: computed between cluster centroids
        
        Args:
            embeddings: Embedding matrix (n_words, dim)
            word_to_class: Dict mapping word -> list of cluster IDs
            words: List of words in same order as embeddings
            random_state: Random seed
        
        Returns:
            Dictionary with metrics
        """
        np.random.seed(random_state)
        
        # Build cluster -> word indices mapping
        cluster_to_indices = {}
        for i, word in enumerate(words):
            if word in word_to_class:
                for cid in word_to_class[word]:
                    if cid not in cluster_to_indices:
                        cluster_to_indices[cid] = []
                    cluster_to_indices[cid].append(i)
        
        n_clusters = len(cluster_to_indices)
        unique_clusters = sorted(cluster_to_indices.keys())
        
        # Count bridge words
        n_bridge_words = sum(1 for classes in word_to_class.values() if len(classes) > 1)
        
        # Compute intra-cluster similarities
        print("Computing intra-cluster similarities...")
        intra_similarities = []
        cluster_intra = {}
        
        for cid in tqdm(unique_clusters, desc="Intra-cluster"):
            indices = cluster_to_indices[cid]
            
            if len(indices) < 2:
                cluster_intra[cid] = np.nan
                continue
            
            cluster_emb = embeddings[indices]
            
            # Full matrix for small clusters
            norms = np.linalg.norm(cluster_emb, axis=1)
            valid_mask = norms > 0
            
            if valid_mask.sum() < 2:
                cluster_intra[cid] = np.nan
                continue
            
            cluster_emb_valid = cluster_emb[valid_mask]
            cluster_sim = cosine_similarity(cluster_emb_valid)
            triu_indices = np.triu_indices(len(cluster_emb_valid), k=1)
            intra_vals = cluster_sim[triu_indices].tolist()
            
            intra_similarities.extend(intra_vals)
            cluster_intra[cid] = float(np.mean(intra_vals)) if intra_vals else np.nan
        
        # Compute cluster centroids for inter-cluster similarity
        print("Computing cluster centroids...")
        centroids = []
        valid_cluster_ids = []
        
        for cid in tqdm(unique_clusters, desc="Computing centroids"):
            indices = cluster_to_indices[cid]
            cluster_emb = embeddings[indices]
            
            norms = np.linalg.norm(cluster_emb, axis=1)
            valid_mask = norms > 0
            
            if valid_mask.sum() > 0:
                centroid = np.mean(cluster_emb[valid_mask], axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 0:
                    centroids.append(centroid / centroid_norm)
                    valid_cluster_ids.append(cid)
        
        # Compute inter-cluster similarity between centroids
        inter_similarities = []
        if len(centroids) > 1:
            print(f"Computing inter-cluster similarity between {len(centroids)} centroids...")
            centroids_matrix = np.vstack(centroids)
            centroid_sim_matrix = np.dot(centroids_matrix, centroids_matrix.T)
            triu_indices = np.triu_indices(len(centroids), k=1)
            inter_similarities = centroid_sim_matrix[triu_indices].tolist()
        
        # Summary metrics
        mean_intra = float(np.mean(intra_similarities)) if intra_similarities else np.nan
        mean_inter = float(np.mean(inter_similarities)) if inter_similarities else np.nan
        separation_gap = mean_intra - mean_inter
        
        return {
            "mean_intra_similarity": mean_intra,
            "mean_inter_similarity": mean_inter,
            "separation_gap": separation_gap,
            "per_cluster_intra": cluster_intra,
            "n_clusters": n_clusters,
            "n_bridge_words": n_bridge_words,
            "intra_std": float(np.std(intra_similarities)) if intra_similarities else np.nan,
            "inter_std": float(np.std(inter_similarities)) if inter_similarities else np.nan,
            "n_intra_pairs": len(intra_similarities),
            "n_inter_pairs": len(inter_similarities),
            "centroids": np.array(centroids) if centroids else None,
            "centroid_cluster_ids": valid_cluster_ids,
        }

    def add_synonym_encodings(
        self,
        df: pd.DataFrame
    ):
        pass

    def analyze_synonym_clusters(
        self,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        label_col: str = "Word",
        synonym_col: str = "synonyms",
        aggregation_method: str = "mean",
        synonym_delimiter: str = ";",
        clustering_strategy: Literal["direct", "transitive"] = "direct",
        run_kmeans: bool = True,
        kmeans_random_state: int = 42,
        number_of_random_samples_to_use: int = None,
        number_of_minimum_synonyms: int = 8,
        verbose: int = 2,
    ) -> dict:
        # Prepare dataframe
        # Use pipeline frame if no parameter is given
        # Filter invalid embeddings and throw error if there are no valid embeddings
        # Probably messed up encoding then
        # Randomly sample if applicable

        print("=" * 60)
        print("SYNONYM CLUSTERS")
        print("=" * 60)

        def is_valid_embedding(emb):
            if emb is None:
                return False
            if not isinstance(emb, np.ndarray):
                return False
            if np.isnan(emb).any():
                return False
            return True

        def remove_nan(df):
            df_valid = df[df[embedding_col].apply(is_valid_embedding)].copy().reset_index(drop=True)
            if len(df_valid) == 0:
                raise ValueError("No valid embeddings found")
            return df_valid
        
        def random_sample_if_required(df):
            if number_of_random_samples_to_use is not None:
                df_valid = df_valid.sample(n=number_of_random_samples_to_use)
                print(f"Sumsamples {number_of_random_samples_to_use} samples from initial dataframe.")
                return df_valid
            return df
        
        def filter_out_small_clusters(df):
            df_filtered = df[df[synonym_col].apply(lambda x: len(self.parse_synonyms(x)) >= number_of_minimum_synonyms)]
            print(f"Analyzing {len(df_filtered)} words with valid embeddings and valid cluster size")
            self.report.print(f"Analyzing {len(df_filtered)} words with valid embeddings and valid cluster size")
            return df_filtered.reset_index(drop=True)

        def build_synonym_classessss(df):
            # Build synonym classes
            # Primariliy use direct clutsering strategy
            print(f"Building synonym classes (strategy: {clustering_strategy})...")
            df_valid, word_to_class = self.build_synonym_classes( # with list word_to_class = {"bright": [0, 1], "happy": [2]} 
                df, label_col, synonym_col, synonym_delimiter, strategy=clustering_strategy
            )
            df_valid = df_valid[df_valid[label_col].isin(word_to_class.keys())].reset_index(drop=True)
            print(f"Filtered to {len(df_valid)} words with cluster assignments")

        # Use pipeline dataframe if no df is given
        if df is None:
            assert self.data_manager.get_current_dataframe() is not None
            df = self.data_manager.get_current_dataframe()
        
        df_valid = remove_nan(df)
        df_valid = random_sample_if_required(df_valid)
        df_valid = filter_out_small_clusters(df_valid)

        # Build synonym classes
        # Primariliy use direct clutsering strategy
        print(f"Building synonym classes (strategy: {clustering_strategy})...")
        df_valid, word_to_class = self.build_synonym_classes( # with list word_to_class = {"bright": [0, 1], "happy": [2]} 
            df=df_valid, label_col=label_col, synonym_col=synonym_col, embedding_col=embedding_col, delimiter=synonym_delimiter, strategy=clustering_strategy
        )
        df_valid = df_valid[df_valid[label_col].isin(word_to_class.keys())].reset_index(drop=True)
        print(f"Filtered to {len(df_valid)} words with cluster assignments")
        # df_valid = df_valid[df_valid[embedding_col].notna()].copy().reset_index(drop=True)

        # Count clusters and bridge words
        all_cluster_ids = set()
        n_bridge_words = 0
        for classes in word_to_class.values():
            all_cluster_ids.update(classes)
            if len(classes) > 1:
                n_bridge_words += 1
        
        n_classes = len(all_cluster_ids)
        print(f"Found {n_classes} synonym classes")
        print(f"Found {n_bridge_words} bridge words (belong to multiple clusters)")
        
        # Store cluster assignments in dataframe
        df_valid["synonym_classes"] = df_valid[label_col].map(word_to_class)
        missing_mapping = df_valid[df_valid["synonym_classes"].isna()][label_col].tolist()
        print(f"Words without cluster assignment: {len(missing_mapping)}")
        if missing_mapping:
            print(f"  Examples: {missing_mapping[:10]}")
            print(len(missing_mapping))
             
            first_missing = missing_mapping[0]
            print(f"\nDebug: Full row for '{first_missing}':")
            row = df_valid[df_valid[label_col] == first_missing].iloc[0]
            print(f"  Word: {row[label_col]}")
            print(f"  Synonyms: {row[synonym_col]}")
            print(f"  Embedding type: {type(row[embedding_col])}")
            print(f"  Embedding shape: {row[embedding_col].shape if hasattr(row[embedding_col], 'shape') else 'N/A'}")
            
            # Check if this word is in word_to_class
            print(f"  In word_to_class: {first_missing in word_to_class}")
            
            # Check if this word appears in any cluster
            found_in_clusters = []
            for word, classes in word_to_class.items():
                if word == first_missing:
                    found_in_clusters = classes
                    break
            print(f"  Cluster assignments: {found_in_clusters if found_in_clusters else 'None'}")
        df_valid["is_bridge_word"] = df_valid["synonym_classes"].apply(lambda x: len(x) > 1 if x else False)
        
        # Cluster size distribution (counting bridge words in all their clusters)
        cluster_sizes = {}
        for classes in word_to_class.values():
            for cid in classes:
                cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
        
        sizes = list(cluster_sizes.values())
        print(f"Cluster size distribution:")
        print(f"  Min: {min(sizes)}, Max: {max(sizes)}, Mean: {np.mean(sizes):.1f}, Median: {np.median(sizes):.1f}")
        

        # Aggregate embeddings
        print(f"Aggregating embeddings using '{aggregation_method}' method...")
        sample_shapes = [df_valid[embedding_col].iloc[i].shape for i in range(min(3, len(df_valid)))]
        print(f"Sample embedding shapes before aggregation: {sample_shapes}")
        
        embeddings_list = []
        for emb in tqdm(df_valid[embedding_col], desc="Aggregating"):
            # print("New Sample")
            # print(type(emb))
            # print(emb)
            try:
                embeddings_list.append(self.aggregate_embeddings(emb, method=aggregation_method))
            except Exception as e:
                print("Error when aggregating")
                print(emb)
                break

        print(f"Sample embedding shape after aggregation: {embeddings_list[0].shape}")
        
        embeddings_matrix = np.vstack(embeddings_list).astype(np.float32)
        print(f"Embedding matrix shape: {embeddings_matrix.shape}")
        
        if embeddings_matrix.shape[0] != len(df_valid):
            raise ValueError(f"Embedding matrix has {embeddings_matrix.shape[0]} rows but expected {len(df_valid)}")
        
        # Normalize embeddings
        print("Normalizing embeddings...")
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_normalized = embeddings_matrix / norms
        
        words = df_valid[label_col].tolist()
        
        # Compute metrics
        print("Computing synonym class metrics...")
        synonym_metrics = self.compute_cluster_metrics(embeddings_normalized, word_to_class, words)
        
        print(f"  Intra-cluster similarity: {synonym_metrics['mean_intra_similarity']:.4f}")
        print(f"  Inter-cluster similarity: {synonym_metrics['mean_inter_similarity']:.4f}")
        print(f"  Separation gap: {synonym_metrics['separation_gap']:.4f}")
        
        # ------------------------------------------------------------------------------------
        # Linear Probe: Can a linear classifier distinguish same-cluster pairs?
        print("\n" + "=" * 60)
        print("LINEAR PROBE")
        print("=" * 60)


        n_pairs_per_class = int(len(words) / 4)
        np.random.seed(42)

        # Build cluster -> word indices mapping
        cluster_to_word_indices = {}
        for i, word in enumerate(words):
            for cid in word_to_class[word]:
                cluster_to_word_indices.setdefault(cid, []).append(i)

        # Collect ALL same-cluster pairs
        all_same_pairs = []
        for cid, indices in cluster_to_word_indices.items():
            if len(indices) >= 2:
                all_same_pairs.extend(combinations(indices, 2))

        print(f"Total possible same-cluster pairs: {len(all_same_pairs)}")

        # Collect ALL different-cluster pairs (by building set of same-pairs for fast lookup)
        same_pairs_set = set(all_same_pairs)
        same_pairs_set.update((j, i) for i, j in all_same_pairs)  # Add reverse pairs

        all_diff_pairs = []
        for i, j in combinations(range(len(words)), 2):
            if (i, j) not in same_pairs_set:
                all_diff_pairs.append((i, j))

        print(f"Total possible different-cluster pairs: {len(all_diff_pairs)}")

        # Sample equal numbers from each
        n_same = min(n_pairs_per_class, len(all_same_pairs))
        n_diff = min(n_pairs_per_class, len(all_diff_pairs))
        n_each = min(n_same, n_diff)  # Use the smaller of the two

        same_indices = np.random.choice(len(all_same_pairs), n_each, replace=False)
        diff_indices = np.random.choice(len(all_diff_pairs), n_each, replace=False)

        sampled_same = [all_same_pairs[i] for i in same_indices]
        sampled_diff = [all_diff_pairs[i] for i in diff_indices]

        # Build features and labels
        X_pairs = []
        y_pairs = []

        for i, j in sampled_same:
            X_pairs.append(np.abs(embeddings_normalized[i] - embeddings_normalized[j]))
            y_pairs.append(1)

        for i, j in sampled_diff:
            X_pairs.append(np.abs(embeddings_normalized[i] - embeddings_normalized[j]))
            y_pairs.append(0)

        X_pairs = np.array(X_pairs)
        y_pairs = np.array(y_pairs)

        # Shuffle
        shuffle_idx = np.random.permutation(len(y_pairs))
        X_pairs = X_pairs[shuffle_idx]
        y_pairs = y_pairs[shuffle_idx]

        print(f"Sampled {n_each} pairs per class ({2 * n_each} total)")
        self.report.print(f"Sampled {n_each} pairs per class ({2 * n_each} total)")

        # Train and evaluate
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X_pairs, y_pairs, cv=5, scoring='accuracy')

        print(f"\nLinear Probe Results (5-fold CV):")
        print(f"  Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"  Baseline (random): 0.500")
        self.report.print(f"\nLinear Probe Results (5-fold CV):")
        self.report.print(f"  Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        self.report.print(f"  Baseline (random): 0.500")
        print(f"  Lift over baseline: {scores.mean() - 0.5:+.3f}")
        # self.report.print(f"Linear Probe Accuracy (balanced): {scores.mean():.3f} ± {scores.std():.3f} <br>")
        # ------------------------------------------------------------------------------------

        results = {
            "df": df_valid,
            "words": words,
            "embeddings": embeddings_normalized,
            "word_to_class": word_to_class,
            "synonym_metrics": synonym_metrics,
            "clustering_strategy": clustering_strategy,
            "kmeans_labels": None,
            "kmeans_metrics": None,
            "comparison_metrics": None,
        }
        
        self.plot_centroid_heatmap(results, save_path=os.path.join(self.output_directory, f"cluster_heatmap.png"))

        # K-Means
        if run_kmeans and n_classes < len(df_valid):
            self.report.add_heading("K-Means Cluster Analysis", level=4)
            print(f"\nRunning MiniBatchKMeans with K={n_classes}...")
            kmeans = MiniBatchKMeans(
                n_clusters=n_classes,
                random_state=kmeans_random_state,
                batch_size=min(10000, len(df_valid)),
                n_init=3
            )
            print(f"Embedding space has {len(embeddings_normalized)} embeddings.")
            kmeans_labels = kmeans.fit_predict(embeddings_normalized)
            df_valid["kmeans_cluster"] = kmeans_labels
            
            # Convert kmeans labels to same format (list of single cluster)
            kmeans_word_to_class = {word: [int(kmeans_labels[i])] for i, word in enumerate(words)}
            
            kmeans_metrics = self.compute_cluster_metrics(embeddings_normalized, kmeans_word_to_class, words)
            
            print(f"  Intra-cluster similarity: {kmeans_metrics['mean_intra_similarity']:.4f}")
            print(f"  Inter-cluster similarity: {kmeans_metrics['mean_inter_similarity']:.4f}")
            print(f"  Separation gap: {kmeans_metrics['separation_gap']:.4f}")
            
            # For comparison metrics, use primary cluster (first one) for each word
            primary_labels = np.array([word_to_class[w][0] for w in words])
            
            comparison_metrics = {
                "adjusted_rand_index": adjusted_rand_score(primary_labels, kmeans_labels),
                "normalized_mutual_info": normalized_mutual_info_score(
                    primary_labels, kmeans_labels, average_method="arithmetic"
                ),
            }
            
            

            print(f"\nClustering comparison: <br>")
            print(f"  Adjusted Rand Index: {comparison_metrics['adjusted_rand_index']:.4f} <br>")
            print(f"  Normalized Mutual Info: {comparison_metrics['normalized_mutual_info']:.4f} <br>")
            self.report.print(f"\nClustering comparison: <br>")
            self.report.print(f"  Adjusted Rand Index: {comparison_metrics['adjusted_rand_index']:.4f} <br>")
            self.report.print(f"  Normalized Mutual Info: {comparison_metrics['normalized_mutual_info']:.4f} <br>")
            
            results["kmeans_labels"] = kmeans_labels
            results["kmeans_metrics"] = kmeans_metrics
            results["comparison_metrics"] = comparison_metrics
            results["df"] = df_valid
        
        return results

    def plot_centroid_heatmap(
        self,
        results: dict,
        figsize: tuple = (12, 10),
        cmap: str = "RdYlBu",
        max_clusters_display: int = 200,
        save_path: str = None,
    ) -> go.Figure:
        import plotly.graph_objects as go

        centroids = results["synonym_metrics"]["centroids"]
        cluster_ids = results["synonym_metrics"]["centroid_cluster_ids"]
        strategy = results.get("clustering_strategy", "unknown")

        if centroids is None or len(centroids) == 0:
            print("No centroids available for plotting")
            return None

        n_clusters = len(cluster_ids)

        # Compute centroid similarity matrix
        centroid_sim = cosine_similarity(centroids)

        # If too many clusters, show subset
        if n_clusters > max_clusters_display:
            print(f"Showing top {max_clusters_display} clusters by size")
            centroid_sim = centroid_sim[:max_clusters_display, :max_clusters_display]
            cluster_ids = cluster_ids[:max_clusters_display]

        show_labels = len(cluster_ids) <= 50
        tick_labels = [str(cid) for cid in cluster_ids] if show_labels else []

        fig = go.Figure(
            data=go.Heatmap(
                z=centroid_sim,
                x=tick_labels,
                y=tick_labels,
                colorscale=cmap,
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Cosine Similarity"),
                hovertemplate="Cluster X: %{x}<br>Cluster Y: %{y}<br>Similarity: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Cluster Centroid Similarities ({strategy})<br>{len(cluster_ids)} clusters shown",
                font=dict(size=14),
            ),
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=8),
                showticklabels=show_labels,
            ),
            yaxis=dict(
                tickfont=dict(size=8),
                showticklabels=show_labels,
                autorange="reversed",  # Match seaborn's top-to-bottom orientation
            ),
        )

        self.report.add_figure(fig)

        # if save_path:
        #     if save_path.endswith(".html"):
        #         fig.write_html(save_path)
        #     else:
        #         fig.write_image(save_path)
        #     print(f"Saved heatmap to {save_path}")

        return fig

    def save_analysis_results(
        self,
        results: dict,
        prefix: str = "synonym_cluster_analysis",
        label_col: str = "Word",
    ):
        """Save all analysis results to files."""
        import json
        
        strategy = results.get("clustering_strategy", "unknown")
        prefix = f"{prefix}_{strategy}"
        
        # Save plots
        self.plot_centroid_heatmap(results, save_path=os.path.join(self.output_directory, f"{prefix}_heatmap.png"))
        # self.plot_metrics_summary(results, save_path=os.path.join(self.output_directory, f"{prefix}_metrics.png"))
        
        # Save cluster details
        cluster_details = get_cluster_details(results, label_col)
        details_path = os.path.join(self.output_directory, f"{prefix}_cluster_details.csv")
        cluster_details.to_csv(details_path, index=False)
        print(f"Saved cluster details to {details_path}")
        
        # Save bridge words list
        bridge_words = [(w, classes) for w, classes in results["word_to_class"].items() if len(classes) > 1]
        if bridge_words:
            bridge_df = pd.DataFrame(bridge_words, columns=["word", "clusters"])
            bridge_df["clusters"] = bridge_df["clusters"].apply(str)
            bridge_path = os.path.join(self.output_directory, f"{prefix}_bridge_words.csv")
            bridge_df.to_csv(bridge_path, index=False)
            print(f"Saved {len(bridge_words)} bridge words to {bridge_path}")
        
        # Save metrics as JSON
        metrics = {
            "clustering_strategy": strategy,
            "synonym_metrics": {
                k: v for k, v in results["synonym_metrics"].items()
                if k not in ["per_cluster_intra", "centroids", "centroid_cluster_ids"]
            },
        }
        
        if results.get("kmeans_metrics"):
            metrics["kmeans_metrics"] = {
                k: v for k, v in results["kmeans_metrics"].items()
                if k not in ["per_cluster_intra", "centroids", "centroid_cluster_ids"]
            }
        
        if results.get("comparison_metrics"):
            metrics["comparison_metrics"] = results["comparison_metrics"]
        
        metrics_path = os.path.join(self.output_directory, f"{prefix}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")



    # ======================================
    # ========== Analogy Analysis ==========
    # ======================================
    def analyze_analogies_from_csv(
        self,
        analogy_csv_path: str,
        df: pd.DataFrame = None,
        embedding_col: str = "patch_embeddings_array",
        aggregation_method: str = "mean",
        label_col: str = "Word",
        top_k_values: list = [1, 5, 10],
        verbose: bool = True,
        output_path: str = None,
        category_filter: list[str] | None = None,
        sample_per_category: int | None = None,
        sample_seed: int = 42,
        encode_missing_words: bool = False,
    ):
        """
        Test word analogies from a CSV file and report statistics per category.

                
        word_1 is to word_2 as word_3 is to word_4
        man is to women as king is to queen

        word_1 - word_2 + word_3 =? word_4
        man - women + king = queen? no
        
        it should be
        word_3 - word_1 + word_2 = word_4
        king - man + women = queen

        
        Args:
            df: DataFrame with embeddings (from load_parquet_with_embeddings)
            analogy_csv_path: Path to CSV with analogy tests
            embedding_col: Column containing embeddings
            aggregation_method: How to aggregate variable-length embeddings
            label_col: Column containing word labels
            top_k_values: List of k values for top-k accuracy reporting
            verbose: Whether to print detailed results
            output_path: Path to save detailed results CSV (optional)
        
        Returns:
            dict with per-category and overall statistics
        """
        if df is None:
            assert self.data_manager.get_current_dataframe() is not None
            df = self.data_manager.get_current_dataframe()

        analysis_df = df
        if encode_missing_words:
            self.assert_model_loaded()

            current_model_type = self.data_manager.current_model_type
            current_subsample = self.data_manager.current_subsample
            current_decomposition = self.data_manager.current_decomposition

            if current_decomposition == "rbf":
                try:
                    self.data_manager.prepare_data(
                        current_model_type,
                        subsample=current_subsample,
                        decomposition=None,
                        compute_nn=False,
                    )
                    analysis_df = self.data_manager.get_current_dataframe().copy()
                finally:
                    self.data_manager.current_model_type = current_model_type
                    self.data_manager.current_subsample = current_subsample
                    self.data_manager.current_decomposition = current_decomposition
            elif current_decomposition is not None:
                raise NotImplementedError(
                    "Dynamic analogy encoding is currently only supported for the "
                    f"'rbf' decomposition, not '{current_decomposition}'."
                )

        # Sanity check
        df_valid = analysis_df[analysis_df[embedding_col].notna()].copy()
        if len(df_valid) == 0:
            raise ValueError("No valid embeddings found")

        df_valid = df_valid[df_valid[label_col].notna()].copy()
        df_valid[label_col] = df_valid[label_col].astype(str).str.strip()
        df_valid = df_valid[df_valid[label_col] != ""]
        df_valid = df_valid.drop_duplicates(subset=[label_col], keep="first").reset_index(drop=True)

        # Load analogy CSV
        analogy_df = pd.read_csv(analogy_csv_path)
        if category_filter is not None:
            analogy_df = analogy_df[~analogy_df["category"].isin(category_filter)]

        if sample_per_category is not None:
            sampled_groups = [
                group.sample(
                    n=min(sample_per_category, len(group)),
                    random_state=sample_seed,
                )
                for _, group in analogy_df.groupby("category")
            ]
            analogy_df = pd.concat(sampled_groups, ignore_index=True)

        counts = analogy_df["category"].value_counts()


        fig = go.Figure(go.Bar(x=counts.index, y=counts.values))
        fig.update_layout(
            xaxis_title="Category", 
            yaxis_title="Count", 
            title="Category Distribution",
            template="plotly_white",
            width=self.report.default_width,
            height=self.report.default_height,
        )
        self.report.add_figure(fig)


        analogy_words = set(analogy_df[["word_1", "word_2", "word_3", "word_4"]].values.flatten())
        df_valid = df_valid[df_valid[label_col].isin(analogy_words)].copy()

        encoded_missing_word_count = 0
        if encode_missing_words:
            existing_words = set(df_valid[label_col])
            missing_words = sorted(analogy_words - existing_words)

            if missing_words:
                print(f"Encoding {len(missing_words)} missing analogy words...")
                new_rows = []
                for word in tqdm(missing_words, desc="Encoding missing analogy words"):
                    with torch.no_grad():
                        result = self.encoder_model.encode_text(word, device=self.pipeline_device)
                    patch_emb = self._tensor_to_numpy_float32(result["patch_embeddings"])
                    new_rows.append({
                        label_col: word,
                        embedding_col: patch_emb,
                    })

                if new_rows:
                    df_valid = pd.concat([df_valid, pd.DataFrame(new_rows)], ignore_index=True)
                    encoded_missing_word_count = len(new_rows)

            if self.data_manager.current_decomposition == "rbf":
                df_valid = self.data_manager.apply_rbf(
                    df=df_valid,
                    embedding_col=embedding_col,
                    aggregation_method=aggregation_method,
                    n_components=200,
                    gamma=0.5,
                    random_state=42,
                )


        embeddings = df_valid[embedding_col].apply(
            lambda x: self.aggregate_embeddings(x, method=aggregation_method)
        )
        X = np.vstack(embeddings.values)
        labels = df_valid[label_col].values
        
        # Create word to embedding mapping
        word_to_idx = {word: idx for idx, word in enumerate(labels)}
        word_to_emb = {word: X[idx] for idx, word in enumerate(labels)}
        vocabulary = set(labels)


        # Sanity check columns
        required_cols = ["category", "word_1", "word_2", "word_3", "word_4"]
        missing_cols = [col for col in required_cols if col not in analogy_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in analogy CSV: {missing_cols}")
        
        print("=" * 70)
        print("WORD ANALOGY ANALYSIS")
        print("=" * 70)
        print(f"\nLoaded {len(analogy_df)} analogy tests from {analogy_csv_path}")
        if sample_per_category is not None:
            print(f"Sampled up to {sample_per_category} analogies per category with seed {sample_seed}")
        if encode_missing_words:
            print(f"Encoded missing analogy words: {encoded_missing_word_count}")
        print(f"Vocabulary size: {len(vocabulary)}")
        print(f"Categories: {analogy_df['category'].nunique()}")
        
        
        category_results = defaultdict(lambda: {
            "total": 0,
            "valid": 0,  # All 4 words in vocabulary
            "skipped": 0,  # Missing words
            "correct_at_k": {k: 0 for k in top_k_values},
            "reciprocal_ranks": [],
            "details": [],
        })
        

        # Process each analogy
        for idx, row in tqdm(analogy_df.iterrows(), total=len(analogy_df)):
            category = row["category"]
            word_1, word_2, word_3, word_4 = row["word_1"], row["word_2"], row["word_3"], row["word_4"]
            
            category_results[category]["total"] += 1
            
            # Check if all words are in vocabulary
            words = [word_1, word_2, word_3, word_4]
            missing = [w for w in words if w not in vocabulary]
            
            if missing:
                category_results[category]["skipped"] += 1
                category_results[category]["details"].append({
                    "word_1": word_1, "word_2": word_2, 
                    "word_3": word_3, "word_4": word_4,
                    "status": "skipped",
                    "missing_words": missing,
                    "predicted": None,
                    "rank": None,
                })
                continue
            
            category_results[category]["valid"] += 1
            
            # Compute analogy vector: word_1 - word_2 + word_3 ≈ word_4
            # analogy_vec = word_to_emb[word_1] - word_to_emb[word_2] + word_to_emb[word_3]
            # Compute analogy vector: word_3 - word_1 + word_2 ≈ word_4
            analogy_vec = word_to_emb[word_3] - word_to_emb[word_1] + word_to_emb[word_2]
            
            # Find most similar words
            similarities = cosine_similarity([analogy_vec], X)[0]
            
            # Don't match self
            for word in [word_1, word_2, word_3]:
                similarities[word_to_idx[word]] = -np.inf
            
            # Get ranking
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_words = [labels[i] for i in sorted_indices]
            
            # Find rank of expected word (word_4)
            try:
                rank = sorted_words.index(word_4) + 1  # 1-indexed
            except ValueError:
                rank = len(sorted_words) + 1  # Not found
            
            # Track top-k accuracy
            for k in top_k_values:
                if rank <= k:
                    category_results[category]["correct_at_k"][k] += 1
            
            # Track reciprocal rank (for MRR)
            category_results[category]["reciprocal_ranks"].append(1.0 / rank)
            
            # Store details
            top_n = min(5, len(sorted_words))
            top_5_predictions = [(sorted_words[i], similarities[sorted_indices[i]]) for i in range(top_n)]
            category_results[category]["details"].append({
                "word_1": word_1, "word_2": word_2, 
                "word_3": word_3, "word_4": word_4,
                "status": "evaluated",
                "missing_words": [],
                "predicted": sorted_words[0],
                "predicted_similarity": similarities[sorted_indices[0]],
                "expected_rank": rank,
                "top_5": top_5_predictions,
            })
        
        # Calculate statistics per category
        if verbose:
            print(f"\n{'=' * 70}")
            print("RESULTS BY CATEGORY")
            print(f"{'=' * 70}")

        # Run per category    
        category_stats = {}

        # Vocab Coverage
        melted = analogy_df.melt(
            id_vars="category",
            value_vars=["word_1", "word_2", "word_3", "word_4"],
            value_name="word",
        ).drop(columns="variable")

        melted = melted.drop_duplicates(subset=["category", "word"])

        category_counts = (
            melted.groupby("category")["word"]
            .count()
            .reset_index()
            .rename(columns={"word": "unique_words"})
        )

        print("=" * 70)

        print(category_counts)
    
        
        for category in sorted(category_results.keys()):
            results = category_results[category]
            valid = results["valid"]
            
            if valid > 0:
                stats = {
                    "total": results["total"],
                    "valid": valid,
                    "skipped": results["skipped"],
                    "coverage": valid / results["total"] * 100,
                    "mrr": np.mean(results["reciprocal_ranks"]),
                    "vocab_count": int(category_counts.loc[category_counts['category'] == category]["unique_words"].item())
                }

                # print("=" * 70 + "STATS")
                # print(stats)

                for k in top_k_values:
                    stats[f"accuracy_at_{k}"] = results["correct_at_k"][k] / valid * 100
                
                category_stats[category] = stats
                
                if verbose:
                    print(f"\n{'-' * 70}")
                    print(f"Category: {category}")
                    print(f"{'-' * 70}")
                    print(f"  Total analogies: {stats['total']}")
                    print(f"  Valid (all words in vocab): {stats['valid']} ({stats['coverage']:.1f}%)")
                    print(f"  Skipped (missing words): {stats['skipped']}")
                    print(f"\n  Accuracy:")
                    for k in top_k_values:
                        acc = stats[f'accuracy_at_{k}']
                        correct = results['correct_at_k'][k]
                        print(f"    Top-{k}: {acc:.2f}% ({correct}/{valid})")
                    print(f"  MRR (Mean Reciprocal Rank): {stats['mrr']:.4f}")
        
        # Calculate overall statistics
        overall_valid = sum(r["valid"] for r in category_results.values())
        overall_total = sum(r["total"] for r in category_results.values())
        overall_skipped = sum(r["skipped"] for r in category_results.values())
        all_reciprocal_ranks = [rr for r in category_results.values() for rr in r["reciprocal_ranks"]]
        
        overall_stats = {
            "total": overall_total,
            "valid": overall_valid,
            "skipped": overall_skipped,
            "coverage": overall_valid / overall_total * 100 if overall_total > 0 else 0,
            "mrr": np.mean(all_reciprocal_ranks) if all_reciprocal_ranks else 0,
        }
        
        for k in top_k_values:
            overall_correct = sum(r["correct_at_k"][k] for r in category_results.values())
            overall_stats[f"accuracy_at_{k}"] = overall_correct / overall_valid * 100 if overall_valid > 0 else 0
        
        if verbose:
            print(f"\n{'=' * 70}")
            print("OVERALL STATISTICS")
            print(f"{'=' * 70}")
            print(f"  Total analogies: {overall_stats['total']}")
            print(f"  Valid (all words in vocab): {overall_stats['valid']} ({overall_stats['coverage']:.1f}%)")
            print(f"  Skipped (missing words): {overall_stats['skipped']}")
            print(f"\n  Accuracy:")
            for k in top_k_values:
                acc = overall_stats[f'accuracy_at_{k}']
                correct = sum(r["correct_at_k"][k] for r in category_results.values())
                print(f"    Top-{k}: {acc:.2f}% ({correct}/{overall_valid})")
            print(f"  MRR (Mean Reciprocal Rank): {overall_stats['mrr']:.4f}")
        
        # Create summary DataFrame
        summary_rows = []
        for category, stats in category_stats.items():
            row = {"category": category, **stats}
            summary_rows.append(row)
        
        # Add overall row
        summary_rows.append({"category": "OVERALL", **overall_stats})
        summary_df = pd.DataFrame(summary_rows)
        
        if verbose:
            print(f"\n{'=' * 70}")
            print("SUMMARY TABLE")
            print(f"{'=' * 70}")
            print(summary_df.to_string(index=False))
        
        # Save detailed results if requested
        if output_path:
            # Flatten details for CSV export
            detail_rows = []
            for category, results in category_results.items():
                for detail in results["details"]:
                    detail_row = {
                        "category": category,
                        "word_1": detail["word_1"],
                        "word_2": detail["word_2"],
                        "word_3": detail["word_3"],
                        "word_4_expected": detail["word_4"],
                        "status": detail["status"],
                        "missing_words": ", ".join(detail["missing_words"]) if detail["missing_words"] else "",
                        "word_4_predicted": detail.get("predicted"),
                        "expected_rank": detail.get("expected_rank"),
                    }
                    
                    # Add top 5 predictions
                    if detail.get("top_5"):
                        for i, (word, sim) in enumerate(detail["top_5"]):
                            detail_row[f"pred_{i+1}"] = word
                            detail_row[f"pred_{i+1}_sim"] = f"{sim:.4f}"
                    
                    detail_rows.append(detail_row)
            
            detail_df = pd.DataFrame(detail_rows)
            detail_df.to_csv(output_path, index=False)
            print(f"\nDetailed results saved to {output_path}")
            
            # Also save summary
            summary_path = output_path.replace(".csv", "_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary saved to {summary_path}")
        
        return {
            "category_stats": category_stats,
            "overall_stats": overall_stats,
            "category_results": dict(category_results),
            "summary_df": summary_df,
            "vocabulary": vocabulary,
            "sample_per_category": sample_per_category,
            "sample_seed": sample_seed,
            "encoded_missing_word_count": encoded_missing_word_count,
        }

    def visualize_analogy_results(
        self,
        results: dict,
        save_path: str = "analogy_results",
    ):
        """
        Visualize analogy test results using Plotly (four separate figures).
        """

        category_stats = results["category_stats"]
        overall_stats = results["overall_stats"]

        if not category_stats:
            print("No results to visualize")
            return

        categories = list(category_stats.keys())
        acc_keys = sorted([k for k in overall_stats.keys() if k.startswith("accuracy_at_")])
        acc_keys = sorted(acc_keys, key=lambda k: int(k.replace("accuracy_at_", "")))
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]


        fig1 = go.Figure()
        for i, k in enumerate(acc_keys[:3]):
            k_val = k.replace("accuracy_at_", "")
            values = [category_stats[c].get(k, 0) for c in categories]
            fig1.add_trace(go.Bar(
                x=categories,
                y=values,
                name=f"Top-{k_val}",
                marker_color=colors[i % len(colors)],
            ))
        fig1.update_layout(
            title="Accuracy at Different K by Category",
            yaxis_title="Accuracy (%)",
            barmode="group",
            xaxis_tickangle=-45,
            height=self.report.default_height,
            width=self.report.default_width,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )



        coverage = [category_stats[c].get("coverage", 0) for c in categories]
        valid = [category_stats[c].get("valid", 0) for c in categories]
        overall_coverage = overall_stats.get("coverage", 0)

        vocab_count = [category_stats[c]["vocab_count"] for c in categories]
        print(vocab_count)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=vocab_count,
            x=categories,
            # orientation="h",
            marker_color="orange",
            name="Distribution",
            text=[f"n={c}" for c in vocab_count],
            textposition="outside",
            textfont=dict(size=9),
        ))
        fig2.update_layout(
            title="Distribution of vocabulary by category",
            # xaxis_title="C",
            height=self.report.default_height,
            width=self.report.default_width,
            template="plotly_white",
        )

        # Save and show all figures
        for i, (fig, name) in enumerate(
            zip(
                [fig1, fig2],
                [f"accuracy_at_k ({sum(vocab_count)})", "distribution"],
            ),
            start=1,
        ):
            self.report.add_heading(name, level=5)
            self.report.add_figure(fig)


    
    # ======================================
    # ========== Helper Functions ==========
    # ======================================
    def load_csv(self, csv_name: str, set_current_dataframe, filter_out_duplicate_subset: Optional[List[str]]):
        df = pd.read_csv(os.path.join(self.data_directory, csv_name))
        if filter_out_duplicate_subset is not None:
            df = df.drop_duplicates(subset=filter_out_duplicate_subset)
        if set_current_dataframe:
            self.current_dataframe = df
        return df

    def load_parquet_with_embeddings(self, parquet_name, set_current_dataframe, filter_out_duplicate_subset: Optional[List[str]], hard_data_path: str = None):
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

        # print(df["Word"].count())
        # print(df["Word"].nunique())
        # print(df["synonyms"].count())
        
        if filter_out_duplicate_subset is not None:
            df = df.drop_duplicates(subset=filter_out_duplicate_subset)
            # print(filter_out_duplicate_subset)
            # print(df["Word"].count())
            # print(df["Word"].nunique())
            # print(df["synonyms"].count())
        if set_current_dataframe:
            self.current_dataframe = df
            print(df.head())
        return df

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

    def parse_synonyms(
        self,
        synonym_value,
        delimiter: str = ";"
    ) -> list[str]:
        """
        Parse synonym string into list of synonyms.
        TODO: might need to adjust this to account for | in alias words
        """
        if pd.isna(synonym_value) or synonym_value == "":
            return []
        return [s.strip() for s in str(synonym_value).split(delimiter) if s.strip()]



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
        n_components: int = 50,
        gamma: float = 0.1,
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
            agg = self.aggregate_embeddings(emb)
            aggregated.append(agg)
        
        embeddings = np.vstack(aggregated)
        print(f"Input shape: {embeddings.shape}")
        
        # Apply RBF mapping
        print(f"Fitting RBF mapping (gamma={gamma}, n_components={n_components})...")
        rbf_map = RBFSampler(gamma=gamma, n_components=n_components)
        transformed = rbf_map.fit_transform(embeddings)
        print(f"Output shape: {transformed.shape}")
        
        if output_col is not None:
            df[output_col] = list(transformed)
        else:
            df[embedding_col] = list(transformed)

        self.current_dataframe = df
        
        return df


    
    # ======================================
    # =========== Plotly Manager ===========
    # ======================================
    class Report():
        report_path = None
        report_name = None
        output_directory: str = None

        default_width: int = 1000
        default_height: int = 500

        def __init__(self, output_directory):
            self.output_directory = output_directory
        
        def start_report(self, report_name, overwrite):
            self.report_name = report_name.replace(".html", "")
            try:
                os.mkdir(os.path.join(self.output_directory, self.report_name))
                print(f"Directory '{directory_name}' created successfully.")
            except Exception as e:
                print(f"An error occurred: {e}")
            self.set_report_path(os.path.join(self.output_directory, self.report_name, report_name))
            if overwrite:
                self.create_new_report()
            elif os.path.exists(os.path.join(self.output_directory, self.report_name, report_name)) and overwrite:
                self.create_new_report()

        def create_new_report(self):
            assert self.report_path is not None

            shutil.copy('output/base.html', self.report_path)

        def write(self):
            assert self.report_path is not None

        def add_figure(self, fig):
            assert self.report_path is not None
            
            if os.path.exists(self.report_path):
                with open(self.report_path, 'a') as f:
                    f.write(fig.to_html(full_html=False, include_plotlyjs=False))
            else:
                fig.write_html(self.report_path)

        def add_heading(self, heading_text: str, level=1):
            if os.path.exists(self.report_path):
                with open(self.report_path, 'a') as f:
                    f.write(f'<h{level}>{heading_text}</h{level}>')
            else:
                assert False

        def start_paragraph(self):
            self.print("<p>")

        def end_paragraph(self):
            self.print("</p>")

        def print(self, text: str):
            if os.path.exists(self.report_path):
                with open(self.report_path, 'a') as f:
                    f.write(f'{text}<br>')
            else:
                assert False


        def set_report_path(self, report_path: str):
            self.report_path = report_path



    # ======================================
    # ========== Getter / Setter  ==========
    # ======================================
    def set_data_directory(self, data_directory):
        self.data_directory = data_directory
    
    def set_output_directory(self, output_directory):
        self.output_directory = output_directory

    def set_verbose(self, verbose):
        self.verbose = verbose
    
    def set_pipeline_device(self, device):
        self.pipeline_device = device
        # DataManager encodes on its own device attribute (default cuda:0), so the
        # device has to reach it too — otherwise the model sits on one GPU and the
        # input tensors are built on another.
        self.data_manager.pipeline_device = device

    def set_report_path(self, report_path):
        self.report.set_report_path(report_path)

    def use_dataset(self, dataset_name: str):
        self.dataset_name = dataset_name

    def set_encoder_type(self, encoder_type: str):
        self.encoder_type = encoder_type


    # ======================================
    # ========== Assert Functions ==========
    # ======================================
    def assert_set_directories(self):
        assert self.data_directory is not None
        assert self.output_directory is not None

    def assert_model_loaded(self):
        assert self.encoder_model is not None
