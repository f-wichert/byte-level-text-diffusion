from datetime import datetime
from pipeline.pipeline import Pipeline

SYNONYMS_DATASET = "synonyms"
CONTEXT_DATASET = "context"

# Which dataset types a task accepts; tasks without an entry accept only the
# synonyms (word-table) dataset.
TASK_ACCEPTED_DATASETS = {
    "covariance": (SYNONYMS_DATASET, CONTEXT_DATASET),
    "explained_variance": (SYNONYMS_DATASET, CONTEXT_DATASET),
}


def run_analysis(analysis):
    # Validate task/dataset pairings up front, before any model loading or
    # encoding happens — a config typo should fail immediately.
    for task in analysis["tasks"]:
        dataset_type = task["dataset"] if "dataset" in task.keys() else SYNONYMS_DATASET
        accepted = TASK_ACCEPTED_DATASETS.get(task["name"], (SYNONYMS_DATASET,))
        if dataset_type not in accepted:
            raise ValueError(
                f"Task '{task['name']}' does not accept dataset '{dataset_type}' "
                f"(accepted: {accepted})."
            )

    pipeline = Pipeline(
        data_directory=analysis["data_directory"],
        output_directory=analysis["output_directory"]
    )
    pipeline.set_data_directory(analysis["data_directory"])
    pipeline.set_output_directory(analysis["output_directory"])
    pipeline.set_pipeline_device(analysis["device"])
    # pipeline.set_encoder_type(encoder_type)
    pipeline.load_encoder_model(
        analysis["encoder_type"],
        ae_config=analysis["ae_config"] if "ae_config" in analysis else None,
        ae_checkpoint=analysis["ae_checkpoint"] if "ae_checkpoint" in analysis else None,
    )
    # pipeline.encoder_type may carry an AE-variant tag after loading, which
    # keeps reports of different checkpoints from overwriting each other.
    pipeline.start_report(f"{datetime.today().strftime('%Y-%m-%d')}_{pipeline.encoder_type}.html", "Latent Space Analysis")
    # pipeline.start_report(f"{datetime.today().strftime('%Y-%m-%d')}_{analysis['encoder_type']}.html", "Latent Space Analysis")

    for task in analysis["tasks"]:
        dataset_type = task["dataset"] if "dataset" in task.keys() else SYNONYMS_DATASET
        if dataset_type == CONTEXT_DATASET:
            subsample = None  # the context corpus has no subsampling semantics
        else:
            subsample = task["subsample"] if "subsample" in task.keys() else analysis["default_subsample"]

        pipeline.data_manager.prepare_data(
            analysis["encoder_type"],
            subsample=subsample,
            decomposition=task["decomp"] if "decomp" in task.keys() else analysis["default_decomposition"],
            compute_nn=task["compute_nn"] if "compute_nn" in task.keys() else dataset_type != CONTEXT_DATASET,
            dataset=dataset_type,
        )

        match task["name"]:
            case "pca":
                pipeline.report.add_heading("PCA", level=3)
                pipeline.report_current_dataset()
                result = pipeline.visualize_embeddings_pca(
                    embedding_col="patch_embeddings_array",
                    aggregation_method="mean",
                    n_components=2,
                    color_by_column="part_of_speech",
                    show_labels=False,
                    max_labels=30,
                    # save_path=f'output/pca.png',
                    # save_name=f'pca.png',
                    save=True,
                    show_plot=False,
                )

            case "t-sne" :
                pipeline.report.add_heading("T-SNE", level=3)
                pipeline.report_current_dataset()
                result = pipeline.visualize_embeddings_tsne(
                    embedding_col="patch_embeddings_array",
                    aggregation_method="mean",
                    color_by="part_of_speech",
                    perplexity=30,
                    show_labels=True,
                    max_labels=30,
                    # save_path=f'output/tsne_visualization.png',
                    save=True,
                    show_plot=True,
                )

            case "morphological_neighbour":
                pipeline.report.add_heading("Morphological Neighbour Analysis", level=3)
                pipeline.report_current_dataset()
                morphology_neighbour_results = pipeline.analyze_morphological_neighbors(
                    prefix_lengths=[2, 3, 4],
                    suffix_lengths=[2, 3, 4],
                )
                pipeline.plot_morphological_neighbor_percentages(morphology_neighbour_results)

            case "morphological_neighbour_by_pos":
                pipeline.report.add_heading("Morphological Neighbour Analysis by POS", level=3)
                pipeline.report_current_dataset()
                morphology_neighbour_pos_results = pipeline.analyze_morphological_neighbors_by_pos()
                pipeline.plot_morphological_neighbors_by_pos(morphology_neighbour_pos_results)

            case "synonym_basic":
                pipeline.report.add_heading("Synonym Basic Analysis", level=3)
                pipeline.report_current_dataset()
                synonym_results = pipeline.analyze_synonyms()

            case "synonym_to_neighbour":
                pipeline.report.add_heading("Synonym To Neighbour Analysis", level=3)
                pipeline.report_current_dataset()
                pipeline.analyze_neighbour_to_synonym_relationship(
                    n_neighbors = task["n_neighbors"] if "n_neighbors" in task.keys() else 10,
                    min_synonyms = 8,
                    n_examples = 5,
                )

            case "typo_robustness":
                pipeline.report.add_heading("Typo Robustness Analysis", level=3)
                pipeline.report_current_dataset()
                pipeline.analyze_typo_robustness(
                    max_words=None,
                    min_word_length=4,
                    n_examples=5,
                )

            case "isotropy":
                pipeline.report.add_heading("Isotropy Analysis", level=3)
                pipeline.report_current_dataset()
                pipeline.analyze_isotropy(
                    embedding_col="patch_embeddings_array",
                    aggregation_method="mean",
                )

            case "covariance":
                pipeline.report.add_heading("Covariance / Latent Geometry Analysis", level=3)
                pipeline.report_current_dataset()
                pipeline.analyze_covariance(
                    top_k=task["top_k"] if "top_k" in task.keys() else 32,
                    min_position=task["min_position"] if "min_position" in task.keys() else 8,
                    diffusion_t_window=task["t_window"] if "t_window" in task.keys() else True,
                )

            case "explained_variance":
                pipeline.report.add_heading("Explained Variance Analysis", level=3)
                pipeline.report_current_dataset()
                pipeline.analyze_explained_variance(
                    target_dims=task["target_dims"] if "target_dims" in task.keys() else None,
                    min_position=task["min_position"] if "min_position" in task.keys() else 8,
                )

            case "linear_probe_word_length":
                pipeline.report.add_heading("Word Length Linear Probe", level=3)
                pipeline.report_current_dataset()
                pipeline.analyze_linear_probe_word_length(
                    word_lengths=task["word_lengths"] if "word_lengths" in task else [4, 5, 6, 7, 8, 9, 10],
                    n_words_per_length=task["n_words_per_length"] if "n_words_per_length" in task else 100,
                    aggregation_method="mean",
                    n_examples=50,
                )

            case "synonym_cluster":
                pipeline.report.add_heading("Synonym Cluster Analysis", level=3)
                pipeline.report_current_dataset()
                synonym_cluster_results = pipeline.analyze_synonym_clusters(
                    embedding_col="patch_embeddings_array",
                    label_col="Word",
                    synonym_col="synonyms",
                    clustering_strategy="direct",
                    run_kmeans=True,
                    # number_of_random_samples_to_use=10000
                )

            case "analogy":
                pipeline.report.add_heading("Analogy Analysis", level=3)
                pipeline.report_current_dataset()
                analogy_results = pipeline.analyze_analogies_from_csv(
                    analogy_csv_path="data/questions-words.csv",
                    top_k_values=[1, 5, 10],
                    output_path="output/analogy_results_detailed.csv",
                    label_col="Word",
                    category_filter=["capital-world", "capital-common-countries", "city-in-state", "currency"],
                    sample_per_category=400,
                    sample_seed=42,
                    encode_missing_words=True,
                )

                pipeline.visualize_analogy_results(analogy_results)


            case _:
                print(f"ERROR: No task found named: {task['name']}!")
