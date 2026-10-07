from copy import deepcopy

from app.pipeline.config import PipelineConfig
from app.pipeline.experiment import (
    ExperimentConfig,
    ExperimentRunner,
)
from app.pipeline.comparison import (
    ExperimentComparator,
)


# ============================================================
# INPUT
# ============================================================

IMAGE_PATHS = [
    "dataset/input/test001.jpg",
]


# ============================================================
# CREATE EXPERIMENTS
# ============================================================

def create_experiments():

    base_config = PipelineConfig()

    # --------------------------------------------------------
    # Experiment 1
    # --------------------------------------------------------

    config_1 = deepcopy(base_config)

    config_1.enhancement.method = "clahe"
    config_1.enhancement.clip_limit = 1.0

    config_1.segmentation.method = "otsu"

    experiment_1 = ExperimentConfig(
        name="clahe_1_0_otsu",
        config=config_1,
    )

    # --------------------------------------------------------
    # Experiment 2
    # --------------------------------------------------------

    config_2 = deepcopy(base_config)

    config_2.enhancement.method = "clahe"
    config_2.enhancement.clip_limit = 2.0

    config_2.segmentation.method = "otsu"

    experiment_2 = ExperimentConfig(
        name="clahe_2_0_otsu",
        config=config_2,
    )

    # --------------------------------------------------------
    # Experiment 3
    # --------------------------------------------------------

    config_3 = deepcopy(base_config)

    config_3.enhancement.method = "clahe"
    config_3.enhancement.clip_limit = 4.0

    config_3.segmentation.method = "otsu"

    experiment_3 = ExperimentConfig(
        name="clahe_4_0_otsu",
        config=config_3,
    )

    return [
        experiment_1,
        experiment_2,
        experiment_3,
    ]


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(comparisons):

    print()
    print("=" * 110)
    print("EXPERIMENT COMPARISON")
    print("=" * 110)

    header = (
        f"{'Experiment':<25}"
        f"{'Images':>8}"
        f"{'Plates':>8}"
        f"{'FG Ratio':>12}"
        f"{'Mean':>12}"
        f"{'Std':>12}"
        f"{'Components':>14}"
        f"{'Valid':>12}"
    )

    print(header)
    print("-" * 110)

    for result in comparisons:

        print(
            f"{result.name:<25}"
            f"{result.image_count:>8}"
            f"{result.plate_count:>8}"
            f"{result.avg_foreground_ratio:>12.4f}"
            f"{result.avg_mean_intensity:>12.2f}"
            f"{result.avg_std_intensity:>12.2f}"
            f"{result.avg_total_components:>14.2f}"
            f"{result.avg_valid_components:>12.2f}"
        )

    print("-" * 110)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXPERIMENT COMPARISON TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Buat experiment
    # --------------------------------------------------------

    experiments = create_experiments()

    print()
    print(
        f"Jumlah experiment : "
        f"{len(experiments)}"
    )

    # --------------------------------------------------------
    # Jalankan experiment
    # --------------------------------------------------------

    print()
    print("Menjalankan experiment...")

    runner = ExperimentRunner(
        output_dir="output/experiments"
    )

    results = runner.run(
        image_paths=IMAGE_PATHS,
        experiment_configs=experiments,
    )

    print(
        "Experiment selesai."
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print()
    print("Menghitung comparison...")

    comparator = ExperimentComparator()

    comparisons = comparator.compare(
        results
    )

    print(
        "Comparison selesai."
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_comparison(
        comparisons
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST COMPARISON SELESAI")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()