from copy import deepcopy

from app.pipeline.config import PipelineConfig
from app.pipeline.experiment import (
    ExperimentConfig,
    ExperimentRunner,
)


# ============================================================
# INPUT IMAGE
# ============================================================

IMAGE_PATHS = [
    "dataset/input/test001.jpg",
]


# ============================================================
# CREATE EXPERIMENT CONFIG
# ============================================================

def create_experiments():
    """
    Membuat konfigurasi eksperimen.

    Setiap eksperimen menggunakan pipeline yang sama,
    tetapi parameter enhancement/segmentation berbeda.
    """

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
# PRINT EXPERIMENT CONFIGURATION
# ============================================================

def print_experiment_configurations(
    experiments,
):
    print()
    print("=" * 70)
    print("EXPERIMENT CONFIGURATIONS")
    print("=" * 70)

    for index, experiment in enumerate(
        experiments,
        start=1,
    ):
        config = experiment.config

        print()
        print(f"Experiment #{index}")
        print(f"Name          : {experiment.name}")
        print(
            f"Enhancement   : "
            f"{config.enhancement.method}"
        )
        print(
            f"Clip limit    : "
            f"{config.enhancement.clip_limit}"
        )
        print(
            f"Segmentation  : "
            f"{config.segmentation.method}"
        )


# ============================================================
# PRINT RESULT
# ============================================================

def print_results(results):
    print()
    print("=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)

    for experiment in results:

        print()
        print("-" * 70)
        print(
            f"Experiment : "
            f"{experiment.name}"
        )
        print("-" * 70)

        batch_result = experiment.result

        print(
            f"Total image : "
            f"{batch_result.image_count}"
        )

        print(
            f"Total plate : "
            f"{batch_result.total_plate_count}"
        )

        for image_index, image_result in enumerate(
            batch_result.images,
            start=1,
        ):

            print()
            print(
                f"Image #{image_index}"
            )

            print(
                f"Jumlah plate : "
                f"{image_result.plate_count}"
            )

            for plate in image_result.plates:

                print(
                    f"  Plate #{plate.index}"
                )

                print(
                    f"    Confidence : "
                    f"{plate.confidence:.3f}"
                )

                features = plate.features

                threshold = features.get(
                    "threshold"
                )

                if threshold is None:
                    threshold_text = "N/A"
                else:
                    threshold_text = (
                        f"{threshold:.2f}"
                    )

                print(
                    f"    Threshold  : "
                    f"{threshold_text}"
                )

                print(
                    f"    FG ratio   : "
                    f"{features.get('foreground_ratio', 'N/A')}"
                )

                print(
                    f"    Components : "
                    f"{features.get('total_components', 'N/A')}"
                )

                print(
                    f"    Valid      : "
                    f"{features.get('valid_components', 'N/A')}"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXPERIMENT PIPELINE TEST")
    print("=" * 70)

    print()
    print("File berhasil dijalankan.")

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    print()
    print("Input images")
    print("-" * 70)

    for index, image_path in enumerate(
        IMAGE_PATHS,
        start=1,
    ):
        print(
            f"{index}. {image_path}"
        )

    # --------------------------------------------------------
    # Create experiments
    # --------------------------------------------------------

    experiments = create_experiments()

    print_experiment_configurations(
        experiments
    )

    # --------------------------------------------------------
    # Create runner
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MEMBUAT EXPERIMENT RUNNER")
    print("=" * 70)

    runner = ExperimentRunner(
        output_dir="output/experiments"
    )

    print(
        "ExperimentRunner berhasil dibuat."
    )

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MENJALANKAN EXPERIMENT")
    print("=" * 70)

    results = runner.run(
        image_paths=IMAGE_PATHS,
        experiment_configs=experiments,
    )

    print(
        "Semua experiment selesai."
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_results(
        results
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST EXPERIMENT SELESAI")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()