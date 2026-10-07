from pathlib import Path

from app.pipeline.config import PipelineConfig
from app.pipeline.experiment import (
    ExperimentConfig,
    ExperimentRunner,
)
from app.pipeline.output_manager import (
    OutputManager,
)
from app.pipeline.report import (
    ExperimentReporter,
)


IMAGE_PATHS = [
    "dataset/input/test001.jpg",
]


def create_experiments():

    base_config = PipelineConfig()

    # ------------------------------------------------------------
    # Experiment 1
    # ------------------------------------------------------------

    config_1 = PipelineConfig()

    config_1.enhancement.method = "clahe"
    config_1.enhancement.clip_limit = 1.0
    config_1.segmentation.method = "otsu"

    experiment_1 = ExperimentConfig(
        name="clahe_1_0_otsu",
        config=config_1,
    )

    # ------------------------------------------------------------
    # Experiment 2
    # ------------------------------------------------------------

    config_2 = PipelineConfig()

    config_2.enhancement.method = "clahe"
    config_2.enhancement.clip_limit = 2.0
    config_2.segmentation.method = "otsu"

    experiment_2 = ExperimentConfig(
        name="clahe_2_0_otsu",
        config=config_2,
    )

    # ------------------------------------------------------------
    # Experiment 3
    # ------------------------------------------------------------

    config_3 = PipelineConfig()

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


def main():

    print("=" * 70)
    print("EXPERIMENT REPORT TEST")
    print("=" * 70)

    experiments = create_experiments()

    # ============================================================
    # RUN EXPERIMENTS
    # ============================================================

    runner = ExperimentRunner(
        output_dir="output/report_test"
    )

    print()
    print("Menjalankan experiment...")

    results = runner.run(
        image_paths=IMAGE_PATHS,
        experiment_configs=experiments,
    )

    print("Experiment selesai.")

    # ============================================================
    # SAVE OUTPUT
    # ============================================================

    for experiment_result in results:

        output_manager = OutputManager(
            base_dir=(
                Path("output")
                / "report_test"
                / experiment_result.name
            )
        )

        for image_index, image_result in enumerate(
            experiment_result.result.images,
            start=1,
        ):

            image_name = (
                f"image_{image_index:02d}"
            )

            output_manager.save_pipeline_result(
                result=image_result,
                image_name=image_name,
                config=experiment_result.config,
            )

    # ============================================================
    # REPORT
    # ============================================================

    reporter = ExperimentReporter(
        output_dir="output/report_test"
    )

    print()
    print("Membuat report...")

    reports = reporter.generate(
        results
    )

    print("Report selesai.")

    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    print()
    print("=" * 70)
    print("REPORT SUMMARY")
    print("=" * 70)

    for report in reports:

        print()
        print(
            f"Experiment : "
            f"{report['experiment']}"
        )

        print(
            f"Image      : "
            f"{report['image_count']}"
        )

        print(
            f"Plate      : "
            f"{report['plate_count']}"
        )

    # ============================================================
    # CHECK FILES
    # ============================================================

    output_dir = Path(
        "output/report_test"
    )

    print()
    print("=" * 70)
    print("FILE HASIL")
    print("=" * 70)

    for file in sorted(
        output_dir.rglob("*")
    ):

        if file.is_file():
            print(file)

    # ============================================================
    # ASSERT
    # ============================================================

    comparison_file = (
        output_dir
        / "experiment_comparison.csv"
    )

    assert comparison_file.exists()

    for experiment in experiments:

        report_file = (
            output_dir
            / experiment.name
            / "report.json"
        )

        assert report_file.exists()

    print()
    print("=" * 70)
    print("EXPERIMENT REPORT TEST BERHASIL")
    print("=" * 70)


if __name__ == "__main__":
    main()