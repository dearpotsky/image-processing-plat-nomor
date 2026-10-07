from pathlib import Path

from app.pipeline.config import PipelineConfig
from app.pipeline.experiment import (
    ExperimentConfig,
    ExperimentRunner,
)
from app.pipeline.output_manager import (
    OutputManager,
)


IMAGE_PATHS = [
    "dataset/input/test001.jpg",
]


def create_experiment():

    config = PipelineConfig()

    config.enhancement.method = "clahe"
    config.enhancement.clip_limit = 2.0

    config.segmentation.method = "otsu"

    return ExperimentConfig(
        name="output_test",
        config=config,
    )


def main():

    print("=" * 70)
    print("OUTPUT MANAGER TEST")
    print("=" * 70)

    experiment = create_experiment()

    print()
    print("Experiment:")
    print(
        f"  {experiment.name}"
    )

    # ------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------

    runner = ExperimentRunner(
        output_dir="output/test"
    )

    print()
    print("Menjalankan pipeline...")

    results = runner.run(
        image_paths=IMAGE_PATHS,
        experiment_configs=[
            experiment
        ],
    )

    print(
        "Pipeline selesai."
    )

    # ------------------------------------------------------
    # Save result
    # ------------------------------------------------------

    experiment_result = results[0]

    output_manager = OutputManager(
        base_dir=(
            Path("output")
            / "test"
            / experiment.name
        )
    )

    for image_index, image_result in enumerate(
        experiment_result.result.images,
        start=1,
    ):

        image_name = (
            f"image_{image_index:02d}"
        )

        print()
        print(
            f"Menyimpan hasil "
            f"{image_name}..."
        )

        output_manager.save_pipeline_result(
            result=image_result,
            image_name=image_name,
            config=experiment_result.config,
        )

    # ------------------------------------------------------
    # Show output directory
    # ------------------------------------------------------

    output_dir = (
        Path("output")
        / "test"
        / experiment.name
    )

    print()
    print("=" * 70)
    print("OUTPUT SELESAI")
    print("=" * 70)

    print()
    print(
        f"Folder output:"
    )

    print(
        output_dir
    )

    # ------------------------------------------------------
    # List generated files
    # ------------------------------------------------------

    print()
    print("File yang dihasilkan:")
    print("-" * 70)

    if output_dir.exists():

        files = sorted(
            output_dir.rglob("*")
        )

        for file in files:

            if file.is_file():

                print(
                    file
                )

    print()
    print("=" * 70)
    print("OUTPUT MANAGER TEST SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()