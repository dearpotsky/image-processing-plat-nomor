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
        name="delete_test",
        config=config,
    )


def main():
    print("=" * 70)
    print("OUTPUT MANAGER DELETE TEST")
    print("=" * 70)

    experiment = create_experiment()

    # ============================================================
    # RUN PIPELINE
    # ============================================================

    runner = ExperimentRunner(
        output_dir="output"
    )

    print()
    print("Menjalankan pipeline...")

    results = runner.run(
        image_paths=IMAGE_PATHS,
        experiment_configs=[experiment],
    )

    print("Pipeline selesai.")

    experiment_result = results[0]

    # ============================================================
    # OUTPUT MANAGER
    # ============================================================

    output_manager = OutputManager(
        base_dir=(
            Path("output")
            / experiment.name
        )
    )

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    for image_index, image_result in enumerate(
        experiment_result.result.images,
        start=1,
    ):
        image_name = f"image_{image_index:02d}"

        output_manager.save_pipeline_result(
            result=image_result,
            image_name=image_name,
            config=experiment_result.config,
        )

    print()
    print("Hasil berhasil disimpan.")

    # ============================================================
    # LIST IMAGES
    # ============================================================

    print()
    print("Daftar image:")

    images = output_manager.list_images(
        result_name="."
    )

    for image in images:
        print(
            f"  - {image['name']}"
        )

    assert len(images) > 0

    # ============================================================
    # DELETE IMAGE
    # ============================================================

    image_name = images[0]["name"]

    print()
    print(
        f"Menghapus image: {image_name}"
    )

    output_manager.delete_image_result(
        result_name=".",
        image_name=image_name,
    )

    image_path = (
        output_manager.base_dir
        / image_name
    )

    assert not image_path.exists()

    print(
        "Delete image berhasil."
    )

    # ============================================================
    # DELETE EXPERIMENT
    # ============================================================

    print()
    print(
        f"Menghapus experiment: "
        f"{experiment.name}"
    )

    output_manager.delete_result()

    experiment_path = (
        Path("output")
        / experiment.name
    )

    assert not experiment_path.exists()

    print(
        "Delete experiment berhasil."
    )

    # ============================================================
    # FINISH
    # ============================================================

    print()
    print("=" * 70)
    print("DELETE TEST BERHASIL")
    print("=" * 70)


if __name__ == "__main__":
    main()