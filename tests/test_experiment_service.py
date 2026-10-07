from pathlib import Path

from app.pipeline.config import PipelineConfig
from app.pipeline.experiment import ExperimentConfig
from app.pipeline.experiment_service import (
    ExperimentService,
)


IMAGE_PATHS = [
    "dataset/input/test001.jpg",
]


def create_experiments():

    # ============================================================
    # EXPERIMENT 1
    # ============================================================

    config_1 = PipelineConfig()

    config_1.enhancement.method = "clahe"
    config_1.enhancement.clip_limit = 1.0
    config_1.segmentation.method = "otsu"

    experiment_1 = ExperimentConfig(
        name="service_clahe_1_0_otsu",
        config=config_1,
    )

    # ============================================================
    # EXPERIMENT 2
    # ============================================================

    config_2 = PipelineConfig()

    config_2.enhancement.method = "clahe"
    config_2.enhancement.clip_limit = 2.0
    config_2.segmentation.method = "otsu"

    experiment_2 = ExperimentConfig(
        name="service_clahe_2_0_otsu",
        config=config_2,
    )

    return [
        experiment_1,
        experiment_2,
    ]


def main():

    print("=" * 70)
    print("EXPERIMENT SERVICE TEST")
    print("=" * 70)

    # ============================================================
    # SERVICE
    # ============================================================

    service = ExperimentService(
        output_dir="output/service_test"
    )

    experiments = create_experiments()

    # ============================================================
    # RUN
    # ============================================================

    print()
    print("Menjalankan experiment...")

    results = service.run(
        image_paths=IMAGE_PATHS,
        experiment_configs=experiments,
    )

    print("Experiment selesai.")

    # ============================================================
    # CHECK RESULTS
    # ============================================================

    assert len(results) == 2

    print()
    print("Experiment yang berhasil:")

    for result in results:

        print(
            f"  - {result.name}"
        )

    # ============================================================
    # LIST EXPERIMENTS
    # ============================================================

    print()
    print("Daftar experiment tersimpan:")

    experiments_saved = (
        service.list_experiments()
    )

    for experiment in experiments_saved:

        print(
            f"  - {experiment['name']}"
        )

    assert len(experiments_saved) == 2

    # ============================================================
    # LIST IMAGES
    # ============================================================

    experiment_name = (
        "service_clahe_1_0_otsu"
    )

    print()
    print(
        f"Image pada "
        f"{experiment_name}:"
    )

    images = service.list_images(
        experiment_name
    )

    for image in images:

        print(
            f"  - {image['name']}"
        )

    assert len(images) == 1

    # ============================================================
    # CHECK REPORT
    # ============================================================

    report_path = (
        Path("output")
        / "service_test"
        / experiment_name
        / "report.json"
    )

    assert report_path.exists()

    print()
    print(
        f"Report ditemukan:"
    )

    print(
        f"  {report_path}"
    )

    # ============================================================
    # DELETE IMAGE
    # ============================================================

    image_name = images[0]["name"]

    print()
    print(
        f"Menghapus image: "
        f"{image_name}"
    )

    service.delete_image(
        experiment_name=experiment_name,
        image_name=image_name,
    )

    image_path = service.get_image_path(
        experiment_name,
        image_name,
    )

    assert not image_path.exists()

    print(
        "Delete image berhasil."
    )

    # ============================================================
    # DELETE EXPERIMENT
    # ============================================================

    experiment_2_name = (
        "service_clahe_2_0_otsu"
    )

    print()
    print(
        f"Menghapus experiment: "
        f"{experiment_2_name}"
    )

    service.delete_experiment(
        experiment_2_name
    )

    experiment_path = (
        service.get_experiment_path(
            experiment_2_name
        )
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
    print("EXPERIMENT SERVICE TEST BERHASIL")
    print("=" * 70)


if __name__ == "__main__":
    main()