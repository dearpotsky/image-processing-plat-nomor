from app.pipeline.config import PipelineConfig
from app.pipeline.batch_pipeline import (
    BatchImageProcessingPipeline,
)


IMAGE_PATHS = [
    "dataset/input/test001.jpg",
    "dataset/input/test001.jpg",
    "dataset/input/test001.jpg",
]


def main():
    print("=" * 60)
    print("BATCH PIPELINE TEST")
    print("=" * 60)

    # ========================================================
    # CONFIGURATION
    # ========================================================

    config = PipelineConfig()

    print("\nConfiguration")
    print("-" * 60)

    print(
        "Detector model :",
        config.detector.model_path
    )

    print(
        "Confidence     :",
        config.detector.confidence
    )

    print(
        "Resize         :",
        config.preprocessing.target_size
    )

    print(
        "Enhancement    :",
        config.enhancement.method
    )

    print(
        "CLAHE clip     :",
        config.enhancement.clip_limit
    )

    print(
        "Segmentation   :",
        config.segmentation.method
    )

    print(
        "Min area       :",
        config.feature.min_area
    )

    # ========================================================
    # INPUT
    # ========================================================

    print("\nInput images")
    print("-" * 60)

    for index, image_path in enumerate(
        IMAGE_PATHS,
        start=1,
    ):
        print(
            f"{index}. {image_path}"
        )

    # ========================================================
    # CREATE BATCH PIPELINE
    # ========================================================

    pipeline = BatchImageProcessingPipeline(
        config=config
    )

    # ========================================================
    # RUN
    # ========================================================

    print("\nMenjalankan batch pipeline...")

    batch_result = pipeline.run(
        IMAGE_PATHS
    )

    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("BATCH RESULT")
    print("=" * 60)

    print(
        f"\nTotal image : "
        f"{batch_result.image_count}"
    )

    print(
        f"Total plate : "
        f"{batch_result.total_plate_count}"
    )

    # ========================================================
    # IMAGE DETAILS
    # ========================================================

    for image_index, result in enumerate(
        batch_result.images,
        start=1,
    ):

        print("\n" + "-" * 60)
        print(
            f"Image #{image_index}"
        )
        print("-" * 60)

        print(
            f"Jumlah plate: "
            f"{result.plate_count}"
        )

        for plate in result.plates:

            print(
                f"  Plate #{plate.index} | "
                f"Confidence: "
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
                f"    Threshold: "
                f"{threshold_text}"
            )

            if "foreground_ratio" in features:

                print(
                    f"    Foreground ratio: "
                    f"{features['foreground_ratio']:.4f}"
                )

            if "total_components" in features:

                print(
                    f"    Total components: "
                    f"{features['total_components']}"
                )

            if "valid_components" in features:

                print(
                    f"    Valid components: "
                    f"{features['valid_components']}"
                )

    print("\n" + "=" * 60)
    print("BATCH PIPELINE TEST SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()