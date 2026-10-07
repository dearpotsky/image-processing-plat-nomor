import cv2

from app.pipeline.config import PipelineConfig
from app.pipeline.pipeline import ImageProcessingPipeline


IMAGE_PATH = "dataset/input/test001.jpg"


def main():
    print("=" * 60)
    print("IMAGE PROCESSING PIPELINE TEST")
    print("=" * 60)

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Gagal membaca image: {IMAGE_PATH}"
        )

    print(f"\nInput image : {IMAGE_PATH}")
    print(f"Image shape : {image.shape}")

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
        "Grayscale      :",
        config.preprocessing.grayscale
    )

    print(
        "Normalize      :",
        config.preprocessing.normalize
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
        "Tile grid      :",
        config.enhancement.tile_grid_size
    )

    print(
        "Segmentation   :",
        config.segmentation.method
    )

    print(
        "Feature        :",
        config.feature.enabled
    )

    print(
        "Min area       :",
        config.feature.min_area
    )

    # ========================================================
    # CREATE PIPELINE
    # ========================================================

    pipeline = ImageProcessingPipeline(
        config=config
    )

    # ========================================================
    # RUN PIPELINE
    # ========================================================

    print("\nMenjalankan pipeline...")

    result = pipeline.run(image)

    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("PIPELINE RESULT")
    print("=" * 60)

    print(
        f"\nJumlah plate terdeteksi: "
        f"{result.plate_count}"
    )

    if result.plate_count == 0:
        print("\nTidak ada plate yang terdeteksi.")
        return

    # ========================================================
    # PLATE DETAILS
    # ========================================================

    for plate in result.plates:

        print("\n" + "-" * 60)
        print(f"Plate #{plate.index}")
        print("-" * 60)

        print(
            f"Confidence       : "
            f"{plate.confidence:.3f}"
        )

        print(
            f"Bounding box     : "
            f"{plate.box}"
        )

        # ----------------------------------------------------
        # Image shapes
        # ----------------------------------------------------

        if plate.crop is not None:
            print(
                f"Crop shape      : "
                f"{plate.crop.shape}"
            )

        if plate.preprocessing is not None:
            print(
                f"Preprocessing   : "
                f"{plate.preprocessing.shape}"
            )

        if plate.enhancement is not None:
            print(
                f"Enhancement     : "
                f"{plate.enhancement.shape}"
            )

        if plate.segmentation is not None:
            print(
                f"Segmentation    : "
                f"{plate.segmentation.shape}"
            )

        # ----------------------------------------------------
        # Feature results
        # ----------------------------------------------------

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
            f"Threshold        : "
            f"{threshold_text}"
        )

        if features:

            print(
                f"Foreground ratio : "
                f"{features.get('foreground_ratio', 'N/A')}"
            )

            print(
                f"Mean intensity   : "
                f"{features.get('mean_intensity', 'N/A')}"
            )

            print(
                f"Std intensity    : "
                f"{features.get('std_intensity', 'N/A')}"
            )

            print(
                f"Total components : "
                f"{features.get('total_components', 'N/A')}"
            )

            print(
                f"Valid components : "
                f"{features.get('valid_components', 'N/A')}"
            )

    print("\n" + "=" * 60)
    print("PIPELINE TEST SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()