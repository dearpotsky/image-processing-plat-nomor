from pathlib import Path
import csv

import cv2

from app.processing.feature_analysis import (
    FeatureAnalyzer,
)


INPUT_DIR = Path(
    "output/03_segmentation/otsu"
)

OUTPUT_DIR = Path(
    "output/04_feature_analysis"
)

CSV_PATH = (
    OUTPUT_DIR / "feature_analysis.csv"
)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    input_files = sorted(
        INPUT_DIR.glob("*_otsu.jpg")
    )

    if not input_files:
        raise FileNotFoundError(
            "Tidak ditemukan hasil Otsu."
        )

    analyzer = FeatureAnalyzer()

    results = []

    for input_path in input_files:

        image = cv2.imread(
            str(input_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"Gagal membaca: {input_path}"
            )

        features = analyzer.analyze(
            image
        )

        result = {
            "image": input_path.name,
            "width": features["width"],
            "height": features["height"],
            "foreground_ratio": (
                features["foreground_ratio"]
            ),
            "mean_intensity": (
                features["mean_intensity"]
            ),
            "std_intensity": (
                features["std_intensity"]
            ),
            "total_components": (
                features["total_components"]
            ),
            "valid_components": (
                features["valid_components"]
            ),
        }

        results.append(result)

        print()
        print("=" * 50)
        print(input_path.name)
        print("=" * 50)

        print(
            f"Ukuran             : "
            f"{features['width']} x "
            f"{features['height']}"
        )

        print(
            f"Foreground ratio   : "
            f"{features['foreground_ratio']:.4f}"
        )

        print(
            f"Mean intensity     : "
            f"{features['mean_intensity']:.2f}"
        )

        print(
            f"Std intensity      : "
            f"{features['std_intensity']:.2f}"
        )

        print(
            f"Total components   : "
            f"{features['total_components']}"
        )

        print(
            f"Valid components   : "
            f"{features['valid_components']}"
        )

    # ==========================================
    # SIMPAN CSV
    # ==========================================

    fieldnames = [
        "image",
        "width",
        "height",
        "foreground_ratio",
        "mean_intensity",
        "std_intensity",
        "total_components",
        "valid_components",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print(
        f"✓ Hasil analisis disimpan: "
        f"{CSV_PATH}"
    )


if __name__ == "__main__":
    main()