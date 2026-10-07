from pathlib import Path

import cv2

from app.processing.preprocessing import Preprocessor


INPUT_DIR = Path("output/00_detection")
OUTPUT_DIR = Path("output/01_preprocessing")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    preprocessor = Preprocessor(
        target_size=(400, 120)
    )

    plate_files = sorted(
        INPUT_DIR.glob("plate_*.jpg")
    )

    if not plate_files:
        raise FileNotFoundError(
            "Tidak ditemukan hasil crop plate."
        )

    for plate_path in plate_files:
        image = cv2.imread(
            str(plate_path)
        )

        if image is None:
            raise ValueError(
                f"Gagal membaca: {plate_path}"
            )

        result = preprocessor.process(image)

        output_path = (
            OUTPUT_DIR
            / f"{plate_path.stem}_preprocessed.jpg"
        )

        success = cv2.imwrite(
            str(output_path),
            result
        )

        if not success:
            raise IOError(
                f"Gagal menyimpan: {output_path}"
            )

        print(
            f"✓ {plate_path.name}"
            f" → {output_path.name}"
        )

        print(
            f"  Input  : "
            f"{image.shape[1]} x {image.shape[0]}"
        )

        print(
            f"  Output : "
            f"{result.shape[1]} x {result.shape[0]}"
        )

        print(
            f"  Range  : "
            f"{result.min()} - {result.max()}"
        )

        print()

    print("✓ Preprocessing selesai")


if __name__ == "__main__":
    main()