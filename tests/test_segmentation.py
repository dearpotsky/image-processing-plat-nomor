from pathlib import Path

import cv2

from app.processing.segmentation import ImageSegmenter


INPUT_DIR = Path(
    "output/02_enhancement/clip_2"
)

OUTPUT_DIR = Path(
    "output/03_segmentation"
)


def save_image(image, path):
    success = cv2.imwrite(
        str(path),
        image
    )

    if not success:
        raise IOError(
            f"Gagal menyimpan: {path}"
        )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    input_files = sorted(
        INPUT_DIR.glob("*_enhanced.jpg")
    )

    if not input_files:
        raise FileNotFoundError(
            "Tidak ditemukan hasil enhancement clip_2."
        )

    segmenter = ImageSegmenter()

    # Folder hasil eksperimen
    otsu_dir = OUTPUT_DIR / "otsu"
    adaptive_dir = OUTPUT_DIR / "adaptive"

    otsu_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    adaptive_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 50)
    print("SEGMENTATION EXPERIMENT")
    print("=" * 50)

    for input_path in input_files:

        image = cv2.imread(
            str(input_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"Gagal membaca: {input_path}"
            )

        # =========================
        # 1. OTSU
        # =========================

        threshold, otsu_result = (
            segmenter.otsu(image)
        )

        otsu_path = (
            otsu_dir
            / input_path.name.replace(
                "_enhanced",
                "_otsu"
            )
        )

        save_image(
            otsu_result,
            otsu_path
        )

        # =========================
        # 2. ADAPTIVE
        # =========================

        adaptive_result = (
            segmenter.adaptive(
                image,
                block_size=11,
                c=2
            )
        )

        adaptive_path = (
            adaptive_dir
            / input_path.name.replace(
                "_enhanced",
                "_adaptive"
            )
        )

        save_image(
            adaptive_result,
            adaptive_path
        )

        print()
        print(f"Input: {input_path.name}")

        print(
            f"  Otsu threshold : "
            f"{threshold:.2f}"
        )

        print(
            f"  Otsu output    : "
            f"{otsu_path}"
        )

        print(
            f"  Adaptive       : "
            f"blockSize=11, C=2"
        )

        print(
            f"  Adaptive output: "
            f"{adaptive_path}"
        )

    print()
    print("✓ Eksperimen segmentation selesai")


if __name__ == "__main__":
    main()