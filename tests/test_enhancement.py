from pathlib import Path

import cv2

from app.processing.enhancement import CLAHEEnhancer


INPUT_DIR = Path("output/01_preprocessing")
OUTPUT_DIR = Path("output/02_enhancement")


# Parameter eksperimen
CLAHE_CONFIGS = [
    {
        "name": "clip_1",
        "clip_limit": 1.0,
        "tile_grid_size": (8, 8),
    },
    {
        "name": "clip_2",
        "clip_limit": 2.0,
        "tile_grid_size": (8, 8),
    },
    {
        "name": "clip_4",
        "clip_limit": 4.0,
        "tile_grid_size": (8, 8),
    },
]


def main():
    input_files = sorted(
        INPUT_DIR.glob("*_preprocessed.jpg")
    )

    if not input_files:
        raise FileNotFoundError(
            "Tidak ditemukan hasil preprocessing."
        )

    for config in CLAHE_CONFIGS:

        config_name = config["name"]

        output_dir = (
            OUTPUT_DIR / config_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        enhancer = CLAHEEnhancer(
            clip_limit=config["clip_limit"],
            tile_grid_size=config["tile_grid_size"],
        )

        print()
        print("=" * 50)
        print(f"Eksperimen: {config_name}")
        print(
            f"clipLimit = "
            f"{config['clip_limit']}"
        )
        print(
            f"tileGridSize = "
            f"{config['tile_grid_size']}"
        )
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

            enhanced = enhancer.enhance(
                image
            )

            output_path = (
                output_dir
                / input_path.name.replace(
                    "_preprocessed",
                    "_enhanced"
                )
            )

            success = cv2.imwrite(
                str(output_path),
                enhanced
            )

            if not success:
                raise IOError(
                    f"Gagal menyimpan: "
                    f"{output_path}"
                )

            print(
                f"✓ {input_path.name}"
                f" → {output_path.name}"
            )

    print()
    print("✓ Semua eksperimen CLAHE selesai")


if __name__ == "__main__":
    main()