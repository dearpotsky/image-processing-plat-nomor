from pathlib import Path
import csv
import shutil

import cv2

from app.pipeline.config_utils import save_config_json


class OutputManager:
    """
    Mengatur penyimpanan dan pengelolaan hasil pipeline.

    Struktur:

    output/
    └── experiment_name/
        └── image_01/
            ├── config.json
            ├── 00_detection/
            ├── 01_preprocessing/
            ├── 02_enhancement/
            ├── 03_segmentation/
            └── 04_feature_analysis/
                └── feature_analysis.csv
    """

    def __init__(self, base_dir="output"):
        self.base_dir = Path(base_dir)

    # ============================================================
    # SAVE
    # ============================================================

    def save_pipeline_result(
        self,
        result,
        image_name,
        config=None,
    ):
        image_stem = Path(image_name).stem
        output_dir = self.base_dir / image_stem

        directories = {
            "detection": output_dir / "00_detection",
            "preprocessing": output_dir / "01_preprocessing",
            "enhancement": output_dir / "02_enhancement",
            "segmentation": output_dir / "03_segmentation",
            "feature": output_dir / "04_feature_analysis",
        }

        for directory in directories.values():
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        if config is not None:
            save_config_json(
                config,
                output_dir / "config.json",
            )

        for plate in result.plates:
            index = plate.index

            self._save_image(
                plate.crop,
                directories["detection"]
                / f"plate_{index:02d}.jpg",
            )

            self._save_image(
                plate.preprocessing,
                directories["preprocessing"]
                / f"plate_{index:02d}_preprocessed.jpg",
            )

            self._save_image(
                plate.enhancement,
                directories["enhancement"]
                / f"plate_{index:02d}_enhanced.jpg",
            )

            self._save_image(
                plate.segmentation,
                directories["segmentation"]
                / f"plate_{index:02d}_segmented.jpg",
            )

        self._save_features(
            result,
            directories["feature"]
            / "feature_analysis.csv",
        )

    # ============================================================
    # LIST RESULTS
    # ============================================================

    def list_results(self):
        """
        Mengembalikan daftar experiment/result.
        """

        if not self.base_dir.exists():
            return []

        results = []

        for path in sorted(self.base_dir.iterdir()):
            if not path.is_dir():
                continue

            results.append(
                {
                    "name": path.name,
                    "path": path,
                }
            )

        return results

    # ============================================================
    # LIST IMAGES
    # ============================================================

    def list_images(self, result_name):
        """
        Mengembalikan daftar image dalam sebuah experiment (termasuk tiap plat).
        """

        result_dir = self.base_dir / result_name

        if not result_dir.exists():
            return []

        images = []

        for path in sorted(result_dir.iterdir()):
            if not path.is_dir():
                continue

            detection_dir = path / "00_detection"
            found_plates = False
            
            if detection_dir.exists():
                plates = sorted(detection_dir.glob("*.jpg"))
                if plates:
                    found_plates = True
                    for plate_path in plates:
                        images.append(
                            {
                                "name": f"{path.name} - {plate_path.stem}",
                                "path": path,
                            }
                        )
                        
            if not found_plates:
                images.append(
                    {
                        "name": path.name,
                        "path": path,
                    }
                )

        return images

    # ============================================================
    # DELETE IMAGE
    # ============================================================

    def delete_image_result(
        self,
        result_name,
        image_name,
    ):
        """
        Menghapus seluruh hasil satu image.
        """

        if " - " in image_name:
            folder_name, _ = image_name.split(" - ", 1)
        else:
            folder_name = image_name

        image_dir = (
            self.base_dir
            / result_name
            / folder_name
        )

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Hasil image tidak ditemukan: {image_dir}"
            )

        if not image_dir.is_dir():
            raise ValueError(
                f"Path bukan directory: {image_dir}"
            )

        shutil.rmtree(image_dir)

    # ============================================================
    # DELETE RESULT / EXPERIMENT
    # ============================================================

    def delete_result(self, result_name=None):
        """
        Menghapus seluruh hasil experiment.

        Jika result_name diberikan:

            output/
            └── experiment_name/

        Jika result_name=None:

            langsung menghapus self.base_dir.
        """

        if result_name is None:
            result_dir = self.base_dir
        else:
            result_dir = self.base_dir / result_name

        if not result_dir.exists():
            raise FileNotFoundError(
                f"Hasil tidak ditemukan: {result_dir}"
            )

        if not result_dir.is_dir():
            raise ValueError(
                f"Path bukan directory: {result_dir}"
            )

        shutil.rmtree(result_dir)

    # ============================================================
    # SAVE IMAGE
    # ============================================================

    @staticmethod
    def _save_image(
        image,
        output_path,
    ):
        if image is None:
            return

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output_path),
            image,
        )

        if not success:
            raise IOError(
                f"Gagal menyimpan image: {output_path}"
            )

    # ============================================================
    # SAVE FEATURES
    # ============================================================

    @staticmethod
    def _save_features(
        result,
        output_path,
    ):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "plate",
            "confidence",
            "threshold",
            "width",
            "height",
            "foreground_ratio",
            "mean_intensity",
            "std_intensity",
            "total_components",
            "valid_components",
        ]

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for plate in result.plates:
                features = plate.features

                writer.writerow(
                    {
                        "plate": plate.index,
                        "confidence": plate.confidence,
                        "threshold": features.get(
                            "threshold"
                        ),
                        "width": features.get(
                            "width"
                        ),
                        "height": features.get(
                            "height"
                        ),
                        "foreground_ratio": features.get(
                            "foreground_ratio"
                        ),
                        "mean_intensity": features.get(
                            "mean_intensity"
                        ),
                        "std_intensity": features.get(
                            "std_intensity"
                        ),
                        "total_components": features.get(
                            "total_components"
                        ),
                        "valid_components": features.get(
                            "valid_components"
                        ),
                    }
                )