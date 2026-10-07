import csv
import json
from pathlib import Path


class ExperimentReporter:
    """
    Membuat report dari hasil eksperimen.

    Output:

    output/
    └── experiment_name/
        ├── report.json
        └── experiment_comparison.csv
    """

    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)

    # ============================================================
    # GENERATE
    # ============================================================

    def generate(self, experiment_results):
        reports = []

        for experiment in experiment_results:

            report = self._build_experiment_report(
                experiment
            )

            experiment_dir = (
                self.output_dir
                / experiment.name
            )

            experiment_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._save_json(
                report,
                experiment_dir / "report.json",
            )

            reports.append(report)

        self._save_comparison_csv(
            reports
        )

        return reports

    # ============================================================
    # BUILD REPORT
    # ============================================================

    def _build_experiment_report(
        self,
        experiment,
    ):
        batch_result = experiment.result

        images = []

        for image_index, image_result in enumerate(
            batch_result.images,
            start=1,
        ):

            image_data = {
                "image_name": f"image_{image_index:02d}",
                "plate_count": len(
                    image_result.plates
                ),
                "plates": [],
            }

            for plate in image_result.plates:

                features = plate.features

                plate_data = {
                    "plate": plate.index,
                    "confidence": plate.confidence,
                    "features": {
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
                    },
                }

                image_data["plates"].append(
                    plate_data
                )

            images.append(
                image_data
            )

        return {
            "experiment": experiment.name,
            "image_count": batch_result.image_count,
            "plate_count": batch_result.total_plate_count,
            "images": images,
        }

    # ============================================================
    # SAVE JSON
    # ============================================================

    @staticmethod
    def _save_json(
        data,
        output_path,
    ):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )

    # ============================================================
    # SAVE COMPARISON CSV
    # ============================================================

    def _save_comparison_csv(
        self,
        reports,
    ):
        if not reports:
            return

        output_path = (
            self.output_dir
            / "experiment_comparison.csv"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "experiment",
            "image_count",
            "plate_count",
            "avg_foreground_ratio",
            "avg_mean_intensity",
            "avg_std_intensity",
            "avg_total_components",
            "avg_valid_components",
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

            for report in reports:

                metrics = self._calculate_metrics(
                    report
                )

                writer.writerow(
                    {
                        "experiment": report[
                            "experiment"
                        ],
                        "image_count": report[
                            "image_count"
                        ],
                        "plate_count": report[
                            "plate_count"
                        ],
                        **metrics,
                    }
                )

    # ============================================================
    # CALCULATE METRICS
    # ============================================================

    @staticmethod
    def _calculate_metrics(
        report,
    ):
        foreground_values = []
        mean_values = []
        std_values = []
        total_components = []
        valid_components = []

        for image in report["images"]:

            for plate in image["plates"]:

                features = plate["features"]

                value = features.get(
                    "foreground_ratio"
                )

                if value is not None:
                    foreground_values.append(
                        value
                    )

                value = features.get(
                    "mean_intensity"
                )

                if value is not None:
                    mean_values.append(
                        value
                    )

                value = features.get(
                    "std_intensity"
                )

                if value is not None:
                    std_values.append(
                        value
                    )

                value = features.get(
                    "total_components"
                )

                if value is not None:
                    total_components.append(
                        value
                    )

                value = features.get(
                    "valid_components"
                )

                if value is not None:
                    valid_components.append(
                        value
                    )

        return {
            "avg_foreground_ratio":
                ExperimentReporter._average(
                    foreground_values
                ),

            "avg_mean_intensity":
                ExperimentReporter._average(
                    mean_values
                ),

            "avg_std_intensity":
                ExperimentReporter._average(
                    std_values
                ),

            "avg_total_components":
                ExperimentReporter._average(
                    total_components
                ),

            "avg_valid_components":
                ExperimentReporter._average(
                    valid_components
                ),
        }

    # ============================================================
    # AVERAGE
    # ============================================================

    @staticmethod
    def _average(values):

        if not values:
            return 0.0

        return sum(values) / len(values)