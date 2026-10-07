from pathlib import Path

from app.pipeline.experiment import ExperimentRunner
from app.pipeline.output_manager import OutputManager
from app.pipeline.report import ExperimentReporter
from app.pipeline.comparison import ExperimentComparator


class ExperimentService:
    """
    Service utama untuk menjalankan dan mengelola experiment.

    GUI nantinya cukup berinteraksi dengan class ini.

    Flow:

        input images
             ↓
        ExperimentRunner
             ↓
        OutputManager
             ↓
        ExperimentReporter
             ↓
        ExperimentComparator
    """

    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)

        self.runner = ExperimentRunner(
            output_dir=str(self.output_dir)
        )

        self.reporter = ExperimentReporter(
            output_dir=str(self.output_dir)
        )

    # ============================================================
    # RUN EXPERIMENT
    # ============================================================

    def run(
        self,
        image_paths,
        experiment_configs,
    ):
        """
        Menjalankan satu atau beberapa experiment.

        Args:
            image_paths:
                list path image

            experiment_configs:
                list ExperimentConfig

        Returns:
            list[ExperimentResult]
        """

        results = self.runner.run(
            image_paths=image_paths,
            experiment_configs=experiment_configs,
        )

        self._save_results(
            results
        )

        self.reporter.generate(
            results
        )

        return results

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    def _save_results(
        self,
        experiment_results,
    ):
        for experiment in experiment_results:

            experiment_dir = (
                self.output_dir
                / experiment.name
            )

            output_manager = OutputManager(
                base_dir=experiment_dir
            )

            for image_index, image_result in enumerate(
                experiment.result.images,
                start=1,
            ):

                image_name = (
                    f"image_{image_index:02d}"
                )

                output_manager.save_pipeline_result(
                    result=image_result,
                    image_name=image_name,
                    config=experiment.config,
                )

    # ============================================================
    # LIST EXPERIMENTS
    # ============================================================

    def list_experiments(self):
        """
        Mengambil semua experiment yang tersimpan.
        """

        output_manager = OutputManager(
            base_dir=self.output_dir
        )

        return output_manager.list_results()

    # ============================================================
    # LIST IMAGES
    # ============================================================

    def list_images(
        self,
        experiment_name,
    ):
        """
        Mengambil semua image dari experiment.
        """

        experiment_dir = (
            self.output_dir
            / experiment_name
        )

        output_manager = OutputManager(
            base_dir=experiment_dir
        )

        return output_manager.list_images(
            result_name="."
        )

    # ============================================================
    # DELETE IMAGE
    # ============================================================

    def delete_image(
        self,
        experiment_name,
        image_name,
    ):
        """
        Menghapus hasil satu image.
        """

        experiment_dir = (
            self.output_dir
            / experiment_name
        )

        output_manager = OutputManager(
            base_dir=experiment_dir
        )

        output_manager.delete_image_result(
            result_name=".",
            image_name=image_name,
        )

    # ============================================================
    # DELETE EXPERIMENT
    # ============================================================

    def delete_experiment(
        self,
        experiment_name,
    ):
        """
        Menghapus seluruh experiment.
        """

        output_manager = OutputManager(
            base_dir=self.output_dir
        )

        output_manager.delete_result(
            result_name=experiment_name
        )

    # ============================================================
    # GET EXPERIMENT PATH
    # ============================================================

    def get_experiment_path(
        self,
        experiment_name,
    ):
        return (
            self.output_dir
            / experiment_name
        )

    # ============================================================
    # GET IMAGE PATH
    # ============================================================

    def get_image_path(
        self,
        experiment_name,
        image_name,
    ):
        return (
            self.output_dir
            / experiment_name
            / image_name
        )

    # ============================================================
    # COMPARE
    # ============================================================

    @staticmethod
    def compare(
        experiment_results,
    ):
        """
        Membandingkan beberapa hasil experiment.
        """

        comparator = ExperimentComparator()

        return comparator.compare(
            experiment_results
        )