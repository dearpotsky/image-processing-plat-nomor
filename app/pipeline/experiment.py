from dataclasses import dataclass
from pathlib import Path

from app.pipeline.config import PipelineConfig
from app.pipeline.batch_pipeline import (
    BatchImageProcessingPipeline,
)
from app.pipeline.result import BatchResult


@dataclass
class ExperimentConfig:
    name: str
    config: PipelineConfig


@dataclass
class ExperimentResult:
    name: str
    config: PipelineConfig
    result: BatchResult


class ExperimentRunner:

    def __init__(
        self,
        output_dir: str = "output/experiments",
    ):
        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run(
        self,
        image_paths,
        experiment_configs,
    ):

        results = []

        for experiment_config in (
            experiment_configs
        ):

            config = experiment_config.config

            # Pastikan nama experiment juga
            # masuk ke configuration.
            config.experiment.name = (
                experiment_config.name
            )

            pipeline = (
                BatchImageProcessingPipeline(
                    config=config
                )
            )

            batch_result = pipeline.run(
                image_paths
            )

            results.append(
                ExperimentResult(
                    name=experiment_config.name,
                    config=config,
                    result=batch_result,
                )
            )

        return results