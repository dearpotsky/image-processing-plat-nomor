from pathlib import Path

import cv2

from app.pipeline.pipeline import (
    ImageProcessingPipeline,
)

from app.pipeline.result import (
    BatchResult,
)

from app.pipeline.config import (
    PipelineConfig,
)


class BatchImageProcessingPipeline:

    def __init__(
        self,
        config: PipelineConfig | None = None,
    ):

        self.config = (
            config
            or PipelineConfig()
        )

        self.pipeline = (
            ImageProcessingPipeline(
                config=self.config
            )
        )

    def run(
        self,
        image_paths,
    ):

        batch_result = BatchResult()

        for image_path in image_paths:

            image_path = Path(
                image_path
            )

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                raise ValueError(
                    f"Gagal membaca image: "
                    f"{image_path}"
                )

            result = self.pipeline.run(
                image
            )

            batch_result.add_image(
                result
            )

        return batch_result