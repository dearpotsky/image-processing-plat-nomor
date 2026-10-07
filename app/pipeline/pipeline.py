from pathlib import Path

from app.detection.plate_detector import (
    PlateDetector,
)

from app.detection.plate_cropper import (
    PlateCropper,
)

from app.processing.preprocessing import (
    Preprocessor,
)

from app.processing.enhancement import (
    ImageEnhancer,
)

from app.processing.segmentation import (
    ImageSegmenter,
)

from app.processing.feature_analysis import (
    FeatureAnalyzer,
)

from app.pipeline.result import (
    PipelineResult,
    PlateResult,
)

from app.pipeline.config import (
    PipelineConfig,
)

from app.pipeline.config_validator import (
    ConfigValidator,
)


class ImageProcessingPipeline:

    def __init__(
        self,
        config: PipelineConfig | None = None,
    ):

        self.config = (
            config
            or PipelineConfig()
        )

        self.validator = ConfigValidator()

        self.validator.validate(
            self.config
        )

        self.output_dir = Path(
            self.config.output.base_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ------------------------------------------------
        # Detector
        # ------------------------------------------------

        detector_config = (
            self.config.detector
        )

        self.detector = PlateDetector(
            model_path=(
                detector_config.model_path
            ),
            confidence=(
                detector_config.confidence
            ),
        )

        # ------------------------------------------------
        # Crop
        # ------------------------------------------------

        self.cropper = PlateCropper()

        # ------------------------------------------------
        # Preprocessing
        # ------------------------------------------------

        preprocessing_config = (
            self.config.preprocessing
        )

        self.preprocessor = Preprocessor(
            target_size=(
                preprocessing_config.target_size
            ),
            grayscale=(
                preprocessing_config.grayscale
            ),
            normalize=(
                preprocessing_config.normalize
            ),
            interpolation=(
                preprocessing_config.interpolation
            ),
        )

        # ------------------------------------------------
        # Enhancement
        # ------------------------------------------------

        self.enhancer = ImageEnhancer()

        # ------------------------------------------------
        # Segmentation
        # ------------------------------------------------

        self.segmenter = ImageSegmenter()

        # ------------------------------------------------
        # Feature
        # ------------------------------------------------

        feature_config = (
            self.config.feature
        )

        self.feature_analyzer = (
            FeatureAnalyzer(
                min_area=(
                    feature_config.min_area
                ),
                connectivity=(
                    feature_config.connectivity
                ),
            )
        )

    def run(self, image):

        if image is None:
            raise ValueError(
                "Image tidak boleh None."
            )

        result = PipelineResult(
            image=image
        )

        # =================================================
        # DETECTION
        # =================================================

        detections = (
            self.detector.detect(image)
        )

        # =================================================
        # PROCESS EACH PLATE
        # =================================================

        for index, detection in enumerate(
            detections,
            start=1,
        ):

            box = detection["box"]
            confidence = (
                detection["confidence"]
            )

            # ---------------------------------------------
            # Crop
            # ---------------------------------------------

            crop = self.cropper.crop(
                image,
                box,
            )

            # ---------------------------------------------
            # Preprocessing
            # ---------------------------------------------

            preprocessing_config = (
                self.config.preprocessing
            )

            if preprocessing_config.enabled:

                preprocessed = (
                    self.preprocessor.process(
                        crop
                    )
                )

            else:

                preprocessed = crop

            # ---------------------------------------------
            # Enhancement
            # ---------------------------------------------

            enhancement_config = (
                self.config.enhancement
            )

            if enhancement_config.enabled:

                enhanced = (
                    self.enhancer.enhance(
                        preprocessed,
                        method=(
                            enhancement_config.method
                        ),
                        clip_limit=(
                            enhancement_config.clip_limit
                        ),
                        tile_grid_size=(
                            enhancement_config
                            .tile_grid_size
                        ),
                    )
                )

            else:

                enhanced = preprocessed

            # ---------------------------------------------
            # Segmentation
            # ---------------------------------------------

            segmentation_config = (
                self.config.segmentation
            )

            if segmentation_config.enabled:

                threshold, segmented = (
                    self.segmenter.segment(
                        enhanced,
                        method=(
                            segmentation_config
                            .method
                        ),
                        block_size=(
                            segmentation_config
                            .block_size
                        ),
                        c=(
                            segmentation_config.c
                        ),
                        invert=(
                            segmentation_config.invert
                        ),
                    )
                )

            else:

                threshold = None
                segmented = enhanced

            # ---------------------------------------------
            # Feature Analysis
            # ---------------------------------------------

            feature_config = (
                self.config.feature
            )

            if feature_config.enabled:

                features = (
                    self.feature_analyzer.analyze(
                        segmented
                    )
                )

            else:

                features = {}

            # ---------------------------------------------
            # Store result
            # ---------------------------------------------

            if threshold is not None:

                features = {
                    "threshold":
                        threshold,
                    **features,
                }

            plate_result = PlateResult(
                index=index,
                box=box,
                confidence=confidence,
                crop=crop,
                preprocessing=preprocessed,
                enhancement=enhanced,
                segmentation=segmented,
                features=features,
            )

            result.add_plate(
                plate_result
            )

        return result