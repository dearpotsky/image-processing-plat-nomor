from app.pipeline.config import PipelineConfig


class ConfigValidationError(ValueError):
    """Error untuk configuration yang tidak valid."""


class ConfigValidator:

    VALID_INTERPOLATIONS = {
        "INTER_NEAREST",
        "INTER_LINEAR",
        "INTER_CUBIC",
        "INTER_AREA",
        "INTER_LANCZOS4",
    }

    VALID_ENHANCEMENT_METHODS = {
        "none",
        "clahe",
    }

    VALID_SEGMENTATION_METHODS = {
        "none",
        "otsu",
        "adaptive",
    }

    def validate(self, config: PipelineConfig):
        errors = []

        errors.extend(
            self._validate_detector(
                config
            )
        )

        errors.extend(
            self._validate_preprocessing(
                config
            )
        )

        errors.extend(
            self._validate_enhancement(
                config
            )
        )

        errors.extend(
            self._validate_segmentation(
                config
            )
        )

        errors.extend(
            self._validate_feature(
                config
            )
        )

        if errors:
            message = (
                "Configuration tidak valid:\n"
                + "\n".join(
                    f"- {error}"
                    for error in errors
                )
            )

            raise ConfigValidationError(
                message
            )

        return True

    # ======================================================
    # DETECTOR
    # ======================================================

    def _validate_detector(self, config):
        errors = []

        detector = config.detector

        if not detector.model_path:
            errors.append(
                "detector.model_path "
                "tidak boleh kosong."
            )

        if not (
            0.0
            <= detector.confidence
            <= 1.0
        ):
            errors.append(
                "detector.confidence "
                "harus berada pada 0.0 sampai 1.0."
            )

        return errors

    # ======================================================
    # PREPROCESSING
    # ======================================================

    def _validate_preprocessing(
        self,
        config,
    ):
        errors = []

        preprocessing = (
            config.preprocessing
        )

        if (
            preprocessing.target_width
            <= 0
        ):
            errors.append(
                "preprocessing.target_width "
                "harus > 0."
            )

        if (
            preprocessing.target_height
            <= 0
        ):
            errors.append(
                "preprocessing.target_height "
                "harus > 0."
            )

        if (
            preprocessing.interpolation
            not in self.VALID_INTERPOLATIONS
        ):
            errors.append(
                "preprocessing.interpolation "
                f"tidak dikenal: "
                f"{preprocessing.interpolation}"
            )

        return errors

    # ======================================================
    # ENHANCEMENT
    # ======================================================

    def _validate_enhancement(
        self,
        config,
    ):
        errors = []

        enhancement = (
            config.enhancement
        )

        if (
            enhancement.method
            not in self.VALID_ENHANCEMENT_METHODS
        ):
            errors.append(
                "enhancement.method "
                f"tidak dikenal: "
                f"{enhancement.method}"
            )

        if enhancement.clip_limit <= 0:
            errors.append(
                "enhancement.clip_limit "
                "harus > 0."
            )

        if (
            enhancement.tile_grid_width
            <= 0
        ):
            errors.append(
                "enhancement.tile_grid_width "
                "harus > 0."
            )

        if (
            enhancement.tile_grid_height
            <= 0
        ):
            errors.append(
                "enhancement.tile_grid_height "
                "harus > 0."
            )

        return errors

    # ======================================================
    # SEGMENTATION
    # ======================================================

    def _validate_segmentation(
        self,
        config,
    ):
        errors = []

        segmentation = (
            config.segmentation
        )

        if (
            segmentation.method
            not in self.VALID_SEGMENTATION_METHODS
        ):
            errors.append(
                "segmentation.method "
                f"tidak dikenal: "
                f"{segmentation.method}"
            )

        if segmentation.method == "adaptive":

            if (
                segmentation.block_size
                <= 1
            ):
                errors.append(
                    "segmentation.block_size "
                    "harus > 1."
                )

            elif (
                segmentation.block_size
                % 2
                == 0
            ):
                errors.append(
                    "segmentation.block_size "
                    "harus bernilai ganjil."
                )

        return errors

    # ======================================================
    # FEATURE
    # ======================================================

    def _validate_feature(
        self,
        config,
    ):
        errors = []

        feature = config.feature

        if feature.connectivity not in (
            4,
            8,
        ):
            errors.append(
                "feature.connectivity "
                "hanya boleh 4 atau 8."
            )

        if feature.min_area < 0:
            errors.append(
                "feature.min_area "
                "tidak boleh negatif."
            )

        return errors