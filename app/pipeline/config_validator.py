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
        "contrast_stretching",
        "histogram_equalization",
        "lpf",
        "hpf",
        "retinex",
        "clahe",
        "gamma_correction",
        "unsharp_masking",
        "morphology_open",
    }

    VALID_SEGMENTATION_METHODS = {
        "none",
        "global",
        "adaptive",
        "otsu",
        "sobel",
        "canny",
        "prewitt",
        "region_growing",
        "region_split_merge",
        "kmeans",
        "fuzzy_cmeans",
        "watershed",
        "active_contour",
        "clear_border",
    }

    def validate(self, config: PipelineConfig):
        errors = []

        errors.extend(
            self._validate_detector(
                config
            )
        )

        errors.extend(
            self._validate_ocr(
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

        if not (0.0 <= detector.confidence <= 1.0):
            errors.append("detector.confidence harus berada pada 0.0 sampai 1.0.")
            
        if not (0.0 <= detector.padding_v <= 1.0):
            errors.append("detector.padding_v harus berada pada 0.0 sampai 1.0.")
            
        if not (0.0 <= detector.padding_left <= 1.0):
            errors.append("detector.padding_left harus berada pada 0.0 sampai 1.0.")
            
        if not (0.0 <= detector.padding_right <= 1.0):
            errors.append("detector.padding_right harus berada pada 0.0 sampai 1.0.")

        return errors

    # ======================================================
    # OCR
    # ======================================================
    
    def _validate_ocr(self, config):
        errors = []
        ocr = config.ocr

        from app.detection.ocr import VALID_OCR_ENGINES
        if ocr.engine.lower() not in VALID_OCR_ENGINES:
            errors.append(
                f"ocr.engine tidak dikenal: '{ocr.engine}'. "
                f"Pilihan valid: {sorted(VALID_OCR_ENGINES)}"
            )

        if not (0.0 <= ocr.text_threshold <= 1.0):
            errors.append("ocr.text_threshold harus berada pada 0.0 sampai 1.0.")
            
        if not (0.0 <= ocr.low_text <= 1.0):
            errors.append("ocr.low_text harus berada pada 0.0 sampai 1.0.")
            
        if ocr.mag_ratio <= 0.0:
            errors.append("ocr.mag_ratio harus > 0.0.")
            
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

        if not enhancement.steps:
            pass # Kosong berarti tidak ada enhancement
        else:
            for i, step in enumerate(enhancement.steps):
                if step.method not in self.VALID_ENHANCEMENT_METHODS:
                    errors.append(
                        f"enhancement.steps[{i}] memiliki nilai tidak dikenal: {step.method}"
                    )

                if step.limit < 0:
                    errors.append(
                        f"enhancement.steps[{i}].limit "
                        "harus >= 0."
                    )

                if (
                    step.tile_grid_width
                    <= 0
                ):
                    errors.append(
                        f"enhancement.steps[{i}].tile_grid_width "
                        "harus > 0."
                    )

                if (
                    step.tile_grid_height
                    <= 0
                ):
                    errors.append(
                        f"enhancement.steps[{i}].tile_grid_height "
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

        if not segmentation.steps:
            pass # Kosong berarti tidak ada segmentation
        else:
            for i, step in enumerate(segmentation.steps):
                if step.method not in self.VALID_SEGMENTATION_METHODS:
                    errors.append(
                        f"segmentation.steps[{i}] memiliki nilai tidak dikenal: {step.method}"
                    )

                if step.limit < 0:
                    errors.append(
                        f"segmentation.steps[{i}].limit harus >= 0."
                    )

                if step.method == "adaptive":
                    if segmentation.block_size <= 1:
                        errors.append("segmentation.block_size harus > 1.")
                    elif segmentation.block_size % 2 == 0:
                        errors.append("segmentation.block_size harus bernilai ganjil.")

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