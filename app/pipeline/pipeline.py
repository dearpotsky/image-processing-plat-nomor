from pathlib import Path
import logging

from app.detection.plate_detector import (
    PlateDetector,
)

from app.detection.plate_cropper import (
    PlateCropper,
)

from app.detection.ocr import (
    PlateReader,
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
            config=self.config.detector
        )

        # ------------------------------------------------
        # Crop & OCR
        # ------------------------------------------------

        self.cropper = PlateCropper()
        self.reader = PlateReader(
            config=self.config.ocr
        )

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
            resize=(
                getattr(preprocessing_config, "resize", True)
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
        logger = logging.getLogger('pipeline')
        logger.info("Memulai proses...")
        
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

        logger = logging.getLogger('pipeline')
        logger.info("-------------------------------------------------")
        detector_enabled = getattr(self.config.detector, "enabled", True)

        if not detector_enabled:
            logger.info("Tahap 1: Deteksi Plat di-BYPASS (Mode Direct Plate / Gambar Plat Langsung)")
            logger.info("-> Gambar input langsung diproses sebagai plat nomor utuh (tanpa pemotongan YOLO).")
            h, w = image.shape[:2]
            detections = [{
                "box": (0.0, 0.0, float(w), float(h)),
                "confidence": 1.0,
            }]
        else:
            logger.info("Tahap 1: Deteksi Area Plat Nomor menggunakan YOLO")
            detections = (
                self.detector.detect(image)
            )

            if not detections:
                logger.info("-> YOLO gagal mendeteksi adanya plat pada gambar.")
            else:
                logger.info(f"-> YOLO menemukan {len(detections)} plat.")

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

            if not detector_enabled:
                logger.info(f"Plat {index}: Menggunakan gambar penuh sebagai plat (tanpa crop/pecah)...")
                crop = image.copy()
            else:
                logger.info(f"Plat {index}: Melakukan pemotongan (cropping) gambar pada area deteksi...")
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
                logger.info(f"Plat {index}: Tahap 2: Preprocessing (Grayscale, Resize, Normalization)...")
                preprocessed = (
                    self.preprocessor.process(
                        crop
                    )
                )

            else:
                logger.info(f"Plat {index}: Tahap 2: Preprocessing diabaikan.")
                preprocessed = crop

            # ---------------------------------------------
            # Enhancement
            # ---------------------------------------------

            enhancement_config = (
                self.config.enhancement
            )

            if enhancement_config.enabled and enhancement_config.steps:
                logger.info(f"Plat {index}: Tahap 3: Enhancement dengan {len(enhancement_config.steps)} langkah berseri")
                for i, step in enumerate(enhancement_config.steps, 1):
                    logger.info(f"  -> [Enhancement {i}] Memproses {step.method} (limit={step.limit})")
                
                enhanced = (
                    self.enhancer.enhance(
                        preprocessed,
                        steps=(
                            enhancement_config.steps
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

            if segmentation_config.enabled and segmentation_config.steps:
                logger.info(f"Plat {index}: Tahap 4: Segmentasi (binarisasi) dengan {len(segmentation_config.steps)} langkah berseri")
                for i, step in enumerate(segmentation_config.steps, 1):
                    logger.info(f"  -> [Segmentation {i}] Memproses {step.method} (limit={step.limit})")

                threshold, segmented = (
                    self.segmenter.segment(
                        enhanced,
                        steps=(
                            segmentation_config
                            .steps
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
            # OCR
            # ---------------------------------------------
            
            ocr_engine_name = getattr(self.config.ocr, 'engine', 'easyocr').upper()
            logger.info(f"Plat {index}: Tahap 5: Ekstraksi Teks (OCR) menggunakan {ocr_engine_name}...")
            # Coba membaca gambar biner hasil segmentasi terlebih dahulu
            plate_text, text_prob = self.reader.read_text(segmented)
            if not plate_text and enhanced is not None:
                # Fallback: jika segmented kosong (sering terjadi pada PaddleOCR terhadap citra biner keras),
                # baca gambar enhanced yang memiliki informasi kontur teks lebih kaya.
                logger.info(f"Plat {index}: -> Hasil OCR pada biner kosong, mencoba membaca citra enhanced...")
                fb_text, fb_prob = self.reader.read_text(enhanced)
                if fb_text:
                    plate_text, text_prob = fb_text, fb_prob

            if plate_text:
                logger.info(f"Plat {index}: -> Teks plat terbaca: '{plate_text}' (Kepercayaan: {text_prob:.2f})")
            else:
                logger.info(f"Plat {index}: -> Gagal membaca teks plat (OCR kosong).")

            # ---------------------------------------------
            # Feature Analysis
            # ---------------------------------------------

            feature_config = (
                self.config.feature
            )

            if feature_config.enabled and segmented is not None:
                logger.info(f"Plat {index}: Tahap 6: Analisis Fitur Gambar Biner (menghitung luas objek, dll)...")
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
                text=plate_text,
                text_confidence=text_prob,
                features=features,
            )

            result.add_plate(
                plate_result
            )

        # =================================================
        # WATERMARKING
        # =================================================

        watermark_config = self.config.watermark
        if watermark_config.enabled and detections:
            logger.info("Tahap 7: Memberikan Watermarking (Kotak & Teks OCR) pada citra awal...")
            import cv2
            watermarked = image.copy()
            for index, detection in enumerate(detections, start=1):
                x1, y1, x2, y2 = [int(v) for v in detection["box"]]
                confidence = detection["confidence"]
                
                # Draw green border
                cv2.rectangle(
                    watermarked, 
                    (x1, y1), 
                    (x2, y2), 
                    (0, 255, 0), 
                    2
                )
                
                # Use OCR Text if available
                plate_res = result.plates[index-1]
                if plate_res.text:
                    text = f"{plate_res.text} ({plate_res.text_confidence:.2f})"
                else:
                    text = f"Plat {index} ({confidence:.2f})"
                
                # Text background
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    watermarked, 
                    (x1, y1 - 25), 
                    (x1 + tw, y1), 
                    (0, 255, 0), 
                    -1
                )
                
                # Text
                cv2.putText(
                    watermarked, 
                    text, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 0, 0), 
                    2
                )
                
            result.watermarked = watermarked

        return result