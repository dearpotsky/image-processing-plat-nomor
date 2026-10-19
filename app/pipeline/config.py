from dataclasses import dataclass, field


# ============================================================
# DETECTOR
# ============================================================

@dataclass
class DetectorConfig:
    enabled: bool = True  # True: Deteksi YOLO + Crop. False: Direct Plate (gambar langsung dianggap plat utuh tanpa pemotongan)

    model_path: str = (
        "models/plate_detector/"
        "plate_detector_yolo11n.pt"
    )

    confidence: float = 0.25
    padding_v: float = 0.12
    padding_left: float = 0.25
    padding_right: float = 0.20

# ============================================================
# OCR
# ============================================================

@dataclass
class OcrConfig:
    # Pilihan engine: "easyocr" atau "paddleocr"
    engine: str = "easyocr"

    text_threshold: float = 0.6
    low_text: float = 0.3
    mag_ratio: float = 1.2

# ============================================================
# PREPROCESSING
# ============================================================

@dataclass
class PreprocessingConfig:
    enabled: bool = True

    resize: bool = True  # Jika False, mempertahankan resolusi asli agar tidak pecah/blur
    target_width: int = 400
    target_height: int = 120

    interpolation: str = "INTER_CUBIC"

    grayscale: bool = True
    normalize: bool = True

    @property
    def target_size(self):
        return (
            self.target_width,
            self.target_height,
        )


# ============================================================
# ENHANCEMENT
# ============================================================

@dataclass
class EnhancementStep:
    method: str
    limit: float = 0.0
    tile_grid_width: int = 8
    tile_grid_height: int = 8

    @property
    def tile_grid_size(self):
        return (
            self.tile_grid_width,
            self.tile_grid_height,
        )

@dataclass
class EnhancementConfig:
    enabled: bool = True

    steps: list[EnhancementStep] = field(
        default_factory=list
    )


# ============================================================
# SEGMENTATION
# ============================================================

@dataclass
class SegmentationStep:
    method: str
    limit: float = 0.0

@dataclass
class SegmentationConfig:
    enabled: bool = True

    steps: list[SegmentationStep] = field(
        default_factory=list
    )

    # Adaptive threshold
    block_size: int = 11
    c: int = 2

    # Binary polarity
    invert: bool = False


# ============================================================
# FEATURE ANALYSIS
# ============================================================

@dataclass
class FeatureConfig:
    enabled: bool = True

    connected_components: bool = True

    connectivity: int = 8

    min_area: int = 20


# ============================================================
# WATERMARK
# ============================================================

@dataclass
class WatermarkConfig:
    enabled: bool = False

# ============================================================
# OUTPUT
# ============================================================

@dataclass
class OutputConfig:
    base_dir: str = "output"

    save_detection: bool = True
    save_preprocessing: bool = True
    save_enhancement: bool = True
    save_segmentation: bool = True
    save_features: bool = True
    save_watermark: bool = True

    save_csv: bool = True


# ============================================================
# EXPERIMENT
# ============================================================

@dataclass
class ExperimentConfig:
    name: str = "default"

    save_results: bool = True


# ============================================================
# MAIN CONFIG
# ============================================================

@dataclass
class PipelineConfig:

    detector: DetectorConfig = field(
        default_factory=DetectorConfig
    )
    
    ocr: OcrConfig = field(
        default_factory=OcrConfig
    )

    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )

    enhancement: EnhancementConfig = field(
        default_factory=EnhancementConfig
    )

    segmentation: SegmentationConfig = field(
        default_factory=SegmentationConfig
    )

    feature: FeatureConfig = field(
        default_factory=FeatureConfig
    )

    watermark: WatermarkConfig = field(
        default_factory=WatermarkConfig
    )

    output: OutputConfig = field(
        default_factory=OutputConfig
    )

    experiment: ExperimentConfig = field(
        default_factory=ExperimentConfig
    )