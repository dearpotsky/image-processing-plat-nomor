from dataclasses import dataclass, field


# ============================================================
# DETECTOR
# ============================================================

@dataclass
class DetectorConfig:
    model_path: str = (
        "models/plate_detector/"
        "plate_detector_yolo11n.pt"
    )

    confidence: float = 0.25


# ============================================================
# PREPROCESSING
# ============================================================

@dataclass
class PreprocessingConfig:
    enabled: bool = True

    target_width: int = 400
    target_height: int = 120

    interpolation: str = "INTER_AREA"

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
class EnhancementConfig:
    enabled: bool = True

    method: str = "clahe"

    clip_limit: float = 2.0
    tile_grid_width: int = 8
    tile_grid_height: int = 8

    @property
    def tile_grid_size(self):
        return (
            self.tile_grid_width,
            self.tile_grid_height,
        )


# ============================================================
# SEGMENTATION
# ============================================================

@dataclass
class SegmentationConfig:
    enabled: bool = True

    method: str = "otsu"

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

    output: OutputConfig = field(
        default_factory=OutputConfig
    )

    experiment: ExperimentConfig = field(
        default_factory=ExperimentConfig
    )