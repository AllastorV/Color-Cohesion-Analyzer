"""
Color Cohesion Analyzer - Configuration
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import os

class SamplingMode(Enum):
    DRAFT = "draft"
    BALANCED = "balanced"
    ACCURATE = "accurate"

class ColorSpace(Enum):
    OKLAB = "oklab"
    CIELAB = "cielab"

@dataclass
class AnalysisConfig:
    """Configuration for color analysis"""
    # Color processing
    color_space: ColorSpace = ColorSpace.OKLAB
    palette_size: int = 8
    min_dominance_threshold: float = 0.01  # 1% minimum
    
    # Clustering
    clustering_iterations: int = 100
    use_mini_batch: bool = True
    batch_size: int = 1024
    
    # Video sampling
    sampling_mode: SamplingMode = SamplingMode.BALANCED
    draft_sample_interval: int = 30  # Every 30 frames
    balanced_sample_interval: int = 10
    accurate_sample_interval: int = 3
    
    # Shot detection
    shot_detection_threshold: float = 30.0
    min_shot_duration_frames: int = 12
    
    # Performance
    use_gpu: bool = False
    max_dimension: int = 512  # Downscale for processing
    cache_enabled: bool = True
    
    # Metrics
    deltaE_threshold: float = 10.0  # For outlier detection
    
    @property
    def sample_interval(self) -> int:
        if self.sampling_mode == SamplingMode.DRAFT:
            return self.draft_sample_interval
        elif self.sampling_mode == SamplingMode.BALANCED:
            return self.balanced_sample_interval
        return self.accurate_sample_interval

@dataclass
class UIConfig:
    """Configuration for UI appearance"""
    dark_mode: bool = True
    show_hex_labels: bool = True
    show_dominance: bool = True
    compact_mode: bool = False
    
    # Blueprint colors
    background_color: str = "#1a1f2e"
    grid_color: str = "#252b3d"
    node_color: str = "#2d3548"
    accent_color: str = "#4a9eff"
    text_color: str = "#e0e4eb"
    warning_color: str = "#ff6b4a"
    success_color: str = "#4aff6b"
    
    # Node sizing
    node_width: int = 180
    node_height: int = 120
    palette_node_width: int = 240
    palette_node_height: int = 160

@dataclass
class ExportConfig:
    """Configuration for exports"""
    output_directory: str = ""
    export_png_palettes: bool = True
    export_json_report: bool = True
    export_csv_timeline: bool = True
    export_ase: bool = True
    export_summary_png: bool = True
    export_lut: bool = False  # Experimental
    
    swatch_width: int = 100
    swatch_height: int = 100
    
    def get_output_path(self, filename: str) -> str:
        return os.path.join(self.output_directory, filename)

# Global configs
ANALYSIS_CONFIG = AnalysisConfig()
UI_CONFIG = UIConfig()
EXPORT_CONFIG = ExportConfig()

# Supported file formats
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp', '.exr'}
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.mxf', '.prores'}

def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_IMAGE_FORMATS

def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_VIDEO_FORMATS

def is_supported_file(path: str) -> bool:
    return is_image_file(path) or is_video_file(path)
