"""
Color Cohesion Analyzer - Core Module
"""

from .color_space import (
    ColorConverter,
    PaletteColor,
    delta_e_76,
    delta_e_2000,
    get_warm_cool_balance,
    get_saturation_distribution,
    calculate_palette_entropy,
    detect_skin_tone_deviation
)

from .palette_extraction import (
    Palette,
    PaletteExtractor,
    PaletteAggregator,
    compute_palette_distance
)

from .video_processor import (
    Shot,
    VideoInfo,
    VideoAnalysis,
    VideoProcessor,
    compute_file_hash
)

from .metrics import (
    AssetMetrics,
    ProjectMetrics,
    ConflictMap,
    MetricsEngine,
    check_gamut_warnings,
    simulate_colorblind
)

__all__ = [
    # Color space
    'ColorConverter',
    'PaletteColor',
    'delta_e_76',
    'delta_e_2000',
    'get_warm_cool_balance',
    'get_saturation_distribution',
    'calculate_palette_entropy',
    'detect_skin_tone_deviation',
    
    # Palette extraction
    'Palette',
    'PaletteExtractor',
    'PaletteAggregator',
    'compute_palette_distance',
    
    # Video processing
    'Shot',
    'VideoInfo',
    'VideoAnalysis',
    'VideoProcessor',
    'compute_file_hash',
    
    # Metrics
    'AssetMetrics',
    'ProjectMetrics',
    'ConflictMap',
    'MetricsEngine',
    'check_gamut_warnings',
    'simulate_colorblind'
]
