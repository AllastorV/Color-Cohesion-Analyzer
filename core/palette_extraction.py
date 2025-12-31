"""
Color Cohesion Analyzer - Palette Extraction Engine
Perceptual clustering using k-means in perceptual color spaces
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None
    MiniBatchKMeans = None

from .color_space import (
    ColorConverter, PaletteColor, delta_e_76,
    calculate_palette_entropy, get_warm_cool_balance,
    get_saturation_distribution
)
from config import ANALYSIS_CONFIG, ColorSpace


@dataclass
class Palette:
    """Represents an extracted color palette"""
    colors: List[PaletteColor]
    source_name: str
    source_type: str  # "image", "video", "shot", "consensus", "average", "outlier"
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def entropy(self) -> float:
        return calculate_palette_entropy(self.colors)
    
    @property
    def dominant_color(self) -> Optional[PaletteColor]:
        if not self.colors:
            return None
        return max(self.colors, key=lambda c: c.dominance)
    
    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "colors": [c.to_dict() for c in self.colors],
            "entropy": self.entropy,
            "metadata": self.metadata
        }
    
    def get_lab_array(self) -> np.ndarray:
        """Get palette colors as Lab array"""
        return np.array([c.lab for c in self.colors])


class PaletteExtractor:
    """
    Extract perceptually accurate color palettes from images
    Uses clustering in OKLab or CIELAB color space
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.palette_size = ANALYSIS_CONFIG.palette_size
        self.use_mini_batch = ANALYSIS_CONFIG.use_mini_batch
        self.batch_size = ANALYSIS_CONFIG.batch_size
        self.max_iterations = ANALYSIS_CONFIG.clustering_iterations
        self.min_dominance = ANALYSIS_CONFIG.min_dominance_threshold
        self.color_space = ANALYSIS_CONFIG.color_space
        
        if config:
            self.__dict__.update(config)
    
    def _rgb_to_perceptual(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to perceptual color space"""
        if self.color_space == ColorSpace.OKLAB:
            return ColorConverter.rgb_to_oklab(rgb)
        return ColorConverter.rgb_to_lab(rgb)
    
    def _perceptual_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        """Convert perceptual space back to RGB"""
        if self.color_space == ColorSpace.OKLAB:
            return ColorConverter.oklab_to_rgb(lab)
        return ColorConverter.lab_to_rgb(lab)
    
    def extract_from_pixels(
        self,
        pixels: np.ndarray,
        source_name: str = "unknown",
        source_type: str = "image"
    ) -> Palette:
        """
        Extract palette from pixel array
        
        Args:
            pixels: RGB pixels array (N, 3) with values 0-255
            source_name: Name identifier for the source
            source_type: Type of source (image, video, shot)
        
        Returns:
            Palette object with extracted colors
        """
        if len(pixels) == 0:
            return Palette([], source_name, source_type)
        
        # Convert to perceptual space
        lab_pixels = self._rgb_to_perceptual(pixels)
        
        # Check if sklearn is available
        if not SKLEARN_AVAILABLE:
            # Fallback: simple quantization without clustering
            return self._fallback_palette_extraction(lab_pixels, pixels, source_name, source_type)
        
        # Perform clustering
        n_clusters = min(self.palette_size, len(lab_pixels))
        
        if self.use_mini_batch and len(lab_pixels) > self.batch_size:
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters,
                max_iter=self.max_iterations,
                batch_size=self.batch_size,
                n_init=3,
                random_state=42
            )
        else:
            clusterer = KMeans(
                n_clusters=n_clusters,
                max_iter=self.max_iterations,
                n_init=10,
                random_state=42
            )
        
        labels = clusterer.fit_predict(lab_pixels)
        centers = clusterer.cluster_centers_
        
        # Calculate dominance for each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_pixels = len(pixels)
        
        palette_colors = []
        for i, center_lab in enumerate(centers):
            dominance = counts[i] / total_pixels if i < len(counts) else 0
            
            # Skip colors below threshold
            if dominance < self.min_dominance:
                continue
            
            # Convert center back to RGB
            center_rgb = self._perceptual_to_rgb(center_lab)
            
            palette_colors.append(PaletteColor(
                hex_code=ColorConverter.rgb_to_hex(tuple(center_rgb)),
                rgb=tuple(center_rgb),
                lab=tuple(center_lab),
                dominance=dominance
            ))
        
        # Sort by dominance
        palette_colors.sort(key=lambda c: c.dominance, reverse=True)
        
        return Palette(palette_colors, source_name, source_type)
    
    def _fallback_palette_extraction(
        self,
        lab_pixels: np.ndarray,
        rgb_pixels: np.ndarray,
        source_name: str,
        source_type: str
    ) -> Palette:
        """
        Fallback palette extraction when sklearn is not available
        Uses simple histogram-based color quantization
        """
        # Simple approach: divide color space into bins
        n_bins = self.palette_size
        
        # Quantize Lab values
        L_bins = np.linspace(0, 100, n_bins + 1)
        
        # Find representative colors by averaging in bins
        palette_colors = []
        total_pixels = len(rgb_pixels)
        
        # Sort pixels by L channel and divide into bins
        l_values = lab_pixels[:, 0]
        sorted_indices = np.argsort(l_values)
        bin_size = len(sorted_indices) // n_bins
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_indices)
            
            if start_idx >= end_idx:
                continue
            
            bin_indices = sorted_indices[start_idx:end_idx]
            bin_lab = lab_pixels[bin_indices]
            bin_rgb = rgb_pixels[bin_indices]
            
            # Average color in bin
            avg_lab = np.mean(bin_lab, axis=0)
            avg_rgb = np.mean(bin_rgb, axis=0).astype(np.uint8)
            
            dominance = len(bin_indices) / total_pixels
            
            if dominance >= self.min_dominance:
                palette_colors.append(PaletteColor(
                    hex_code=ColorConverter.rgb_to_hex(tuple(avg_rgb)),
                    rgb=tuple(avg_rgb),
                    lab=tuple(avg_lab),
                    dominance=dominance
                ))
        
        palette_colors.sort(key=lambda c: c.dominance, reverse=True)
        return Palette(palette_colors[:self.palette_size], source_name, source_type)
    
    def extract_from_image(
        self,
        image: np.ndarray,
        source_name: str = "unknown",
        max_pixels: int = 50000
    ) -> Palette:
        """
        Extract palette from image array
        
        Args:
            image: RGB image array (H, W, 3)
            source_name: Name identifier
            max_pixels: Maximum pixels to sample
        
        Returns:
            Palette object
        """
        # Flatten image to pixels
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3)
        
        # Subsample if too many pixels
        if len(pixels) > max_pixels:
            indices = np.random.choice(len(pixels), max_pixels, replace=False)
            pixels = pixels[indices]
        
        return self.extract_from_pixels(pixels, source_name, "image")


class PaletteAggregator:
    """
    Aggregate multiple palettes to compute consensus, average, and outliers
    Implements the Three-Center Model
    """
    
    def __init__(self):
        self.palettes: List[Palette] = []
        self.extractor = PaletteExtractor()
    
    def add_palette(self, palette: Palette):
        """Add a palette to the aggregation"""
        self.palettes.append(palette)
    
    def clear(self):
        """Clear all palettes"""
        self.palettes = []
    
    def _fallback_global_average(self, all_labs: np.ndarray, all_weights: np.ndarray) -> Palette:
        """Fallback when sklearn not available"""
        # Simple weighted average
        n_colors = min(ANALYSIS_CONFIG.palette_size, len(all_labs))
        
        # Sort by weight and pick top colors
        sorted_indices = np.argsort(all_weights)[::-1]
        
        colors = []
        for i in range(n_colors):
            idx = sorted_indices[i % len(sorted_indices)]
            lab = all_labs[idx]
            rgb = self.extractor._perceptual_to_rgb(lab)
            
            colors.append(PaletteColor(
                hex_code=ColorConverter.rgb_to_hex(tuple(rgb)),
                rgb=tuple(rgb),
                lab=tuple(lab),
                dominance=all_weights[idx]
            ))
        
        # Normalize
        total = sum(c.dominance for c in colors)
        if total > 0:
            for c in colors:
                c.dominance /= total
        
        return Palette(colors, "Global Average", "average")
    
    def _fallback_consensus(self, all_labs: np.ndarray, all_weights: np.ndarray, min_presence: float) -> Palette:
        """Fallback consensus when sklearn not available"""
        # Use weighted average approach
        return self._fallback_global_average(all_labs, all_weights)
    
    def _collect_all_colors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collect all colors with their weights from all palettes"""
        all_labs = []
        all_weights = []
        
        for palette in self.palettes:
            for color in palette.colors:
                all_labs.append(color.lab)
                all_weights.append(color.dominance)
        
        return np.array(all_labs), np.array(all_weights)
    
    def compute_global_average(self) -> Palette:
        """
        Compute Global Average Palette
        Mathematical weighted average across all assets
        """
        if not self.palettes:
            return Palette([], "Global Average", "average")
        
        all_labs, all_weights = self._collect_all_colors()
        
        if len(all_labs) == 0:
            return Palette([], "Global Average", "average")
        
        # Check if sklearn is available
        if not SKLEARN_AVAILABLE:
            return self._fallback_global_average(all_labs, all_weights)
        
        # Weighted k-means clustering
        n_clusters = min(ANALYSIS_CONFIG.palette_size, len(all_labs))
        
        clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            max_iter=100,
            n_init=3,
            random_state=42
        )
        
        # Weight samples by dominance
        sample_weight = all_weights / all_weights.sum()
        repeated_indices = np.random.choice(
            len(all_labs), 
            size=min(10000, len(all_labs) * 10),
            p=sample_weight
        )
        weighted_samples = all_labs[repeated_indices]
        
        labels = clusterer.fit_predict(weighted_samples)
        centers = clusterer.cluster_centers_
        
        # Calculate new dominances
        unique_labels, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        colors = []
        for i, center in enumerate(centers):
            dominance = counts[i] / total if i < len(counts) else 0
            rgb = self.extractor._perceptual_to_rgb(center)
            
            colors.append(PaletteColor(
                hex_code=ColorConverter.rgb_to_hex(tuple(rgb)),
                rgb=tuple(rgb),
                lab=tuple(center),
                dominance=dominance
            ))
        
        colors.sort(key=lambda c: c.dominance, reverse=True)
        
        return Palette(colors, "Global Average", "average")
    
    def compute_consensus(self, min_presence: float = 0.5) -> Palette:
        """
        Compute Consensus Palette
        Colors that appear consistently across majority of assets
        
        Args:
            min_presence: Minimum fraction of palettes a color must appear in
        """
        if not self.palettes:
            return Palette([], "Consensus", "consensus")
        
        all_labs, all_weights = self._collect_all_colors()
        
        if len(all_labs) == 0:
            return Palette([], "Consensus", "consensus")
        
        # Check if sklearn is available
        if not SKLEARN_AVAILABLE:
            return self._fallback_consensus(all_labs, all_weights, min_presence)
        
        # First cluster all colors
        n_clusters = ANALYSIS_CONFIG.palette_size * 2
        clusterer = KMeans(n_clusters=min(n_clusters, len(all_labs)), n_init=10, random_state=42)
        
        all_labels = clusterer.fit_predict(all_labs)
        
        # Track which palettes contribute to each cluster
        cluster_palette_presence = {i: set() for i in range(clusterer.n_clusters)}
        cluster_weights = {i: [] for i in range(clusterer.n_clusters)}
        
        idx = 0
        for pal_idx, palette in enumerate(self.palettes):
            for color in palette.colors:
                cluster_id = all_labels[idx]
                cluster_palette_presence[cluster_id].add(pal_idx)
                cluster_weights[cluster_id].append(all_weights[idx])
                idx += 1
        
        # Select clusters present in sufficient palettes
        n_palettes = len(self.palettes)
        consensus_clusters = []
        
        for cluster_id, palettes in cluster_palette_presence.items():
            presence = len(palettes) / n_palettes
            if presence >= min_presence:
                avg_weight = np.mean(cluster_weights[cluster_id])
                consensus_clusters.append((cluster_id, presence, avg_weight))
        
        # Sort by presence * weight
        consensus_clusters.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        # Build consensus palette
        colors = []
        for cluster_id, presence, _ in consensus_clusters[:ANALYSIS_CONFIG.palette_size]:
            center = clusterer.cluster_centers_[cluster_id]
            rgb = self.extractor._perceptual_to_rgb(center)
            
            # Dominance based on presence across palettes
            dominance = presence / len(consensus_clusters) if consensus_clusters else 0
            
            colors.append(PaletteColor(
                hex_code=ColorConverter.rgb_to_hex(tuple(rgb)),
                rgb=tuple(rgb),
                lab=tuple(center),
                dominance=dominance
            ))
        
        # Normalize dominances
        if colors:
            total = sum(c.dominance for c in colors)
            for c in colors:
                c.dominance = c.dominance / total if total > 0 else 0
        
        return Palette(colors, "Consensus", "consensus")
    
    def detect_outliers(self, threshold: float = None) -> Tuple[Palette, List[Tuple[str, float]]]:
        """
        Detect outlier colors and assets
        
        Returns:
            - Outlier palette with divergent colors
            - List of (asset_name, divergence_score) tuples
        """
        if threshold is None:
            threshold = ANALYSIS_CONFIG.deltaE_threshold
        
        consensus = self.compute_consensus()
        if not consensus.colors:
            return Palette([], "Outliers", "outlier"), []
        
        consensus_labs = np.array([c.lab for c in consensus.colors])
        
        outlier_colors = []
        asset_divergences = []
        
        for palette in self.palettes:
            palette_divergence = 0
            divergent_count = 0
            
            for color in palette.colors:
                # Find minimum distance to any consensus color
                color_lab = np.array(color.lab)
                distances = delta_e_76(color_lab, consensus_labs)
                min_distance = np.min(distances)
                
                if min_distance > threshold:
                    outlier_colors.append(color)
                    divergent_count += 1
                
                palette_divergence += min_distance * color.dominance
            
            asset_divergences.append((palette.source_name, palette_divergence))
        
        # Sort by divergence
        asset_divergences.sort(key=lambda x: x[1], reverse=True)
        
        # Deduplicate outlier colors by clustering
        if outlier_colors:
            outlier_labs = np.array([c.lab for c in outlier_colors])
            n_outliers = min(ANALYSIS_CONFIG.palette_size, len(outlier_labs))
            
            if n_outliers > 0 and SKLEARN_AVAILABLE:
                clusterer = KMeans(n_clusters=n_outliers, n_init=5, random_state=42)
                clusterer.fit(outlier_labs)
                
                final_outliers = []
                for center in clusterer.cluster_centers_:
                    rgb = self.extractor._perceptual_to_rgb(center)
                    final_outliers.append(PaletteColor(
                        hex_code=ColorConverter.rgb_to_hex(tuple(rgb)),
                        rgb=tuple(rgb),
                        lab=tuple(center),
                        dominance=1.0 / n_outliers
                    ))
                
                outlier_colors = final_outliers
        
        return Palette(outlier_colors, "Outliers", "outlier"), asset_divergences


def compute_palette_distance(palette1: Palette, palette2: Palette) -> float:
    """
    Compute perceptual distance between two palettes
    Uses weighted average of minimum color distances
    """
    if not palette1.colors or not palette2.colors:
        return float('inf')
    
    labs1 = np.array([c.lab for c in palette1.colors])
    labs2 = np.array([c.lab for c in palette2.colors])
    weights1 = np.array([c.dominance for c in palette1.colors])
    
    total_distance = 0
    for i, lab1 in enumerate(labs1):
        distances = delta_e_76(lab1, labs2)
        min_dist = np.min(distances)
        total_distance += min_dist * weights1[i]
    
    return total_distance / weights1.sum() if weights1.sum() > 0 else float('inf')
