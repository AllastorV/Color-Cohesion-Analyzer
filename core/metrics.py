"""
Color Cohesion Analyzer - Metrics & Cohesion System
Creative decision support through comprehensive color analysis
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .color_space import (
    ColorConverter, delta_e_76, delta_e_2000,
    get_warm_cool_balance, get_saturation_distribution,
    calculate_palette_entropy, detect_skin_tone_deviation
)
from .palette_extraction import Palette, compute_palette_distance
from config import ANALYSIS_CONFIG


@dataclass
class AssetMetrics:
    """Comprehensive metrics for a single asset"""
    source_name: str
    source_type: str
    
    # Cohesion metrics
    distance_to_consensus: float = 0.0
    cohesion_score: float = 1.0  # 0-1, higher = more cohesive
    
    # Palette characteristics
    entropy: float = 0.0
    warm_cool_balance: float = 0.0  # -1 (cool) to 1 (warm)
    
    # Saturation
    saturation_mean: float = 0.0
    saturation_std: float = 0.0
    low_saturation_ratio: float = 0.0
    high_saturation_ratio: float = 0.0
    
    # Divergence
    divergent_colors: List[str] = field(default_factory=list)
    divergence_magnitude: float = 0.0
    
    # Optional
    skin_tone_deviation: float = 0.0
    
    # Flags
    is_outlier: bool = False
    
    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "cohesion_score": round(self.cohesion_score, 3),
            "distance_to_consensus": round(self.distance_to_consensus, 2),
            "entropy": round(self.entropy, 3),
            "warm_cool_balance": round(self.warm_cool_balance, 3),
            "saturation": {
                "mean": round(self.saturation_mean, 2),
                "std": round(self.saturation_std, 2),
                "low_ratio": round(self.low_saturation_ratio, 3),
                "high_ratio": round(self.high_saturation_ratio, 3)
            },
            "divergent_colors": self.divergent_colors,
            "divergence_magnitude": round(self.divergence_magnitude, 2),
            "skin_tone_deviation": round(self.skin_tone_deviation, 2),
            "is_outlier": self.is_outlier
        }


@dataclass  
class ProjectMetrics:
    """Aggregate metrics for entire project"""
    total_assets: int = 0
    total_images: int = 0
    total_videos: int = 0
    total_shots: int = 0
    
    # Overall cohesion
    average_cohesion: float = 0.0
    cohesion_variance: float = 0.0
    
    # Distribution stats
    warm_bias: float = 0.0
    cool_bias: float = 0.0
    neutral_ratio: float = 0.0
    
    # Outlier summary
    outlier_count: int = 0
    outlier_percentage: float = 0.0
    most_divergent_asset: str = ""
    most_divergent_score: float = 0.0
    
    # Color consistency
    dominant_hue_consistency: float = 0.0
    palette_entropy_avg: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "total_assets": self.total_assets,
            "composition": {
                "images": self.total_images,
                "videos": self.total_videos,
                "shots": self.total_shots
            },
            "cohesion": {
                "average": round(self.average_cohesion, 3),
                "variance": round(self.cohesion_variance, 4)
            },
            "temperature_distribution": {
                "warm_bias": round(self.warm_bias, 3),
                "cool_bias": round(self.cool_bias, 3),
                "neutral_ratio": round(self.neutral_ratio, 3)
            },
            "outliers": {
                "count": self.outlier_count,
                "percentage": round(self.outlier_percentage, 2),
                "most_divergent": self.most_divergent_asset,
                "divergence_score": round(self.most_divergent_score, 2)
            },
            "consistency": {
                "dominant_hue": round(self.dominant_hue_consistency, 3),
                "avg_entropy": round(self.palette_entropy_avg, 3)
            }
        }


@dataclass
class ConflictMap:
    """Map of conflicts/divergences in the project"""
    asset_pairs: List[Tuple[str, str, float]] = field(default_factory=list)  # (asset1, asset2, distance)
    color_conflicts: List[Dict] = field(default_factory=list)  # Colors causing conflicts
    
    def get_top_conflicts(self, n: int = 10) -> List[Tuple[str, str, float]]:
        """Get the n most divergent asset pairs"""
        sorted_pairs = sorted(self.asset_pairs, key=lambda x: x[2], reverse=True)
        return sorted_pairs[:n]
    
    def to_dict(self) -> dict:
        return {
            "top_conflicts": [
                {"asset1": a1, "asset2": a2, "distance": round(d, 2)}
                for a1, a2, d in self.get_top_conflicts(10)
            ],
            "color_conflicts": self.color_conflicts
        }


class MetricsEngine:
    """
    Compute comprehensive metrics for color cohesion analysis
    """
    
    def __init__(self):
        self.deltaE_threshold = ANALYSIS_CONFIG.deltaE_threshold
    
    def compute_asset_metrics(
        self,
        palette: Palette,
        consensus_palette: Optional[Palette] = None
    ) -> AssetMetrics:
        """
        Compute all metrics for a single asset
        """
        metrics = AssetMetrics(
            source_name=palette.source_name,
            source_type=palette.source_type
        )
        
        if not palette.colors:
            return metrics
        
        # Get Lab values
        lab_array = palette.get_lab_array()
        
        # Entropy
        metrics.entropy = palette.entropy
        
        # Warm/cool balance
        metrics.warm_cool_balance = get_warm_cool_balance(lab_array)
        
        # Saturation distribution
        sat_dist = get_saturation_distribution(lab_array)
        metrics.saturation_mean = sat_dist["mean"]
        metrics.saturation_std = sat_dist["std"]
        metrics.low_saturation_ratio = sat_dist["low_sat_ratio"]
        metrics.high_saturation_ratio = sat_dist["high_sat_ratio"]
        
        # Skin tone deviation
        metrics.skin_tone_deviation = detect_skin_tone_deviation(lab_array)
        
        # Distance to consensus (if provided)
        if consensus_palette and consensus_palette.colors:
            metrics.distance_to_consensus = compute_palette_distance(palette, consensus_palette)
            
            # Compute cohesion score (inverse of distance, normalized)
            # Using sigmoid-like function for smooth 0-1 range
            normalized_distance = metrics.distance_to_consensus / (self.deltaE_threshold * 2)
            metrics.cohesion_score = 1 / (1 + normalized_distance)
            
            # Find divergent colors
            consensus_labs = consensus_palette.get_lab_array()
            for color in palette.colors:
                color_lab = np.array(color.lab)
                min_dist = np.min(delta_e_76(color_lab, consensus_labs))
                
                if min_dist > self.deltaE_threshold:
                    metrics.divergent_colors.append(color.hex_code)
                    metrics.divergence_magnitude += min_dist * color.dominance
            
            # Mark as outlier if significant divergence
            metrics.is_outlier = (
                len(metrics.divergent_colors) >= len(palette.colors) / 2 or
                metrics.cohesion_score < 0.5
            )
        
        return metrics
    
    def compute_project_metrics(
        self,
        asset_metrics: List[AssetMetrics]
    ) -> ProjectMetrics:
        """
        Compute aggregate project metrics
        """
        if not asset_metrics:
            return ProjectMetrics()
        
        project = ProjectMetrics()
        project.total_assets = len(asset_metrics)
        
        # Count asset types
        for m in asset_metrics:
            if m.source_type == "image":
                project.total_images += 1
            elif m.source_type == "video":
                project.total_videos += 1
            elif m.source_type == "shot":
                project.total_shots += 1
        
        # Cohesion statistics
        cohesion_scores = [m.cohesion_score for m in asset_metrics]
        project.average_cohesion = np.mean(cohesion_scores)
        project.cohesion_variance = np.var(cohesion_scores)
        
        # Temperature distribution
        warm_cool_values = [m.warm_cool_balance for m in asset_metrics]
        project.warm_bias = np.mean([v for v in warm_cool_values if v > 0.1]) if any(v > 0.1 for v in warm_cool_values) else 0
        project.cool_bias = abs(np.mean([v for v in warm_cool_values if v < -0.1])) if any(v < -0.1 for v in warm_cool_values) else 0
        project.neutral_ratio = sum(1 for v in warm_cool_values if -0.1 <= v <= 0.1) / len(warm_cool_values)
        
        # Outlier statistics
        outliers = [m for m in asset_metrics if m.is_outlier]
        project.outlier_count = len(outliers)
        project.outlier_percentage = (project.outlier_count / project.total_assets) * 100
        
        # Most divergent
        most_divergent = max(asset_metrics, key=lambda m: m.distance_to_consensus)
        project.most_divergent_asset = most_divergent.source_name
        project.most_divergent_score = most_divergent.distance_to_consensus
        
        # Entropy average
        entropies = [m.entropy for m in asset_metrics]
        project.palette_entropy_avg = np.mean(entropies)
        
        return project
    
    def compute_conflict_map(
        self,
        palettes: List[Palette]
    ) -> ConflictMap:
        """
        Compute conflict map between all asset pairs
        """
        conflict_map = ConflictMap()
        
        n = len(palettes)
        for i in range(n):
            for j in range(i + 1, n):
                distance = compute_palette_distance(palettes[i], palettes[j])
                
                # Only record significant divergences
                if distance > self.deltaE_threshold:
                    conflict_map.asset_pairs.append((
                        palettes[i].source_name,
                        palettes[j].source_name,
                        distance
                    ))
        
        # Identify specific color conflicts
        # Find colors that appear in some palettes but strongly conflict with others
        all_colors = []
        for palette in palettes:
            for color in palette.colors:
                all_colors.append({
                    "hex": color.hex_code,
                    "lab": color.lab,
                    "source": palette.source_name
                })
        
        # Cluster colors and find those with high variance across palettes
        if len(all_colors) > 1:
            try:
                from sklearn.cluster import DBSCAN
                
                labs = np.array([c["lab"] for c in all_colors])
                
                # DBSCAN to find color clusters
                clustering = DBSCAN(eps=self.deltaE_threshold, min_samples=2).fit(labs)
                
                # Find colors that don't cluster well (outliers)
                outlier_indices = np.where(clustering.labels_ == -1)[0]
                
                for idx in outlier_indices[:10]:  # Limit to top 10
                    conflict_map.color_conflicts.append({
                        "hex": all_colors[idx]["hex"],
                        "source": all_colors[idx]["source"],
                        "type": "isolated_color"
                    })
            except ImportError:
                # sklearn not available, skip DBSCAN clustering
                pass
        
        return conflict_map
    
    def generate_cohesion_report(
        self,
        palettes: List[Palette],
        consensus_palette: Palette
    ) -> Dict:
        """
        Generate comprehensive cohesion report
        """
        # Compute metrics for each asset
        asset_metrics = []
        for palette in palettes:
            metrics = self.compute_asset_metrics(palette, consensus_palette)
            asset_metrics.append(metrics)
        
        # Compute project metrics
        project_metrics = self.compute_project_metrics(asset_metrics)
        
        # Compute conflict map
        conflict_map = self.compute_conflict_map(palettes)
        
        return {
            "project": project_metrics.to_dict(),
            "assets": [m.to_dict() for m in asset_metrics],
            "conflicts": conflict_map.to_dict(),
            "consensus_palette": consensus_palette.to_dict(),
            "recommendations": self._generate_recommendations(project_metrics, asset_metrics)
        }
    
    def _generate_recommendations(
        self,
        project: ProjectMetrics,
        assets: List[AssetMetrics]
    ) -> List[str]:
        """
        Generate actionable recommendations based on metrics
        """
        recommendations = []
        
        if project.average_cohesion < 0.5:
            recommendations.append(
                "Overall color cohesion is low. Consider reviewing assets that diverge "
                "significantly from the consensus palette."
            )
        
        if project.outlier_percentage > 30:
            recommendations.append(
                f"{project.outlier_count} assets ({project.outlier_percentage:.0f}%) are marked as outliers. "
                "This may indicate intentional stylistic variation or inconsistency issues."
            )
        
        if project.cohesion_variance > 0.1:
            recommendations.append(
                "High variance in cohesion scores suggests inconsistent color treatment "
                "across different parts of the project."
            )
        
        # Temperature recommendations
        if project.warm_bias > 0.3 and project.cool_bias > 0.3:
            recommendations.append(
                "Mixed warm and cool palettes detected. Verify this is intentional "
                "for contrast or scene differentiation."
            )
        
        # Specific asset recommendations
        high_divergence_assets = [a for a in assets if a.divergence_magnitude > 20]
        if high_divergence_assets:
            names = [a.source_name for a in high_divergence_assets[:3]]
            recommendations.append(
                f"Assets with high color divergence: {', '.join(names)}. "
                "These may need color grading attention."
            )
        
        if not recommendations:
            recommendations.append(
                "Color cohesion analysis looks good. The project maintains "
                "consistent color language across assets."
            )
        
        return recommendations


# Gamut warning thresholds
GAMUT_REC709 = {
    "r_max": 255, "g_max": 255, "b_max": 255,
    "name": "Rec.709 (sRGB)"
}

GAMUT_P3 = {
    "r_max": 255, "g_max": 255, "b_max": 255,  # Simplified
    "name": "DCI-P3"
}

GAMUT_REC2020 = {
    "r_max": 255, "g_max": 255, "b_max": 255,  # Simplified
    "name": "Rec.2020"
}


def check_gamut_warnings(palette: Palette, target_gamut: str = "rec709") -> List[Dict]:
    """
    Check if palette colors might have gamut issues
    This is a simplified check - full implementation would need ICC profiles
    """
    warnings = []
    
    # For now, check for highly saturated colors that might clip
    for color in palette.colors:
        lab = np.array(color.lab)
        chroma = np.sqrt(lab[1]**2 + lab[2]**2)
        
        # High chroma colors are more likely to have gamut issues
        if chroma > 100:
            warnings.append({
                "hex": color.hex_code,
                "message": f"High saturation color may exceed {target_gamut} gamut",
                "severity": "warning"
            })
        elif chroma > 80:
            warnings.append({
                "hex": color.hex_code,
                "message": f"Saturated color - verify in target colorspace",
                "severity": "info"
            })
    
    return warnings


# Colorblind simulation matrices
COLORBLIND_MATRICES = {
    "protanopia": np.array([
        [0.567, 0.433, 0],
        [0.558, 0.442, 0],
        [0, 0.242, 0.758]
    ]),
    "deuteranopia": np.array([
        [0.625, 0.375, 0],
        [0.7, 0.3, 0],
        [0, 0.3, 0.7]
    ]),
    "tritanopia": np.array([
        [0.95, 0.05, 0],
        [0, 0.433, 0.567],
        [0, 0.475, 0.525]
    ])
}


def simulate_colorblind(rgb: np.ndarray, condition: str = "deuteranopia") -> np.ndarray:
    """
    Simulate how colors appear to colorblind viewers
    """
    if condition not in COLORBLIND_MATRICES:
        return rgb
    
    matrix = COLORBLIND_MATRICES[condition]
    rgb_normalized = rgb.astype(np.float64) / 255
    
    if rgb_normalized.ndim == 1:
        simulated = matrix @ rgb_normalized
    else:
        simulated = rgb_normalized @ matrix.T
    
    return (np.clip(simulated, 0, 1) * 255).astype(np.uint8)
