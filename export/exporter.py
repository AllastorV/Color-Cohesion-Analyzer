"""
Color Cohesion Analyzer - Export System
Export palettes, reports, and creative tools for production pipelines
"""

import json
import csv
import struct
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import os

from config import EXPORT_CONFIG, UI_CONFIG

# Type hints only - avoid circular imports at runtime
if TYPE_CHECKING:
    from core.palette_extraction import Palette
    from core.color_space import ColorConverter

# Lazy imports to avoid circular dependencies
def _get_palette_class():
    from core.palette_extraction import Palette
    return Palette

def _get_color_converter():
    from core.color_space import ColorConverter
    return ColorConverter


class PaletteExporter:
    """
    Export palettes in various formats for pipeline integration
    """
    
    def __init__(self, output_dir: str = ""):
        self.output_dir = output_dir or EXPORT_CONFIG.output_directory
        self.swatch_width = EXPORT_CONFIG.swatch_width
        self.swatch_height = EXPORT_CONFIG.swatch_height
    
    def set_output_dir(self, path: str):
        """Set output directory"""
        self.output_dir = path
        Path(path).mkdir(parents=True, exist_ok=True)
    
    def export_palette_png(
        self,
        palette,  # Palette object
        filename: str = None,
        show_hex: bool = True,
        show_dominance: bool = True,
        horizontal: bool = True
    ) -> str:
        """
        Export palette as PNG swatch image
        
        Args:
            palette: Palette object to export
            
        Returns path to exported file
        """
        if not palette.colors:
            return ""
        
        n_colors = len(palette.colors)
        
        # Calculate dimensions
        text_height = 40 if (show_hex or show_dominance) else 0
        
        if horizontal:
            width = self.swatch_width * n_colors
            height = self.swatch_height + text_height
        else:
            width = self.swatch_width + 150  # Space for text
            height = self.swatch_height * n_colors
        
        # Create image
        img = Image.new('RGB', (width, height), color='#1a1f2e')
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            font_small = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
            font_small = font
        
        for i, color in enumerate(palette.colors):
            if horizontal:
                x = i * self.swatch_width
                y = 0
                rect = (x, y, x + self.swatch_width, y + self.swatch_height)
                text_x = x + 5
                text_y = self.swatch_height + 5
            else:
                x = 0
                y = i * self.swatch_height
                rect = (x, y, x + self.swatch_width, y + self.swatch_height)
                text_x = self.swatch_width + 10
                text_y = y + 10
            
            # Draw color swatch
            draw.rectangle(rect, fill=color.hex_code)
            
            # Draw text
            if show_hex or show_dominance:
                text_lines = []
                if show_hex:
                    text_lines.append(color.hex_code.upper())
                if show_dominance:
                    text_lines.append(f"{color.dominance * 100:.1f}%")
                
                for j, line in enumerate(text_lines):
                    draw.text(
                        (text_x, text_y + j * 15),
                        line,
                        fill='#e0e4eb',
                        font=font_small
                    )
        
        # Save
        if filename is None:
            filename = f"{palette.source_name}_palette.png"
        
        filepath = os.path.join(self.output_dir, filename)
        img.save(filepath)
        
        return filepath
    
    def export_palette_json(self, palette, filename: str = None) -> str:
        """Export palette as JSON"""
        if filename is None:
            filename = f"{palette.source_name}_palette.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(palette.to_dict(), f, indent=2)
        
        return filepath
    
    def export_palette_csv(self, palette, filename: str = None) -> str:
        """Export palette as CSV"""
        if filename is None:
            filename = f"{palette.source_name}_palette.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Hex', 'R', 'G', 'B', 'L', 'a', 'b', 'Dominance %'])
            
            for color in palette.colors:
                writer.writerow([
                    color.hex_code,
                    color.rgb[0], color.rgb[1], color.rgb[2],
                    round(color.lab[0], 2),
                    round(color.lab[1], 2),
                    round(color.lab[2], 2),
                    round(color.dominance * 100, 2)
                ])
        
        return filepath
    
    def export_ase(self, palette, filename: str = None) -> str:
        """
        Export palette as Adobe Swatch Exchange (ASE) file
        """
        if filename is None:
            filename = f"{palette.source_name}.ase"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # ASE file structure
        with open(filepath, 'wb') as f:
            # Header: 'ASEF'
            f.write(b'ASEF')
            
            # Version: 1.0
            f.write(struct.pack('>HH', 1, 0))
            
            # Number of blocks
            f.write(struct.pack('>I', len(palette.colors)))
            
            for color in palette.colors:
                # Block type: 0x0001 = color entry
                f.write(struct.pack('>H', 0x0001))
                
                # Color name (UTF-16BE with length)
                name = color.hex_code.encode('utf-16-be')
                name_length = len(name) // 2 + 1  # +1 for null terminator
                
                # Block length
                block_length = 2 + name_length * 2 + 4 + 4 * 3 + 2
                f.write(struct.pack('>I', block_length))
                
                # Name length and name
                f.write(struct.pack('>H', name_length))
                f.write(name)
                f.write(b'\x00\x00')  # Null terminator
                
                # Color model: 'RGB '
                f.write(b'RGB ')
                
                # RGB values (0-1 floats)
                r = color.rgb[0] / 255.0
                g = color.rgb[1] / 255.0
                b = color.rgb[2] / 255.0
                f.write(struct.pack('>fff', r, g, b))
                
                # Color type: 0 = global
                f.write(struct.pack('>H', 0))
        
        return filepath
    
    def export_cube_lut(
        self,
        palette,  # Palette object
        filename: str = None,
        lut_size: int = 17
    ) -> str:
        """
        Export experimental creative LUT based on palette
        
        WARNING: This is a creative suggestion tool, not a final grade.
        The LUT shifts colors towards the palette's dominant tones.
        """
        if filename is None:
            filename = f"{palette.source_name}_creative.cube"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if not palette.colors:
            return ""
        
        # Get dominant colors
        dominant_labs = np.array([c.lab for c in palette.colors[:3]])
        weights = np.array([c.dominance for c in palette.colors[:3]])
        weights = weights / weights.sum()
        
        # Calculate average target Lab
        target_lab = np.average(dominant_labs, axis=0, weights=weights)
        
        with open(filepath, 'w') as f:
            f.write("# Creative LUT generated by Color Cohesion Analyzer\n")
            f.write("# WARNING: This is a creative suggestion, not a calibrated transform\n")
            f.write(f"# Based on palette: {palette.source_name}\n")
            f.write(f"TITLE \"{palette.source_name} Creative\"\n")
            f.write(f"LUT_3D_SIZE {lut_size}\n\n")
            
            # Get ColorConverter lazily
            ColorConverter = _get_color_converter()
            
            # Generate LUT entries
            for b in range(lut_size):
                for g in range(lut_size):
                    for r in range(lut_size):
                        # Normalize to 0-1
                        r_in = r / (lut_size - 1)
                        g_in = g / (lut_size - 1)
                        b_in = b / (lut_size - 1)
                        
                        # Convert to RGB 0-255
                        rgb = np.array([r_in, g_in, b_in]) * 255
                        
                        # Convert to Lab
                        lab = ColorConverter.rgb_to_lab(rgb.astype(np.uint8))
                        
                        # Subtle shift towards target palette
                        # Only adjust a* and b* channels slightly
                        shift_strength = 0.1  # Very subtle
                        lab[1] = lab[1] + (target_lab[1] - lab[1]) * shift_strength
                        lab[2] = lab[2] + (target_lab[2] - lab[2]) * shift_strength
                        
                        # Convert back to RGB
                        rgb_out = ColorConverter.lab_to_rgb(lab)
                        
                        # Normalize to 0-1
                        r_out = rgb_out[0] / 255
                        g_out = rgb_out[1] / 255
                        b_out = rgb_out[2] / 255
                        
                        f.write(f"{r_out:.6f} {g_out:.6f} {b_out:.6f}\n")
        
        return filepath


class ReportExporter:
    """
    Export comprehensive analysis reports
    """
    
    def __init__(self, output_dir: str = ""):
        self.output_dir = output_dir or EXPORT_CONFIG.output_directory
    
    def set_output_dir(self, path: str):
        """Set output directory"""
        self.output_dir = path
        Path(path).mkdir(parents=True, exist_ok=True)
    
    def export_full_report(
        self,
        report_data: Dict,
        filename: str = "report.json"
    ) -> str:
        """Export complete analysis report as JSON"""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return filepath
    
    def export_timeline_csv(
        self,
        timeline_data: List,  # List of (time, Palette) tuples
        filename: str = "timeline.csv"
    ) -> str:
        """Export video palette timeline as CSV"""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Time (s)', 'Color 1', 'Color 2', 'Color 3', 'Color 4',
                'Dominant Hex', 'Entropy', 'Warm/Cool'
            ])
            
            for time, palette in timeline_data:
                colors = [c.hex_code for c in palette.colors[:4]]
                while len(colors) < 4:
                    colors.append('')
                
                dominant = palette.dominant_color
                dominant_hex = dominant.hex_code if dominant else ''
                
                # Calculate warm/cool from Lab b* channel
                if palette.colors:
                    avg_b = np.mean([c.lab[2] for c in palette.colors])
                    warm_cool = "warm" if avg_b > 5 else "cool" if avg_b < -5 else "neutral"
                else:
                    warm_cool = "unknown"
                
                writer.writerow([
                    round(time, 3),
                    *colors,
                    dominant_hex,
                    round(palette.entropy, 3),
                    warm_cool
                ])
        
        return filepath
    
    def export_summary_csv(
        self,
        assets_data: List[Dict],
        filename: str = "summary.csv"
    ) -> str:
        """Export asset summary as CSV"""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Asset Name', 'Type', 'Cohesion Score',
                'Distance to Consensus', 'Entropy',
                'Warm/Cool Balance', 'Is Outlier'
            ])
            
            for asset in assets_data:
                writer.writerow([
                    asset.get('source_name', ''),
                    asset.get('source_type', ''),
                    asset.get('cohesion_score', 0),
                    asset.get('distance_to_consensus', 0),
                    asset.get('entropy', 0),
                    asset.get('warm_cool_balance', 0),
                    asset.get('is_outlier', False)
                ])
        
        return filepath


class ProjectExporter:
    """
    Export complete project package
    """
    
    def __init__(self, base_output_dir: str = ""):
        self.base_dir = base_output_dir
        self.palette_exporter = PaletteExporter()
        self.report_exporter = ReportExporter()
    
    def export_project(
        self,
        project_name: str,
        palettes: List,  # List of Palette objects
        consensus_palette,  # Palette object
        average_palette,  # Palette object
        outlier_palette,  # Palette object (can be None)
        report_data: Dict,
        timeline_data: Optional[List] = None,
        blueprint_snapshot: Optional[np.ndarray] = None
    ) -> str:
        """
        Export complete project package
        
        Structure:
        /project_name/
            /palettes/
                consensus.png
                average.png
                outliers.png
                asset_1.png
                ...
            /data/
                report.json
                summary.csv
                timeline.csv (if video)
                *.ase
            summary.png (blueprint snapshot)
        """
        # Create directories
        project_dir = os.path.join(self.base_dir, project_name)
        palettes_dir = os.path.join(project_dir, "palettes")
        data_dir = os.path.join(project_dir, "data")
        
        Path(palettes_dir).mkdir(parents=True, exist_ok=True)
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure exporters
        self.palette_exporter.set_output_dir(palettes_dir)
        self.report_exporter.set_output_dir(data_dir)
        
        exported_files = []
        
        # Export central palettes
        if consensus_palette.colors:
            path = self.palette_exporter.export_palette_png(
                consensus_palette, "consensus.png"
            )
            exported_files.append(path)
            
            # ASE export
            self.palette_exporter.set_output_dir(data_dir)
            self.palette_exporter.export_ase(consensus_palette, "consensus.ase")
            self.palette_exporter.set_output_dir(palettes_dir)
        
        if average_palette.colors:
            path = self.palette_exporter.export_palette_png(
                average_palette, "average.png"
            )
            exported_files.append(path)
        
        if outlier_palette.colors:
            path = self.palette_exporter.export_palette_png(
                outlier_palette, "outliers.png"
            )
            exported_files.append(path)
        
        # Export individual asset palettes
        for palette in palettes:
            safe_name = "".join(
                c if c.isalnum() or c in '-_' else '_'
                for c in palette.source_name
            )
            path = self.palette_exporter.export_palette_png(
                palette, f"{safe_name}.png"
            )
            exported_files.append(path)
        
        # Export reports
        self.report_exporter.export_full_report(report_data)
        
        if 'assets' in report_data:
            self.report_exporter.export_summary_csv(report_data['assets'])
        
        if timeline_data:
            self.report_exporter.export_timeline_csv(timeline_data)
        
        # Export blueprint snapshot
        if blueprint_snapshot is not None:
            snapshot_path = os.path.join(project_dir, "summary.png")
            Image.fromarray(blueprint_snapshot).save(snapshot_path)
        
        # Create manifest
        manifest = {
            "project_name": project_name,
            "export_date": str(np.datetime64('today')),
            "contents": {
                "palettes": len(palettes) + 3,  # +3 for consensus, average, outliers
                "report": "data/report.json",
                "summary": "data/summary.csv",
                "timeline": "data/timeline.csv" if timeline_data else None,
                "snapshot": "summary.png" if blueprint_snapshot is not None else None
            }
        }
        
        manifest_path = os.path.join(project_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return project_dir
