"""
Color Cohesion Analyzer - Color Space Conversions
Perceptual color processing using OKLab and CIELAB color spaces
"""

import numpy as np
from typing import Tuple, Union
from dataclasses import dataclass

@dataclass
class PaletteColor:
    """Represents a single color in a palette"""
    hex_code: str
    rgb: Tuple[int, int, int]
    lab: Tuple[float, float, float]  # CIELAB or OKLab
    dominance: float  # 0.0 to 1.0
    
    def to_dict(self) -> dict:
        return {
            "hex": self.hex_code,
            "rgb": list(self.rgb),
            "lab": list(self.lab),
            "dominance": round(self.dominance * 100, 2)
        }

class ColorConverter:
    """
    High-precision color space conversions
    Supports sRGB, Linear RGB, XYZ, CIELAB, and OKLab
    """
    
    # D65 white point
    WHITE_POINT_D65 = np.array([0.95047, 1.0, 1.08883])
    
    # sRGB to XYZ matrix
    SRGB_TO_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ)
    
    # OKLab matrices
    M1 = np.array([
        [0.8189330101, 0.3618667424, -0.1288597137],
        [0.0329845436, 0.9293118715, 0.0361456387],
        [0.0482003018, 0.2643662691, 0.6338517070]
    ])
    
    M2 = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ])
    
    M1_INV = np.linalg.inv(M1)
    M2_INV = np.linalg.inv(M2)
    
    @staticmethod
    def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
        """Convert sRGB (0-1) to linear RGB"""
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            np.power((srgb + 0.055) / 1.055, 2.4)
        )
        return linear
    
    @staticmethod
    def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
        """Convert linear RGB to sRGB (0-1)"""
        srgb = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(np.maximum(linear, 0), 1/2.4) - 0.055
        )
        return np.clip(srgb, 0, 1)
    
    @classmethod
    def rgb_to_xyz(cls, rgb: np.ndarray) -> np.ndarray:
        """Convert sRGB (0-255) to XYZ"""
        rgb_normalized = rgb.astype(np.float64) / 255.0
        linear = cls.srgb_to_linear(rgb_normalized)
        
        if linear.ndim == 1:
            return cls.SRGB_TO_XYZ @ linear
        else:
            return linear @ cls.SRGB_TO_XYZ.T
    
    @classmethod
    def xyz_to_rgb(cls, xyz: np.ndarray) -> np.ndarray:
        """Convert XYZ to sRGB (0-255)"""
        if xyz.ndim == 1:
            linear = cls.XYZ_TO_SRGB @ xyz
        else:
            linear = xyz @ cls.XYZ_TO_SRGB.T
        
        srgb = cls.linear_to_srgb(linear)
        return (srgb * 255).astype(np.uint8)
    
    @classmethod
    def xyz_to_lab(cls, xyz: np.ndarray) -> np.ndarray:
        """Convert XYZ to CIELAB"""
        xyz_normalized = xyz / cls.WHITE_POINT_D65
        
        def f(t):
            delta = 6/29
            return np.where(
                t > delta**3,
                np.cbrt(t),
                t / (3 * delta**2) + 4/29
            )
        
        fx = f(xyz_normalized[..., 0])
        fy = f(xyz_normalized[..., 1])
        fz = f(xyz_normalized[..., 2])
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return np.stack([L, a, b], axis=-1)
    
    @classmethod
    def lab_to_xyz(cls, lab: np.ndarray) -> np.ndarray:
        """Convert CIELAB to XYZ"""
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        def f_inv(t):
            delta = 6/29
            return np.where(
                t > delta,
                t**3,
                3 * delta**2 * (t - 4/29)
            )
        
        x = cls.WHITE_POINT_D65[0] * f_inv(fx)
        y = cls.WHITE_POINT_D65[1] * f_inv(fy)
        z = cls.WHITE_POINT_D65[2] * f_inv(fz)
        
        return np.stack([x, y, z], axis=-1)
    
    @classmethod
    def rgb_to_lab(cls, rgb: np.ndarray) -> np.ndarray:
        """Convert sRGB (0-255) to CIELAB"""
        xyz = cls.rgb_to_xyz(rgb)
        return cls.xyz_to_lab(xyz)
    
    @classmethod
    def lab_to_rgb(cls, lab: np.ndarray) -> np.ndarray:
        """Convert CIELAB to sRGB (0-255)"""
        xyz = cls.lab_to_xyz(lab)
        return cls.xyz_to_rgb(xyz)
    
    @classmethod
    def rgb_to_oklab(cls, rgb: np.ndarray) -> np.ndarray:
        """Convert sRGB (0-255) to OKLab"""
        rgb_normalized = rgb.astype(np.float64) / 255.0
        linear = cls.srgb_to_linear(rgb_normalized)
        
        if linear.ndim == 1:
            lms = cls.M1 @ linear
        else:
            lms = linear @ cls.M1.T
        
        # Cube root
        lms_prime = np.cbrt(np.maximum(lms, 0))
        
        if lms_prime.ndim == 1:
            lab = cls.M2 @ lms_prime
        else:
            lab = lms_prime @ cls.M2.T
        
        return lab
    
    @classmethod
    def oklab_to_rgb(cls, lab: np.ndarray) -> np.ndarray:
        """Convert OKLab to sRGB (0-255)"""
        if lab.ndim == 1:
            lms_prime = cls.M2_INV @ lab
        else:
            lms_prime = lab @ cls.M2_INV.T
        
        # Cube
        lms = lms_prime ** 3
        
        if lms.ndim == 1:
            linear = cls.M1_INV @ lms
        else:
            linear = lms @ cls.M1_INV.T
        
        srgb = cls.linear_to_srgb(linear)
        return (np.clip(srgb, 0, 1) * 255).astype(np.uint8)
    
    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex string"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    @staticmethod
    def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
        """Convert hex string to RGB"""
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    Calculate CIE76 Delta E (Euclidean distance in Lab space)
    Simple but effective for perceptual difference
    """
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray, kL: float = 1, kC: float = 1, kH: float = 1) -> float:
    """
    Calculate CIEDE2000 Delta E
    More accurate perceptual difference metric
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    
    # Calculate C and h
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2
    
    G = 0.5 * (1 - np.sqrt(C_avg**7 / (C_avg**7 + 25**7)))
    
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
    
    # Delta values
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime
    
    dh_prime = np.where(
        C1_prime * C2_prime == 0, 0,
        np.where(
            np.abs(h2_prime - h1_prime) <= 180, h2_prime - h1_prime,
            np.where(h2_prime - h1_prime > 180, h2_prime - h1_prime - 360, h2_prime - h1_prime + 360)
        )
    )
    
    dH_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime / 2))
    
    # Averages
    L_avg_prime = (L1 + L2) / 2
    C_avg_prime = (C1_prime + C2_prime) / 2
    
    h_avg_prime = np.where(
        C1_prime * C2_prime == 0, h1_prime + h2_prime,
        np.where(
            np.abs(h1_prime - h2_prime) <= 180, (h1_prime + h2_prime) / 2,
            np.where(h1_prime + h2_prime < 360, (h1_prime + h2_prime + 360) / 2, (h1_prime + h2_prime - 360) / 2)
        )
    )
    
    T = (1 - 0.17 * np.cos(np.radians(h_avg_prime - 30)) +
         0.24 * np.cos(np.radians(2 * h_avg_prime)) +
         0.32 * np.cos(np.radians(3 * h_avg_prime + 6)) -
         0.20 * np.cos(np.radians(4 * h_avg_prime - 63)))
    
    dTheta = 30 * np.exp(-((h_avg_prime - 275) / 25)**2)
    RC = 2 * np.sqrt(C_avg_prime**7 / (C_avg_prime**7 + 25**7))
    SL = 1 + (0.015 * (L_avg_prime - 50)**2) / np.sqrt(20 + (L_avg_prime - 50)**2)
    SC = 1 + 0.045 * C_avg_prime
    SH = 1 + 0.015 * C_avg_prime * T
    RT = -np.sin(np.radians(2 * dTheta)) * RC
    
    dE = np.sqrt(
        (dL_prime / (kL * SL))**2 +
        (dC_prime / (kC * SC))**2 +
        (dH_prime / (kH * SH))**2 +
        RT * (dC_prime / (kC * SC)) * (dH_prime / (kH * SH))
    )
    
    return dE


def get_warm_cool_balance(lab_colors: np.ndarray) -> float:
    """
    Calculate warm/cool balance from -1 (cool) to 1 (warm)
    Based on b* channel in Lab space
    """
    b_values = lab_colors[..., 2]
    avg_b = np.mean(b_values)
    
    # Normalize to -1 to 1 range (b typically ranges -128 to 127 in CIELAB)
    normalized = np.clip(avg_b / 50, -1, 1)
    return float(normalized)


def get_saturation_distribution(lab_colors: np.ndarray) -> dict:
    """
    Analyze saturation distribution in Lab space
    Returns stats about chroma values
    """
    chroma = np.sqrt(lab_colors[..., 1]**2 + lab_colors[..., 2]**2)
    
    return {
        "mean": float(np.mean(chroma)),
        "std": float(np.std(chroma)),
        "min": float(np.min(chroma)),
        "max": float(np.max(chroma)),
        "low_sat_ratio": float(np.mean(chroma < 20)),  # Desaturated
        "high_sat_ratio": float(np.mean(chroma > 60))  # Saturated
    }


def calculate_palette_entropy(palette_colors: list) -> float:
    """
    Calculate color entropy (complexity/dispersion) of a palette
    Higher values = more diverse palette
    """
    if not palette_colors:
        return 0.0
    
    dominances = [c.dominance for c in palette_colors]
    total = sum(dominances)
    
    if total == 0:
        return 0.0
    
    probabilities = [d / total for d in dominances]
    entropy = -sum(p * np.log2(p + 1e-10) for p in probabilities if p > 0)
    
    # Normalize by max possible entropy
    max_entropy = np.log2(len(palette_colors))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


# Skin tone corridor for human subject analysis
SKIN_TONE_LAB_CENTER = np.array([65, 15, 20])  # Approximate center
SKIN_TONE_RADIUS = 25  # DeltaE threshold


def detect_skin_tone_deviation(lab_colors: np.ndarray) -> float:
    """
    Detect deviation from typical skin tone corridor
    Returns average distance from skin tone center for colors in that region
    """
    distances = delta_e_76(lab_colors, SKIN_TONE_LAB_CENTER)
    
    # Find colors that might be skin tones (within broader range)
    potential_skin = distances < SKIN_TONE_RADIUS * 2
    
    if not np.any(potential_skin):
        return 0.0  # No potential skin tones detected
    
    skin_distances = distances[potential_skin]
    return float(np.mean(skin_distances))
