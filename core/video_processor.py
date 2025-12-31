"""
Color Cohesion Analyzer - Video Processing Module
Shot detection, frame sampling, and timeline analysis
"""

import cv2
import numpy as np
from typing import List, Optional, Generator, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

from .color_space import ColorConverter, delta_e_76
from .palette_extraction import Palette, PaletteExtractor
from config import ANALYSIS_CONFIG, SamplingMode


@dataclass
class Shot:
    """Represents a detected shot/scene in a video"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    palette: Optional[Palette] = None
    thumbnail: Optional[np.ndarray] = None
    
    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.duration_seconds, 3),
            "palette": self.palette.to_dict() if self.palette else None
        }


@dataclass
class VideoInfo:
    """Video metadata"""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str = ""
    
    @property
    def filename(self) -> str:
        return Path(self.path).name
    
    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "filename": self.filename,
            "resolution": f"{self.width}x{self.height}",
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration": round(self.duration, 2),
            "codec": self.codec
        }


@dataclass
class VideoAnalysis:
    """Complete video analysis results"""
    info: VideoInfo
    shots: List[Shot] = field(default_factory=list)
    global_palette: Optional[Palette] = None
    timeline_palettes: List[Tuple[float, Palette]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "info": self.info.to_dict(),
            "shots": [s.to_dict() for s in self.shots],
            "global_palette": self.global_palette.to_dict() if self.global_palette else None,
            "timeline": [
                {"time": round(t, 3), "palette": p.to_dict()}
                for t, p in self.timeline_palettes
            ]
        }


class VideoProcessor:
    """
    Video processing engine with shot detection and palette extraction
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.sample_interval = ANALYSIS_CONFIG.sample_interval
        self.shot_threshold = ANALYSIS_CONFIG.shot_detection_threshold
        self.min_shot_duration = ANALYSIS_CONFIG.min_shot_duration_frames
        self.max_dimension = ANALYSIS_CONFIG.max_dimension
        
        self.extractor = PaletteExtractor()
        
        # State for pause/stop
        self._paused = False
        self._stopped = False
        self._progress_callback = None
        
        if config:
            self.__dict__.update(config)
    
    def set_progress_callback(self, callback):
        """Set callback for progress updates: callback(progress: float, message: str)"""
        self._progress_callback = callback
    
    def pause(self):
        """Pause processing"""
        self._paused = True
    
    def resume(self):
        """Resume processing"""
        self._paused = False
    
    def stop(self):
        """Stop processing"""
        self._stopped = True
    
    def reset_state(self):
        """Reset pause/stop state"""
        self._paused = False
        self._stopped = False
    
    def _report_progress(self, progress: float, message: str):
        """Report progress to callback if set"""
        if self._progress_callback:
            self._progress_callback(progress, message)
    
    def _wait_if_paused(self):
        """Block while paused, return False if stopped"""
        while self._paused and not self._stopped:
            import time
            time.sleep(0.1)
        return not self._stopped
    
    def get_video_info(self, video_path: str) -> Optional[VideoInfo]:
        """Get video metadata"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            duration = total_frames / fps if fps > 0 else 0
            
            return VideoInfo(
                path=video_path,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration=duration,
                codec=codec
            )
        finally:
            cap.release()
    
    def _downscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Downscale frame for processing efficiency"""
        h, w = frame.shape[:2]
        
        if max(h, w) <= self.max_dimension:
            return frame
        
        scale = self.max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _compute_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute perceptual difference between frames
        Used for shot detection
        """
        # Downscale for speed
        small1 = cv2.resize(frame1, (64, 64))
        small2 = cv2.resize(frame2, (64, 64))
        
        # Convert to Lab
        lab1 = cv2.cvtColor(small1, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(small2, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Compute histogram difference
        diff = np.abs(lab1 - lab2)
        
        # Weight by luminance and chroma
        l_diff = np.mean(diff[:, :, 0])
        ab_diff = np.mean(diff[:, :, 1:])
        
        return l_diff + ab_diff * 0.5
    
    def detect_shots(self, video_path: str) -> List[Shot]:
        """
        Detect shot boundaries using frame differences
        """
        self.reset_state()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            shots = []
            shot_start = 0
            prev_frame = None
            frame_idx = 0
            
            self._report_progress(0, "Detecting shots...")
            
            while True:
                if not self._wait_if_paused():
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    diff = self._compute_frame_difference(prev_frame, frame)
                    
                    # Shot boundary detected
                    if diff > self.shot_threshold:
                        shot_duration = frame_idx - shot_start
                        
                        if shot_duration >= self.min_shot_duration:
                            shots.append(Shot(
                                start_frame=shot_start,
                                end_frame=frame_idx - 1,
                                start_time=shot_start / fps,
                                end_time=(frame_idx - 1) / fps
                            ))
                        
                        shot_start = frame_idx
                
                prev_frame = frame.copy()
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    progress = frame_idx / total_frames
                    self._report_progress(progress, f"Detecting shots: {len(shots)} found")
            
            # Add final shot
            if frame_idx - shot_start >= self.min_shot_duration:
                shots.append(Shot(
                    start_frame=shot_start,
                    end_frame=frame_idx - 1,
                    start_time=shot_start / fps,
                    end_time=(frame_idx - 1) / fps
                ))
            
            return shots
            
        finally:
            cap.release()
    
    def sample_frames(
        self,
        video_path: str,
        start_frame: int = 0,
        end_frame: int = -1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields sampled frames from video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if end_frame < 0:
                end_frame = total_frames
            
            frame_idx = start_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            while frame_idx < end_frame:
                if not self._wait_if_paused():
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                if (frame_idx - start_frame) % self.sample_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield frame_idx, self._downscale_frame(frame_rgb)
                
                frame_idx += 1
        finally:
            cap.release()
    
    def extract_shot_palette(
        self,
        video_path: str,
        shot: Shot,
        video_name: str = ""
    ) -> Palette:
        """Extract palette for a single shot"""
        all_pixels = []
        
        for frame_idx, frame in self.sample_frames(video_path, shot.start_frame, shot.end_frame):
            pixels = frame.reshape(-1, 3)
            # Subsample pixels
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]
            all_pixels.append(pixels)
        
        if not all_pixels:
            return Palette([], f"{video_name}_shot_{shot.start_frame}", "shot")
        
        combined_pixels = np.vstack(all_pixels)
        
        # Subsample if too many
        if len(combined_pixels) > 50000:
            indices = np.random.choice(len(combined_pixels), 50000, replace=False)
            combined_pixels = combined_pixels[indices]
        
        return self.extractor.extract_from_pixels(
            combined_pixels,
            f"{video_name}_shot_{shot.start_frame}",
            "shot"
        )
    
    def analyze_video(self, video_path: str, fast_mode: bool = True) -> Optional[VideoAnalysis]:
        """
        Complete video analysis:
        
        Args:
            video_path: Path to video file
            fast_mode: If True, skip shot detection and extract single global palette
                      If False, detect shots and extract per-shot palettes
        
        Returns:
            VideoAnalysis object with results
        """
        self.reset_state()
        
        # Get video info
        info = self.get_video_info(video_path)
        if not info:
            return None
        
        video_name = Path(video_path).stem
        
        if fast_mode:
            # Fast mode: Just extract global palette from sampled frames
            return self._analyze_video_fast(video_path, info, video_name)
        else:
            # Full mode: Shot detection + per-shot palettes
            return self._analyze_video_full(video_path, info, video_name)
    
    def _analyze_video_fast(self, video_path: str, info: VideoInfo, video_name: str) -> Optional[VideoAnalysis]:
        """Fast video analysis - single global palette only"""
        self._report_progress(0.1, "Sampling frames...")
        
        all_pixels = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        try:
            total_frames = info.total_frames
            # Sample ~30-50 frames evenly distributed across video
            target_samples = min(50, max(30, total_frames // 100))
            sample_step = max(1, total_frames // target_samples)
            
            frame_idx = 0
            samples_collected = 0
            
            while frame_idx < total_frames:
                if self._stopped:
                    return None
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert and downscale
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_small = self._downscale_frame(frame_rgb)
                
                # Sample pixels from frame
                pixels = frame_small.reshape(-1, 3)
                if len(pixels) > 3000:
                    indices = np.random.choice(len(pixels), 3000, replace=False)
                    pixels = pixels[indices]
                all_pixels.append(pixels)
                
                samples_collected += 1
                frame_idx += sample_step
                
                progress = 0.1 + 0.7 * (frame_idx / total_frames)
                self._report_progress(progress, f"Sampled {samples_collected} frames")
            
        finally:
            cap.release()
        
        if self._stopped:
            return None
        
        self._report_progress(0.85, "Computing palette...")
        
        # Compute global palette
        if all_pixels:
            combined_pixels = np.vstack(all_pixels)
            if len(combined_pixels) > 80000:
                indices = np.random.choice(len(combined_pixels), 80000, replace=False)
                combined_pixels = combined_pixels[indices]
            
            global_palette = self.extractor.extract_from_pixels(
                combined_pixels,
                video_name,
                "video"
            )
        else:
            global_palette = Palette([], video_name, "video")
        
        self._report_progress(1.0, "Video analysis complete")
        
        # Return with empty shots list (fast mode doesn't detect shots)
        return VideoAnalysis(
            info=info,
            shots=[],
            global_palette=global_palette,
            timeline_palettes=[]
        )
    
    def _analyze_video_full(self, video_path: str, info: VideoInfo, video_name: str) -> Optional[VideoAnalysis]:
        """Full video analysis with shot detection"""
        self._report_progress(0.1, "Detecting shots...")
        
        # Detect shots
        shots = self.detect_shots(video_path)
        
        if self._stopped:
            return None
        
        # If no shots detected, treat as single shot
        if not shots:
            shots = [Shot(
                start_frame=0,
                end_frame=info.total_frames - 1,
                start_time=0,
                end_time=info.duration
            )]
        
        self._report_progress(0.3, f"Extracting palettes for {len(shots)} shots...")
        
        # Extract palette per shot
        all_pixels = []
        timeline_palettes = []
        
        for i, shot in enumerate(shots):
            if self._stopped:
                return None
            
            shot.palette = self.extract_shot_palette(video_path, shot, video_name)
            
            # Collect pixels for global palette
            for frame_idx, frame in self.sample_frames(video_path, shot.start_frame, shot.end_frame):
                pixels = frame.reshape(-1, 3)
                if len(pixels) > 5000:
                    indices = np.random.choice(len(pixels), 5000, replace=False)
                    pixels = pixels[indices]
                all_pixels.append(pixels)
            
            # Add to timeline
            timeline_palettes.append((shot.start_time, shot.palette))
            
            progress = 0.3 + 0.5 * (i + 1) / len(shots)
            self._report_progress(progress, f"Processed shot {i + 1}/{len(shots)}")
        
        self._report_progress(0.9, "Computing global palette...")
        
        # Compute global palette
        if all_pixels:
            combined_pixels = np.vstack(all_pixels)
            if len(combined_pixels) > 100000:
                indices = np.random.choice(len(combined_pixels), 100000, replace=False)
                combined_pixels = combined_pixels[indices]
            
            global_palette = self.extractor.extract_from_pixels(
                combined_pixels,
                video_name,
                "video"
            )
        else:
            global_palette = Palette([], video_name, "video")
        
        # Get thumbnails for shots
        cap = cv2.VideoCapture(video_path)
        for shot in shots:
            cap.set(cv2.CAP_PROP_POS_FRAMES, shot.start_frame)
            ret, frame = cap.read()
            if ret:
                thumbnail = cv2.resize(frame, (160, 90))
                shot.thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
        cap.release()
        
        self._report_progress(1.0, "Video analysis complete")
        
        return VideoAnalysis(
            info=info,
            shots=shots,
            global_palette=global_palette,
            timeline_palettes=timeline_palettes
        )
    
    def get_timeline_strip(
        self,
        analysis: VideoAnalysis,
        width: int = 800,
        height: int = 50
    ) -> np.ndarray:
        """
        Generate a visual timeline strip showing palette evolution
        """
        strip = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not analysis.timeline_palettes or analysis.info.duration == 0:
            return strip
        
        duration = analysis.info.duration
        
        for i, (time, palette) in enumerate(analysis.timeline_palettes):
            # Calculate x range for this palette
            x_start = int((time / duration) * width)
            
            if i + 1 < len(analysis.timeline_palettes):
                x_end = int((analysis.timeline_palettes[i + 1][0] / duration) * width)
            else:
                x_end = width
            
            if not palette.colors:
                continue
            
            # Draw palette colors vertically
            n_colors = len(palette.colors)
            for j, color in enumerate(palette.colors):
                y_start = int((j / n_colors) * height)
                y_end = int(((j + 1) / n_colors) * height)
                
                strip[y_start:y_end, x_start:x_end] = color.rgb
        
        return strip


def compute_file_hash(file_path: str) -> str:
    """Compute hash of file for caching"""
    hasher = hashlib.md5()
    
    with open(file_path, 'rb') as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    
    return hasher.hexdigest()
