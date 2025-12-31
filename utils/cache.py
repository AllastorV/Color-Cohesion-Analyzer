"""
Color Cohesion Analyzer - Caching System
File hash-based caching to avoid redundant processing
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import pickle

try:
    import xxhash
    USE_XXHASH = True
except ImportError:
    USE_XXHASH = False

try:
    from diskcache import Cache
    USE_DISKCACHE = True
except ImportError:
    USE_DISKCACHE = False


def compute_file_hash(file_path: str, quick: bool = False) -> str:
    """
    Compute hash of a file for caching
    
    Args:
        file_path: Path to file
        quick: If True, only hash first/last chunks (faster for large files)
    """
    file_size = os.path.getsize(file_path)
    
    if USE_XXHASH:
        hasher = xxhash.xxh64()
    else:
        hasher = hashlib.md5()
    
    # Include file metadata in hash
    stat = os.stat(file_path)
    hasher.update(str(stat.st_size).encode())
    hasher.update(str(stat.st_mtime).encode())
    
    if quick and file_size > 10_000_000:  # > 10MB
        # Hash first and last 1MB
        with open(file_path, 'rb') as f:
            hasher.update(f.read(1_000_000))
            f.seek(-1_000_000, 2)
            hasher.update(f.read())
    else:
        # Full file hash
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
    
    return hasher.hexdigest()


def compute_params_hash(params: Dict) -> str:
    """Compute hash of analysis parameters"""
    params_str = json.dumps(params, sort_keys=True)
    
    if USE_XXHASH:
        return xxhash.xxh64(params_str.encode()).hexdigest()
    return hashlib.md5(params_str.encode()).hexdigest()


@dataclass
class CacheEntry:
    """Cached analysis entry"""
    file_hash: str
    params_hash: str
    palette_data: Dict
    metrics_data: Optional[Dict] = None
    video_data: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        return cls(**data)


class AnalysisCache:
    """
    Cache for analysis results
    Uses disk cache for persistence
    """
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.expanduser("~"),
                ".color_cohesion_analyzer",
                "cache"
            )
        
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        if USE_DISKCACHE:
            self._cache = Cache(cache_dir)
        else:
            self._cache = None
            self._memory_cache: Dict[str, CacheEntry] = {}
    
    def _get_cache_key(self, file_path: str, params: Dict) -> str:
        """Generate cache key from file and params"""
        file_hash = compute_file_hash(file_path, quick=True)
        params_hash = compute_params_hash(params)
        return f"{file_hash}_{params_hash}"
    
    def get(self, file_path: str, params: Dict) -> Optional[CacheEntry]:
        """Get cached entry if exists"""
        key = self._get_cache_key(file_path, params)
        
        if USE_DISKCACHE and self._cache:
            data = self._cache.get(key)
            if data:
                return CacheEntry.from_dict(data)
        elif key in self._memory_cache:
            return self._memory_cache[key]
        
        return None
    
    def set(self, file_path: str, params: Dict, entry: CacheEntry):
        """Store entry in cache"""
        key = self._get_cache_key(file_path, params)
        
        if USE_DISKCACHE and self._cache:
            self._cache.set(key, entry.to_dict())
        else:
            self._memory_cache[key] = entry
    
    def has(self, file_path: str, params: Dict) -> bool:
        """Check if entry exists in cache"""
        key = self._get_cache_key(file_path, params)
        
        if USE_DISKCACHE and self._cache:
            return key in self._cache
        return key in self._memory_cache
    
    def clear(self):
        """Clear the cache"""
        if USE_DISKCACHE and self._cache:
            self._cache.clear()
        else:
            self._memory_cache.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        if USE_DISKCACHE and self._cache:
            return {
                "size": len(self._cache),
                "volume": self._cache.volume(),
            }
        return {
            "size": len(self._memory_cache),
            "volume": 0,
        }


class GPUManager:
    """
    Manage GPU acceleration
    Detects CUDA availability and provides fallback
    """
    
    def __init__(self):
        self._cuda_available = False
        self._gpu_name = "CPU"
        self._check_cuda()
    
    def _check_cuda(self):
        """Check for CUDA availability"""
        try:
            import cupy as cp
            self._cuda_available = True
            
            # Get GPU name
            device = cp.cuda.Device()
            self._gpu_name = device.name
            
        except ImportError:
            self._cuda_available = False
        except Exception:
            self._cuda_available = False
    
    @property
    def cuda_available(self) -> bool:
        return self._cuda_available
    
    @property
    def gpu_name(self) -> str:
        return self._gpu_name
    
    def get_array_module(self):
        """Get numpy or cupy depending on availability"""
        if self._cuda_available:
            try:
                import cupy as cp
                return cp
            except:
                pass
        
        import numpy as np
        return np
    
    def to_gpu(self, array):
        """Transfer array to GPU if available"""
        if self._cuda_available:
            try:
                import cupy as cp
                return cp.asarray(array)
            except:
                pass
        return array
    
    def to_cpu(self, array):
        """Transfer array to CPU"""
        if hasattr(array, 'get'):
            return array.get()
        return array
    
    def get_status_string(self) -> str:
        """Get human-readable GPU status"""
        if self._cuda_available:
            return f"GPU: {self._gpu_name}"
        return "CPU (GPU not available)"


# Global instances
_cache: Optional[AnalysisCache] = None
_gpu_manager: Optional[GPUManager] = None


def get_cache() -> AnalysisCache:
    """Get global cache instance"""
    global _cache
    if _cache is None:
        _cache = AnalysisCache()
    return _cache


def get_gpu_manager() -> GPUManager:
    """Get global GPU manager instance"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager
