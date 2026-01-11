"""
Optimized Face Processing Pipeline
Enhanced for speed and efficiency
"""

import cv2
import numpy as np
from typing import List, Optional
from functools import lru_cache
import threading

from deeptrace.face_detector import detect_many_faces
from deeptrace.types import Face, VisionFrame

# Thread-local storage for face cache
_thread_local = threading.local()

class OptimizedFaceCache:
    """LRU cache for face detection results to avoid redundant processing"""
    
    def __init__(self, maxsize=100):
        self.cache = {}
        self.maxsize = maxsize
        self.access_count = {}
    
    def get_cache_key(self, frame: VisionFrame) -> str:
        """Generate cache key from frame"""
        return hash(frame.tobytes())
    
    def get(self, frame: VisionFrame) -> Optional[List[Face]]:
        """Get cached faces if available"""
        key = self.get_cache_key(frame)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, frame: VisionFrame, faces: List[Face]) -> None:
        """Cache face detection results"""
        if len(self.cache) >= self.maxsize:
            # Remove least recently used
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        key = self.get_cache_key(frame)
        self.cache[key] = faces
        self.access_count[key] = 0

# Global cache instance
_face_cache = OptimizedFaceCache(maxsize=50)


def preprocess_frame_fast(frame: VisionFrame, target_size: tuple = (640, 640)) -> VisionFrame:
    """
    Fast frame preprocessing with optimizations
    - Resize to optimal detection size
    - Apply minimal necessary preprocessing
    """
    h, w = frame.shape[:2]
    
    # Only resize if necessary
    if (w, h) != target_size:
        # Use INTER_LINEAR for speed (faster than INTER_CUBIC)
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    
    return frame


def detect_faces_optimized(frame: VisionFrame, use_cache: bool = True) -> List[Face]:
    """
    Optimized face detection with caching
    
    Args:
        frame: Input vision frame
        use_cache: Whether to use face detection cache
    
    Returns:
        List of detected faces
    """
    if use_cache:
        # Check cache first
        cached_faces = _face_cache.get(frame)
        if cached_faces is not None:
            return cached_faces
    
    # Preprocess for faster detection
    processed_frame = preprocess_frame_fast(frame)
    
    # Detect faces
    faces = detect_many_faces(processed_frame)
    
    # Cache results
    if use_cache:
        _face_cache.put(frame, faces)
    
    return faces


def batch_process_frames(frames: List[VisionFrame], batch_size: int = 8) -> List[List[Face]]:
    """
    Process multiple frames in batches for better throughput
    
    Args:
        frames: List of vision frames
        batch_size: Number of frames to process together
    
    Returns:
        List of face lists for each frame
    """
    results = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        batch_results = []
        
        for frame in batch:
            faces = detect_faces_optimized(frame)
            batch_results.append(faces)
        
        results.extend(batch_results)
    
    return results


@lru_cache(maxsize=32)
def get_optimal_face_size(model_name: str) -> tuple:
    """Get optimal face size for different models"""
    size_map = {
        'inswapper_128': (128, 128),
        'inswapper_128_fp16': (128, 128),
        'ghost_1_256': (256, 256),
        'ghost_2_256': (256, 256),
        'ghost_3_256': (256, 256),
        'blendswap_256': (256, 256),
        'simswap_256': (256, 256),
        'hyperswap_1a_256': (256, 256),
    }
    return size_map.get(model_name, (256, 256))


def optimize_memory_usage():
    """Clear caches and optimize memory"""
    global _face_cache
    _face_cache.cache.clear()
    _face_cache.access_count.clear()
    
    # Force garbage collection
    import gc
    gc.collect()


class FrameBuffer:
    """Circular buffer for frame processing to reduce memory allocation"""
    
    def __init__(self, size: int = 10, frame_shape: tuple = (1080, 1920, 3)):
        self.size = size
        self.frames = [np.zeros(frame_shape, dtype=np.uint8) for _ in range(size)]
        self.index = 0
    
    def get_frame(self) -> VisionFrame:
        """Get next available frame buffer"""
        frame = self.frames[self.index]
        self.index = (self.index + 1) % self.size
        return frame
    
    def reset(self):
        """Reset buffer"""
        self.index = 0


def parallel_face_detection(frames: List[VisionFrame], num_threads: int = 4) -> List[List[Face]]:
    """
    Parallel face detection across multiple frames
    
    Args:
        frames: List of frames to process
        num_threads: Number of threads to use
    
    Returns:
        List of detected faces for each frame
    """
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(detect_faces_optimized, frames))
    
    return results


# Optimized color space conversions
def bgr_to_rgb_fast(frame: VisionFrame) -> VisionFrame:
    """Fast BGR to RGB conversion"""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr_fast(frame: VisionFrame) -> VisionFrame:
    """Fast RGB to BGR conversion"""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


# Pre-allocate buffers for common operations
class BufferPool:
    """Pool of pre-allocated buffers to reduce memory allocation overhead"""
    
    def __init__(self):
        self.buffers = {}
        self.lock = threading.Lock()
    
    def get_buffer(self, shape: tuple, dtype=np.uint8) -> np.ndarray:
        """Get or create buffer of specified shape"""
        key = (shape, dtype)
        
        with self.lock:
            if key not in self.buffers:
                self.buffers[key] = []
            
            if self.buffers[key]:
                return self.buffers[key].pop()
            else:
                return np.zeros(shape, dtype=dtype)
    
    def return_buffer(self, buffer: np.ndarray):
        """Return buffer to pool"""
        key = (buffer.shape, buffer.dtype)
        
        with self.lock:
            if key not in self.buffers:
                self.buffers[key] = []
            
            # Limit pool size
            if len(self.buffers[key]) < 10:
                self.buffers[key].append(buffer)

# Global buffer pool
_buffer_pool = BufferPool()
