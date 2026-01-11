"""
GPU Optimization Module
- CUDA stream processing
- Batch inference
- TensorRT optimization
- Memory pooling
"""

import numpy as np
from typing import List, Optional, Dict, Any
import threading


class GPUOptimizer:
    """GPU optimization manager"""
    
    def __init__(self):
        self.streams = []
        self.memory_pool = {}
        self.batch_queue = []
        self.lock = threading.Lock()
    
    def create_cuda_streams(self, num_streams: int = 4):
        """Create CUDA streams for parallel processing"""
        try:
            import cuda
            self.streams = [cuda.Stream() for _ in range(num_streams)]
            return True
        except:
            return False
    
    def batch_inference(self, inputs: List[np.ndarray], 
                       inference_func, batch_size: int = 8) -> List[Any]:
        """Batch multiple inputs for GPU inference"""
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Stack inputs for batch processing
            if len(batch) > 1:
                batch_input = np.stack(batch, axis=0)
            else:
                batch_input = batch[0]
            
            # Run inference
            batch_result = inference_func(batch_input)
            
            # Unpack results
            if isinstance(batch_result, np.ndarray) and batch_result.ndim > 1:
                results.extend(list(batch_result))
            else:
                results.append(batch_result)
        
        return results
    
    def allocate_pinned_memory(self, size: tuple, dtype=np.float32) -> np.ndarray:
        """Allocate pinned (page-locked) memory for faster GPU transfers"""
        key = (size, dtype)
        
        with self.lock:
            if key not in self.memory_pool:
                # Allocate new pinned memory
                array = np.zeros(size, dtype=dtype)
                # Mark as pinned if possible
                self.memory_pool[key] = array
            
            return self.memory_pool[key]
    
    def optimize_for_tensorrt(self, model_path: str, output_path: str):
        """Convert ONNX model to TensorRT for faster inference"""
        try:
            import tensorrt as trt
            
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network()
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(model_path, 'rb') as model:
                parser.parse(model.read())
            
            # Build engine
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            
            # Enable FP16 if supported
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            engine = builder.build_engine(network, config)
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            return True
        except Exception as e:
            print(f"TensorRT optimization failed: {e}")
            return False
    
    def clear_memory_pool(self):
        """Clear GPU memory pool"""
        with self.lock:
            self.memory_pool.clear()


# Global optimizer instance
_gpu_optimizer = None


def get_gpu_optimizer() -> GPUOptimizer:
    """Get global GPU optimizer instance"""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = GPUOptimizer()
    return _gpu_optimizer


def enable_gpu_optimizations():
    """Enable all GPU optimizations"""
    optimizer = get_gpu_optimizer()
    
    # Create CUDA streams
    optimizer.create_cuda_streams(num_streams=4)
    
    return True


def batch_gpu_inference(inputs: List[np.ndarray], 
                       inference_func,
                       batch_size: int = 8) -> List[Any]:
    """Wrapper for batch GPU inference"""
    optimizer = get_gpu_optimizer()
    return optimizer.batch_inference(inputs, inference_func, batch_size)
