"""
TensorRT optimization configuration for Jetson TX2
"""

import os
import numpy as np
from typing import Dict, Any, Optional

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: TensorRT not available")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    print("Warning: PyCUDA not available")


class JetsonTensorRTOptimizer:
    """TensorRT optimization specifically for Jetson TX2"""
    
    def __init__(self):
        self.logger = None
        self.engine = None
        self.context = None
        
        if TENSORRT_AVAILABLE:
            self.logger = trt.Logger(trt.Logger.WARNING)
        
    def get_jetson_tx2_config(self) -> Dict[str, Any]:
        """Get optimal TensorRT configuration for Jetson TX2"""
        return {
            # Precision settings
            'precision': 'fp16',  # FP16 provides good balance of speed/accuracy
            'int8_calibration': False,  # Skip INT8 for simplicity
            
            # Memory settings (TX2 has limited memory)
            'max_workspace_size': 1 << 28,  # 256MB workspace
            'max_batch_size': 1,  # Single inference for real-time
            
            # Optimization settings
            'builder_optimization_level': 5,  # Max optimization
            'enable_dla': False,  # TX2 doesn't have DLA
            'strict_type_constraints': True,
            'sparse_weights': False,  # TX2 doesn't support sparse
            
            # Platform specific
            'target_platform': 'jetson_tx2',
            'gpu_fallback': True,  # Allow GPU fallback
            'prefer_precision_constraints': False,
            
            # Performance tuning
            'engine_capability': trt.EngineCapability.STANDARD,
            'profiling_verbosity': trt.ProfilingVerbosity.DETAILED,
        }
    
    def build_engine_from_onnx(self, onnx_path: str, 
                              engine_path: str,
                              config: Optional[Dict] = None) -> bool:
        """
        Build TensorRT engine from ONNX model
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Output path for TensorRT engine
            config: Custom configuration (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not TENSORRT_AVAILABLE:
            print("TensorRT not available")
            return False
        
        if not os.path.exists(onnx_path):
            print(f"ONNX model not found: {onnx_path}")
            return False
        
        # Use default config if none provided
        if config is None:
            config = self.get_jetson_tx2_config()
        
        print(f"Building TensorRT engine for Jetson TX2...")
        print(f"Input ONNX: {onnx_path}")
        print(f"Output Engine: {engine_path}")
        print(f"Precision: {config['precision']}")
        
        try:
            # Create builder and config
            builder = trt.Builder(self.logger)
            config_obj = builder.create_builder_config()
            
            # Set workspace size
            config_obj.max_workspace_size = config['max_workspace_size']
            
            # Set precision
            if config['precision'] == 'fp16':
                config_obj.set_flag(trt.BuilderFlag.FP16)
            elif config['precision'] == 'int8':
                config_obj.set_flag(trt.BuilderFlag.INT8)
                # INT8 calibration would be needed here
            
            # Set optimization level
            if hasattr(config_obj, 'builder_optimization_level'):
                config_obj.builder_optimization_level = config['builder_optimization_level']
            
            # Create network
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            
            # Parse ONNX model
            parser = trt.OnnxParser(network, self.logger)
            
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return False
            
            # Set optimization profile for dynamic shapes
            profile = builder.create_optimization_profile()
            
            # Assuming input shape is [batch, 3, height, width]
            input_tensor = network.get_input(0)
            input_shape = input_tensor.shape
            
            print(f"Input tensor shape: {input_shape}")
            
            # Set dynamic shape range (for batch size flexibility)
            if input_shape[0] == -1:  # Dynamic batch size
                min_shape = (1, input_shape[1], input_shape[2], input_shape[3])
                opt_shape = (1, input_shape[1], input_shape[2], input_shape[3])
                max_shape = (config['max_batch_size'], input_shape[1], input_shape[2], input_shape[3])
                profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            
            config_obj.add_optimization_profile(profile)
            
            # Build engine
            print("Building TensorRT engine... This may take several minutes.")
            engine = builder.build_engine(network, config_obj)
            
            if engine is None:
                print("Failed to build TensorRT engine")
                return False
            
            # Serialize and save engine
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            print(f"TensorRT engine built successfully: {engine_path}")
            
            # Print engine info
            self.print_engine_info(engine_path)
            
            return True
            
        except Exception as e:
            print(f"Error building TensorRT engine: {e}")
            return False
    
    def load_engine(self, engine_path: str) -> bool:
        """Load TensorRT engine from file"""
        if not TENSORRT_AVAILABLE:
            return False
        
        if not os.path.exists(engine_path):
            print(f"Engine file not found: {engine_path}")
            return False
        
        try:
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            if self.engine is None:
                print("Failed to load TensorRT engine")
                return False
            
            self.context = self.engine.create_execution_context()
            print(f"TensorRT engine loaded successfully: {engine_path}")
            return True
            
        except Exception as e:
            print(f"Error loading TensorRT engine: {e}")
            return False
    
    def print_engine_info(self, engine_path: str):
        """Print information about the TensorRT engine"""
        if not os.path.exists(engine_path):
            return
        
        file_size = os.path.getsize(engine_path) / (1024 * 1024)  # MB
        print(f"\nTensorRT Engine Information:")
        print(f"  File size: {file_size:.1f} MB")
        
        # Load engine to get more details
        try:
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(self.logger)
                engine = runtime.deserialize_cuda_engine(f.read())
            
            if engine:
                print(f"  Number of bindings: {engine.num_bindings}")
                print(f"  Max batch size: {engine.max_batch_size}")
                
                for i in range(engine.num_bindings):
                    name = engine.get_binding_name(i)
                    shape = engine.get_binding_shape(i)
                    dtype = engine.get_binding_dtype(i)
                    is_input = engine.binding_is_input(i)
                    print(f"  {'Input' if is_input else 'Output'} {i}: {name} - {shape} ({dtype})")
                    
        except Exception as e:
            print(f"  Could not read engine details: {e}")
    
    def benchmark_engine(self, engine_path: str, input_shape: tuple = (1, 3, 640, 640),
                        num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark TensorRT engine performance
        
        Args:
            engine_path: Path to TensorRT engine
            input_shape: Input tensor shape
            num_runs: Number of inference runs
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.load_engine(engine_path):
            return {}
        
        if not PYCUDA_AVAILABLE:
            print("PyCUDA not available for benchmarking")
            return {}
        
        try:
            import time
            
            # Allocate GPU memory
            input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
            output_binding = 1  # Assuming single output
            output_shape = self.engine.get_binding_shape(output_binding)
            output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize
            
            # Allocate GPU memory
            d_input = cuda.mem_alloc(input_size)
            d_output = cuda.mem_alloc(output_size)
            
            # Create dummy input data
            h_input = np.random.rand(*input_shape).astype(np.float32)
            h_output = np.empty(output_shape, dtype=np.float32)
            
            # Copy input to GPU
            cuda.memcpy_htod(d_input, h_input)
            
            # Warm up
            for _ in range(10):
                self.context.execute_v2([int(d_input), int(d_output)])
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                self.context.execute_v2([int(d_input), int(d_output)])
                cuda.Context.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            times = np.array(times)
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            avg_fps = 1.0 / avg_time
            
            results = {
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'min_time_ms': min_time * 1000,
                'max_time_ms': max_time * 1000,
                'avg_fps': avg_fps,
                'input_shape': input_shape,
                'num_runs': num_runs
            }
            
            print(f"\nTensorRT Benchmark Results:")
            print(f"  Average inference time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
            print(f"  Min/Max inference time: {min_time*1000:.2f}ms / {max_time*1000:.2f}ms")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Input shape: {input_shape}")
            
            return results
            
        except Exception as e:
            print(f"Benchmark error: {e}")
            return {}


def main():
    """Demo TensorRT optimization for Jetson TX2"""
    optimizer = JetsonTensorRTOptimizer()
    
    # Print configuration
    config = optimizer.get_jetson_tx2_config()
    print("Jetson TX2 TensorRT Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nTo use this optimizer:")
    print("1. Export your model to ONNX format")
    print("2. Use build_engine_from_onnx() to create TensorRT engine")
    print("3. Use benchmark_engine() to test performance")
    print("4. Deploy the engine with inference scripts")


if __name__ == "__main__":
    main()