"""
Model export utilities for deployment
Supports ONNX, TensorRT, and other deployment formats
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Please install: pip install ultralytics")

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx not available for ONNX export")

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: tensorrt not available for TensorRT export")


class ModelExporter:
    """Handles model export for various deployment formats"""
    
    def __init__(self, model_path: str):
        """
        Initialize model exporter
        
        Args:
            model_path: Path to trained PyTorch model
        """
        self.model_path = model_path
        self.model = None
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.load_model()
    
    def load_model(self):
        """Load the PyTorch model"""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package is required for export")
        
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def export_onnx(self, output_path: str = None, imgsz: int = 640, 
                   opset: int = 11, simplify: bool = True) -> str:
        """
        Export model to ONNX format
        
        Args:
            output_path: Output path for ONNX model
            imgsz: Input image size
            opset: ONNX opset version
            simplify: Whether to simplify the ONNX model
            
        Returns:
            Path to exported ONNX model
        """
        if not ONNX_AVAILABLE:
            print("Warning: ONNX not available, export may fail")
        
        print("Exporting model to ONNX format...")
        
        # Use default output path if not provided
        if output_path is None:
            model_stem = Path(self.model_path).stem
            output_path = f"{model_stem}.onnx"
        
        try:
            # Export using ultralytics
            exported_path = self.model.export(
                format='onnx',
                imgsz=imgsz,
                opset=opset,
                simplify=simplify,
                verbose=True
            )
            
            # Move to specified output path if different
            if str(exported_path) != output_path:
                os.rename(exported_path, output_path)
                exported_path = output_path
            
            print(f"ONNX model exported successfully: {exported_path}")
            
            # Verify the exported model
            if ONNX_AVAILABLE:
                self.verify_onnx_model(exported_path)
            
            return exported_path
            
        except Exception as e:
            raise RuntimeError(f"ONNX export failed: {e}")
    
    def verify_onnx_model(self, onnx_path: str):
        """Verify the exported ONNX model"""
        try:
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            print("✅ ONNX model verification passed")
            
            # Print model info
            print(f"Model input shape: {model.graph.input[0].type.tensor_type.shape}")
            print(f"Model outputs: {len(model.graph.output)}")
            
        except Exception as e:
            print(f"❌ ONNX model verification failed: {e}")
    
    def export_tensorrt(self, output_path: str = None, imgsz: int = 640,
                       precision: str = 'fp16', workspace: int = 4) -> str:
        """
        Export model to TensorRT format
        
        Args:
            output_path: Output path for TensorRT engine
            imgsz: Input image size
            precision: Precision mode ('fp32', 'fp16', 'int8')
            workspace: Workspace size in GB
            
        Returns:
            Path to exported TensorRT engine
        """
        print(f"Exporting model to TensorRT format (precision: {precision})...")
        
        # Use default output path if not provided
        if output_path is None:
            model_stem = Path(self.model_path).stem
            output_path = f"{model_stem}.engine"
        
        try:
            # Export using ultralytics
            exported_path = self.model.export(
                format='engine',
                imgsz=imgsz,
                half=(precision == 'fp16'),
                workspace=workspace,
                verbose=True
            )
            
            # Move to specified output path if different
            if str(exported_path) != output_path:
                os.rename(exported_path, output_path)
                exported_path = output_path
            
            print(f"TensorRT engine exported successfully: {exported_path}")
            return exported_path
            
        except Exception as e:
            raise RuntimeError(f"TensorRT export failed: {e}")
    
    def export_torchscript(self, output_path: str = None, imgsz: int = 640) -> str:
        """
        Export model to TorchScript format
        
        Args:
            output_path: Output path for TorchScript model
            imgsz: Input image size
            
        Returns:
            Path to exported TorchScript model
        """
        print("Exporting model to TorchScript format...")
        
        # Use default output path if not provided
        if output_path is None:
            model_stem = Path(self.model_path).stem
            output_path = f"{model_stem}.torchscript"
        
        try:
            # Export using ultralytics
            exported_path = self.model.export(
                format='torchscript',
                imgsz=imgsz,
                verbose=True
            )
            
            # Move to specified output path if different
            if str(exported_path) != output_path:
                os.rename(exported_path, output_path)
                exported_path = output_path
            
            print(f"TorchScript model exported successfully: {exported_path}")
            return exported_path
            
        except Exception as e:
            raise RuntimeError(f"TorchScript export failed: {e}")
    
    def export_all_formats(self, output_dir: str = "exported_models", 
                          imgsz: int = 640) -> dict:
        """
        Export model to all supported formats
        
        Args:
            output_dir: Directory to save exported models
            imgsz: Input image size
            
        Returns:
            Dictionary mapping format names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        model_stem = Path(self.model_path).stem
        exported_models = {}
        
        # Export ONNX
        try:
            onnx_path = output_path / f"{model_stem}.onnx"
            exported_models['onnx'] = self.export_onnx(str(onnx_path), imgsz)
        except Exception as e:
            print(f"ONNX export failed: {e}")
        
        # Export TensorRT (if available)
        if TENSORRT_AVAILABLE:
            try:
                engine_path = output_path / f"{model_stem}.engine"
                exported_models['tensorrt'] = self.export_tensorrt(str(engine_path), imgsz)
            except Exception as e:
                print(f"TensorRT export failed: {e}")
        
        # Export TorchScript
        try:
            torchscript_path = output_path / f"{model_stem}.torchscript"
            exported_models['torchscript'] = self.export_torchscript(str(torchscript_path), imgsz)
        except Exception as e:
            print(f"TorchScript export failed: {e}")
        
        # Copy original PyTorch model
        pytorch_path = output_path / f"{model_stem}.pt"
        if not pytorch_path.exists():
            import shutil
            shutil.copy2(self.model_path, pytorch_path)
            exported_models['pytorch'] = str(pytorch_path)
        
        print(f"\nExported models summary:")
        for format_name, path in exported_models.items():
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"  {format_name:12}: {path} ({file_size:.1f} MB)")
        
        return exported_models
    
    def benchmark_export_formats(self, exported_models: dict, 
                                test_image_path: str = None, num_runs: int = 50):
        """
        Benchmark inference speed for different export formats
        
        Args:
            exported_models: Dictionary of exported model paths
            test_image_path: Path to test image
            num_runs: Number of inference runs for averaging
        """
        print("\nBenchmarking exported model formats...")
        
        if test_image_path and not os.path.exists(test_image_path):
            print(f"Test image not found: {test_image_path}")
            return
        
        results = {}
        
        for format_name, model_path in exported_models.items():
            if not os.path.exists(model_path):
                continue
            
            try:
                print(f"\nBenchmarking {format_name} model...")
                
                # Load model
                if format_name == 'pytorch':
                    model = YOLO(model_path)
                elif format_name == 'onnx':
                    model = YOLO(model_path)  # Ultralytics supports ONNX inference
                elif format_name == 'tensorrt':
                    model = YOLO(model_path)  # Ultralytics supports TensorRT inference
                else:
                    print(f"Benchmark not implemented for {format_name}")
                    continue
                
                # Warm up
                for _ in range(5):
                    if test_image_path:
                        model.predict(test_image_path, verbose=False)
                    else:
                        # Use dummy input
                        import torch
                        dummy_input = torch.randn(1, 3, 640, 640)
                        model.predict(dummy_input, verbose=False)
                
                # Benchmark
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    if test_image_path:
                        model.predict(test_image_path, verbose=False)
                    else:
                        model.predict(dummy_input, verbose=False)
                    inference_time = time.time() - start_time
                    times.append(inference_time)
                
                avg_time = sum(times) / len(times)
                avg_fps = 1.0 / avg_time
                
                results[format_name] = {
                    'avg_time_ms': avg_time * 1000,
                    'avg_fps': avg_fps,
                    'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
                }
                
                print(f"  Average inference time: {avg_time*1000:.2f}ms")
                print(f"  Average FPS: {avg_fps:.1f}")
                
            except Exception as e:
                print(f"Benchmark failed for {format_name}: {e}")
        
        # Print comparison summary
        if results:
            print("\n" + "="*70)
            print("BENCHMARK RESULTS SUMMARY")
            print("="*70)
            print(f"{'Format':<12} {'Size (MB)':<10} {'Time (ms)':<12} {'FPS':<8}")
            print("-" * 70)
            
            # Sort by FPS (descending)
            sorted_results = sorted(results.items(), 
                                  key=lambda x: x[1]['avg_fps'], reverse=True)
            
            for format_name, metrics in sorted_results:
                print(f"{format_name:<12} {metrics['model_size_mb']:<10.1f} "
                      f"{metrics['avg_time_ms']:<12.2f} {metrics['avg_fps']:<8.1f}")
            
            print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Export waste detection model")
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--format', choices=['onnx', 'tensorrt', 'torchscript', 'all'],
                       default='all', help='Export format')
    parser.add_argument('--output', help='Output path for exported model')
    parser.add_argument('--output-dir', default='exported_models',
                       help='Output directory for multiple exports')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--precision', default='fp16', choices=['fp32', 'fp16', 'int8'],
                       help='Precision for TensorRT export')
    parser.add_argument('--workspace', type=int, default=4,
                       help='TensorRT workspace size in GB')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark exported models')
    parser.add_argument('--test-image', help='Test image for benchmarking')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return
    
    # Initialize exporter
    try:
        exporter = ModelExporter(args.model)
    except Exception as e:
        print(f"Error initializing exporter: {e}")
        return
    
    # Export model(s)
    try:
        if args.format == 'all':
            exported_models = exporter.export_all_formats(args.output_dir, args.imgsz)
        elif args.format == 'onnx':
            output_path = args.output or f"{Path(args.model).stem}.onnx"
            exported_path = exporter.export_onnx(output_path, args.imgsz)
            exported_models = {'onnx': exported_path}
        elif args.format == 'tensorrt':
            output_path = args.output or f"{Path(args.model).stem}.engine"
            exported_path = exporter.export_tensorrt(
                output_path, args.imgsz, args.precision, args.workspace
            )
            exported_models = {'tensorrt': exported_path}
        elif args.format == 'torchscript':
            output_path = args.output or f"{Path(args.model).stem}.torchscript"
            exported_path = exporter.export_torchscript(output_path, args.imgsz)
            exported_models = {'torchscript': exported_path}
        
        # Run benchmark if requested
        if args.benchmark:
            exporter.benchmark_export_formats(exported_models, args.test_image)
        
        print("\nExport completed successfully!")
        
    except Exception as e:
        print(f"Error during export: {e}")


if __name__ == "__main__":
    main()