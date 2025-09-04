"""
废品检测实时推理脚本
支持摄像头、图像文件和视频文件
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("警告: ultralytics不可用。请安装: pip install ultralytics")

from utils import WASTE_CLASS_NAMES, CLASS_COLORS, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD
from utils.visualization import WasteDetectionVisualizer


class WasteDetectionInference:
    """实时废品检测推理引擎"""
    
    def __init__(self, model_path: str, conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                 iou_threshold: float = DEFAULT_IOU_THRESHOLD):
        """
        初始化推理引擎
        
        参数:
            model_path: 训练模型权重的路径
            conf_threshold: 检测的置信度阈值
            iou_threshold: NMS的IoU阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.visualizer = WasteDetectionVisualizer()
        
        self.load_model()
    
    def load_model(self):
        """加载训练好的YOLO模型"""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("推理需要ultralytics包")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict(self, image: np.ndarray) -> List[Dict]:
        """
        Run inference on a single image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Parse results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, score, class_id in zip(boxes, scores, classes):
                    detection = {
                        'box': box.tolist(),
                        'confidence': float(score),
                        'class_id': int(class_id),
                        'class_name': WASTE_CLASS_NAMES[class_id] if class_id < len(WASTE_CLASS_NAMES) else 'unknown'
                    }
                    detections.append(detection)
        
        return detections
    
    def process_image(self, image_path: str, output_path: str = None, 
                     show: bool = True) -> List[Dict]:
        """
        Process a single image file
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image
            show: Whether to display the result
            
        Returns:
            List of detections
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing image: {image_path}")
        
        # Run inference
        start_time = time.time()
        detections = self.predict(image)
        inference_time = time.time() - start_time
        
        print(f"Inference time: {inference_time:.3f}s")
        print(f"Detections found: {len(detections)}")
        
        # Print detection details
        for i, det in enumerate(detections):
            print(f"  {i+1}: {det['class_name']} ({det['confidence']:.3f})")
        
        # Visualize results
        if show or output_path:
            result_image = self.visualizer.visualize_detections(
                image_path, detections, output_path, show
            )
        
        return detections
    
    def process_video(self, video_path: str, output_path: str = None,
                     show: bool = True) -> None:
        """
        Process video file or webcam stream
        
        Args:
            video_path: Path to video file or camera index (e.g., 0 for webcam)
            output_path: Path to save output video
            show: Whether to display the video
        """
        # Open video source
        if video_path.isdigit():
            cap = cv2.VideoCapture(int(video_path))
            print(f"Opening webcam: {video_path}")
        else:
            cap = cv2.VideoCapture(video_path)
            print(f"Opening video: {video_path}")
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output video: {output_path}")
        
        # Performance tracking
        frame_count = 0
        total_inference_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or failed to read frame")
                    break
                
                frame_count += 1
                
                # Run inference
                start_time = time.time()
                detections = self.predict(frame)
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                
                # Draw detections
                for det in detections:
                    frame = self.visualizer.draw_detection_box(
                        frame, det['box'], det['class_id'], det['confidence']
                    )
                
                # Add performance info
                avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
                info_text = f"FPS: {avg_fps:.1f} | Detections: {len(detections)}"
                cv2.putText(frame, info_text, (10, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Save frame if output specified
                if out:
                    out.write(frame)
                
                # Display frame
                if show:
                    cv2.imshow('Waste Detection', frame)
                    
                    # Check for exit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        print("Exit requested by user")
                        break
                
                # Print progress for long videos
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames, avg FPS: {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show:
                cv2.destroyAllWindows()
            
            # Print final statistics
            avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
            print(f"\nProcessing completed:")
            print(f"  Total frames: {frame_count}")
            print(f"  Total time: {total_inference_time:.2f}s")
            print(f"  Average FPS: {avg_fps:.1f}")
    
    def benchmark_performance(self, test_image_path: str, num_runs: int = 100):
        """
        Benchmark inference performance
        
        Args:
            test_image_path: Path to test image
            num_runs: Number of inference runs for averaging
        """
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"Test image not found: {test_image_path}")
        
        image = cv2.imread(test_image_path)
        if image is None:
            raise ValueError(f"Could not load test image: {test_image_path}")
        
        print(f"Benchmarking performance with {num_runs} runs...")
        
        # Warm up
        for _ in range(5):
            self.predict(image)
        
        # Benchmark
        times = []
        for i in range(num_runs):
            start_time = time.time()
            detections = self.predict(image)
            inference_time = time.time() - start_time
            times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs")
        
        # Calculate statistics
        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        avg_fps = 1.0 / avg_time
        
        print(f"\nPerformance Results:")
        print(f"  Average inference time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"  Min/Max inference time: {min_time*1000:.2f}ms / {max_time*1000:.2f}ms")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Image size: {image.shape[1]}x{image.shape[0]}")


def main():
    parser = argparse.ArgumentParser(description="Waste Detection Inference")
    parser.add_argument('--model', required=True, help='Path to trained model weights')
    parser.add_argument('--source', required=True, 
                       help='Input source (image file, video file, or camera index)')
    parser.add_argument('--output', help='Output path for results')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF_THRESHOLD,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=DEFAULT_IOU_THRESHOLD,
                       help='IoU threshold for NMS')
    parser.add_argument('--no-show', action='store_true', 
                       help='Do not display results')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Initialize inference engine
    try:
        inference = WasteDetectionInference(
            args.model, args.conf, args.iou
        )
    except Exception as e:
        print(f"Error initializing inference engine: {e}")
        return
    
    # Run benchmark if requested
    if args.benchmark:
        if os.path.isfile(args.source):
            inference.benchmark_performance(args.source)
        else:
            print("Benchmark requires an image file as source")
        return
    
    # Determine source type and process
    try:
        if args.source.isdigit() or args.source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Video or webcam
            inference.process_video(
                args.source, args.output, not args.no_show
            )
        elif os.path.isfile(args.source):
            # Image file
            inference.process_image(
                args.source, args.output, not args.no_show
            )
        else:
            print(f"Error: Source not found or unsupported: {args.source}")
    
    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    main()