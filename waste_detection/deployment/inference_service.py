"""
Deployment service for waste detection on Jetson TX2
Provides a production-ready inference service with MQTT communication
"""

import os
import sys
import time
import json
import threading
import queue
import argparse
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Optional, Callable

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available")

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("Warning: MQTT client not available")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils import WASTE_CLASS_NAMES, CLASS_COLORS
from utils.visualization import WasteDetectionVisualizer


class WasteDetectionService:
    """Production waste detection service for deployment"""
    
    def __init__(self, config_path: str = "deployment/service_config.json"):
        """Initialize the detection service"""
        self.config = self.load_config(config_path)
        self.model = None
        self.mqtt_client = None
        self.camera = None
        self.visualizer = WasteDetectionVisualizer()
        
        # Threading
        self.detection_queue = queue.Queue(maxsize=10)
        self.running = False
        self.threads = []
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'frames_processed': 0,
            'start_time': time.time(),
            'class_counts': {cls: 0 for cls in WASTE_CLASS_NAMES}
        }
        
        self.setup_model()
        if self.config.get('mqtt', {}).get('enabled', False):
            self.setup_mqtt()
    
    def load_config(self, config_path: str) -> Dict:
        """Load service configuration"""
        default_config = {
            'model': {
                'path': 'models/waste_detection_best.engine',
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
                'input_size': 640
            },
            'camera': {
                'source': 0,  # Default to webcam
                'width': 1280,
                'height': 720,
                'fps': 30
            },
            'mqtt': {
                'enabled': False,
                'broker': 'localhost',
                'port': 1883,
                'topics': {
                    'detections': 'recyclerover/detections',
                    'status': 'recyclerover/status',
                    'stats': 'recyclerover/stats'
                },
                'client_id': 'waste_detection_service'
            },
            'service': {
                'max_fps': 30,
                'save_detections': True,
                'output_dir': '/tmp/detections',
                'log_level': 'INFO'
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge configs
                self.merge_dict(default_config, user_config)
            except Exception as e:
                print(f"Error loading config {config_path}: {e}")
        
        return default_config
    
    def merge_dict(self, base: Dict, update: Dict):
        """Recursively merge dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self.merge_dict(base[key], value)
            else:
                base[key] = value
    
    def setup_model(self):
        """Load the detection model"""
        model_path = self.config['model']['path']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics required for model inference")
        
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded successfully: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def setup_mqtt(self):
        """Setup MQTT client for communication"""
        if not MQTT_AVAILABLE:
            print("MQTT not available, skipping MQTT setup")
            return
        
        mqtt_config = self.config['mqtt']
        
        self.mqtt_client = mqtt.Client(mqtt_config['client_id'])
        
        # Setup callbacks
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
        
        try:
            self.mqtt_client.connect(mqtt_config['broker'], mqtt_config['port'], 60)
            self.mqtt_client.loop_start()
            print(f"MQTT client connected to {mqtt_config['broker']}:{mqtt_config['port']}")
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            self.mqtt_client = None
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            print("Connected to MQTT broker")
            self.publish_status("online")
        else:
            print(f"Failed to connect to MQTT broker: {rc}")
    
    def on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        print("Disconnected from MQTT broker")
    
    def publish_detection(self, detections: List[Dict], frame_id: int):
        """Publish detection results via MQTT"""
        if not self.mqtt_client:
            return
        
        message = {
            'timestamp': datetime.now().isoformat(),
            'frame_id': frame_id,
            'detections': detections,
            'count': len(detections)
        }
        
        topic = self.config['mqtt']['topics']['detections']
        self.mqtt_client.publish(topic, json.dumps(message))
    
    def publish_status(self, status: str):
        """Publish service status via MQTT"""
        if not self.mqtt_client:
            return
        
        message = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'uptime': time.time() - self.stats['start_time']
        }
        
        topic = self.config['mqtt']['topics']['status']
        self.mqtt_client.publish(topic, json.dumps(message))
    
    def publish_stats(self):
        """Publish service statistics via MQTT"""
        if not self.mqtt_client:
            return
        
        runtime = time.time() - self.stats['start_time']
        fps = self.stats['frames_processed'] / runtime if runtime > 0 else 0
        
        message = {
            'timestamp': datetime.now().isoformat(),
            'uptime': runtime,
            'frames_processed': self.stats['frames_processed'],
            'total_detections': self.stats['total_detections'],
            'fps': fps,
            'class_counts': self.stats['class_counts']
        }
        
        topic = self.config['mqtt']['topics']['stats']
        self.mqtt_client.publish(topic, json.dumps(message))
    
    def setup_camera(self):
        """Setup camera input"""
        camera_config = self.config['camera']
        
        self.camera = cv2.VideoCapture(camera_config['source'])
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera: {camera_config['source']}")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
        self.camera.set(cv2.CAP_PROP_FPS, camera_config['fps'])
        
        print(f"Camera initialized: {camera_config['source']}")
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Run object detection on frame"""
        model_config = self.config['model']
        
        try:
            results = self.model.predict(
                frame,
                conf=model_config['confidence_threshold'],
                iou=model_config['iou_threshold'],
                imgsz=model_config['input_size'],
                verbose=False
            )
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, score, class_id in zip(boxes, scores, classes):
                        if class_id < len(WASTE_CLASS_NAMES):
                            detection = {
                                'box': box.tolist(),
                                'confidence': float(score),
                                'class_id': int(class_id),
                                'class_name': WASTE_CLASS_NAMES[class_id]
                            }
                            detections.append(detection)
                            
                            # Update statistics
                            self.stats['class_counts'][WASTE_CLASS_NAMES[class_id]] += 1
            
            self.stats['total_detections'] += len(detections)
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def detection_thread(self):
        """Thread for processing detection queue"""
        while self.running:
            try:
                item = self.detection_queue.get(timeout=1.0)
                if item is None:
                    break
                
                frame, frame_id = item
                
                # Run detection
                detections = self.detect_objects(frame)
                
                # Publish results
                if detections:
                    self.publish_detection(detections, frame_id)
                
                # Save detection results if enabled
                if self.config['service']['save_detections'] and detections:
                    self.save_detection_frame(frame, detections, frame_id)
                
                self.detection_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection thread error: {e}")
    
    def save_detection_frame(self, frame: np.ndarray, detections: List[Dict], frame_id: int):
        """Save frame with detections for debugging"""
        output_dir = Path(self.config['service']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Draw detections on frame
        result_frame = frame.copy()
        for det in detections:
            result_frame = self.visualizer.draw_detection_box(
                result_frame, det['box'], det['class_id'], det['confidence']
            )
        
        # Save frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}_{frame_id:06d}.jpg"
        output_path = output_dir / filename
        
        cv2.imwrite(str(output_path), result_frame)
    
    def run(self):
        """Run the detection service"""
        print("Starting waste detection service...")
        
        self.setup_camera()
        self.running = True
        
        # Start detection thread
        detection_thread = threading.Thread(target=self.detection_thread)
        detection_thread.start()
        self.threads.append(detection_thread)
        
        # Start stats publishing thread
        if self.mqtt_client:
            stats_thread = threading.Thread(target=self.stats_thread)
            stats_thread.start()
            self.threads.append(stats_thread)
        
        # Main camera loop
        frame_id = 0
        max_fps = self.config['service']['max_fps']
        frame_time = 1.0 / max_fps if max_fps > 0 else 0
        
        try:
            print("Service running... Press Ctrl+C to stop")
            
            while self.running:
                start_time = time.time()
                
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                frame_id += 1
                self.stats['frames_processed'] += 1
                
                # Add frame to detection queue (non-blocking)
                try:
                    self.detection_queue.put_nowait((frame, frame_id))
                except queue.Full:
                    # Skip frame if queue is full
                    pass
                
                # Frame rate limiting
                if frame_time > 0:
                    elapsed = time.time() - start_time
                    sleep_time = frame_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        
        finally:
            self.shutdown()
    
    def stats_thread(self):
        """Thread for publishing statistics"""
        while self.running:
            try:
                time.sleep(10)  # Publish stats every 10 seconds
                self.publish_stats()
            except Exception as e:
                print(f"Stats thread error: {e}")
    
    def shutdown(self):
        """Graceful shutdown"""
        print("Shutting down service...")
        
        self.running = False
        
        # Stop camera
        if self.camera:
            self.camera.release()
        
        # Stop detection queue
        self.detection_queue.put(None)
        
        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=5.0)
        
        # Disconnect MQTT
        if self.mqtt_client:
            self.publish_status("offline")
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        print("Service shutdown complete")
    
    def print_status(self):
        """Print current service status"""
        runtime = time.time() - self.stats['start_time']
        fps = self.stats['frames_processed'] / runtime if runtime > 0 else 0
        
        print(f"\nService Status:")
        print(f"  Uptime: {runtime:.1f}s")
        print(f"  Frames processed: {self.stats['frames_processed']}")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Total detections: {self.stats['total_detections']}")
        print(f"  Detection queue size: {self.detection_queue.qsize()}")


def create_default_config(config_path: str):
    """Create default configuration file"""
    default_config = {
        "model": {
            "path": "models/waste_detection_best.engine",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "input_size": 640
        },
        "camera": {
            "source": 0,
            "width": 1280,
            "height": 720,
            "fps": 30
        },
        "mqtt": {
            "enabled": True,
            "broker": "localhost",
            "port": 1883,
            "topics": {
                "detections": "recyclerover/detections",
                "status": "recyclerover/status",
                "stats": "recyclerover/stats"
            },
            "client_id": "waste_detection_service"
        },
        "service": {
            "max_fps": 30,
            "save_detections": False,
            "output_dir": "/tmp/detections",
            "log_level": "INFO"
        }
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Default configuration created: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Waste Detection Deployment Service")
    parser.add_argument('--config', default='deployment/service_config.json',
                       help='Path to service configuration file')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file')
    parser.add_argument('--status', action='store_true',
                       help='Print service status and exit')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config(args.config)
        return
    
    # Create service
    try:
        service = WasteDetectionService(args.config)
        
        if args.status:
            service.print_status()
            return
        
        # Run service
        service.run()
        
    except KeyboardInterrupt:
        print("\nService interrupted by user")
    except Exception as e:
        print(f"Service error: {e}")


if __name__ == "__main__":
    main()