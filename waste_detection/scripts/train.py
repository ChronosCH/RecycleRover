"""
Training script for waste detection model
Uses YOLOv5 for training a custom waste detection model
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch

# Add yolov5 to path if available
try:
    import ultralytics
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Please install: pip install ultralytics")

from utils import WASTE_CLASS_NAMES


class WasteDetectionTrainer:
    """Handles training of waste detection models"""
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.model = None
        
    def load_config(self) -> dict:
        """Load training configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_model(self, pretrained_weights: str = None):
        """Initialize YOLOv5 model"""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package is required for training")
        
        weights = pretrained_weights or self.config['model']['pretrained_weights']
        
        # Create model
        self.model = YOLO(weights)
        
        print(f"Model initialized with weights: {weights}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
        
    def train(self, data_config: str = "config/waste_data.yaml"):
        """Train the waste detection model"""
        if self.model is None:
            self.setup_model()
        
        # Training parameters from config
        training_config = self.config['training']
        model_config = self.config['model']
        hardware_config = self.config['hardware']
        
        # Start training
        results = self.model.train(
            data=data_config,
            epochs=training_config['epochs'],
            batch=training_config['batch_size'],
            imgsz=model_config['input_size'],
            lr0=training_config['learning_rate'],
            momentum=training_config['momentum'],
            weight_decay=training_config['weight_decay'],
            warmup_epochs=training_config['warmup_epochs'],
            project=hardware_config['project'],
            name=hardware_config['name'],
            exist_ok=hardware_config['exist_ok'],
            device=hardware_config['device'],
            workers=hardware_config['workers'],
            patience=self.config['early_stopping']['patience'],
            save_period=10,  # Save checkpoint every 10 epochs
            cache=True,      # Cache images for faster training
            # Additional optimizations
            amp=True,        # Automatic Mixed Precision
            single_cls=False, # Multiple classes
            rect=False,      # Rectangular training
            cos_lr=True,     # Cosine learning rate scheduler
            close_mosaic=10, # Disable mosaic in last 10 epochs
        )
        
        print("Training completed!")
        return results
    
    def validate_model(self, weights_path: str, data_config: str = "config/waste_data.yaml"):
        """Validate trained model performance"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        # Load model with trained weights
        model = YOLO(weights_path)
        
        # Validate
        results = model.val(
            data=data_config,
            imgsz=self.config['model']['input_size'],
            batch=1,  # Use batch=1 for validation
            device=self.config['hardware']['device'],
            plots=True,
            save_json=True,
            verbose=True
        )
        
        # Print results
        print("\nValidation Results:")
        print(f"mAP@0.5: {results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {results.box.map:.4f}")
        
        # Per-class results
        if hasattr(results.box, 'ap_class_index'):
            print("\nPer-class mAP@0.5:")
            for i, class_idx in enumerate(results.box.ap_class_index):
                if class_idx < len(WASTE_CLASS_NAMES):
                    class_name = WASTE_CLASS_NAMES[class_idx]
                    ap50 = results.box.ap50[i] if i < len(results.box.ap50) else 0
                    print(f"  {class_name}: {ap50:.4f}")
        
        return results
    
    def check_performance_targets(self, results):
        """Check if model meets performance targets"""
        targets = self.config['targets']
        
        map50 = results.box.map50
        target_map50 = targets['map_50']
        
        print(f"\nPerformance Check:")
        print(f"Target mAP@0.5: {target_map50}")
        print(f"Achieved mAP@0.5: {map50:.4f}")
        
        if map50 >= target_map50:
            print("✅ Performance target achieved!")
            return True
        else:
            print("❌ Performance target not met. Consider:")
            print("  - Collecting more training data")
            print("  - Improving data quality and annotations")
            print("  - Adjusting hyperparameters")
            print("  - Training for more epochs")
            return False


def main():
    parser = argparse.ArgumentParser(description="Train waste detection model")
    parser.add_argument('--config', default='config/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--data', default='config/waste_data.yaml',
                       help='Path to dataset configuration file')
    parser.add_argument('--weights', default=None,
                       help='Path to pretrained weights')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing model')
    parser.add_argument('--model-path', default=None,
                       help='Path to model for validation')
    
    args = parser.parse_args()
    
    # Check if configuration files exist
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return
    
    if not os.path.exists(args.data):
        print(f"Error: Dataset configuration not found: {args.data}")
        print("Please run: python dataset/data_utils.py --create-yaml")
        return
    
    # Initialize trainer
    trainer = WasteDetectionTrainer(args.config)
    
    if args.validate_only:
        if not args.model_path:
            print("Error: --model-path required for validation")
            return
        
        results = trainer.validate_model(args.model_path, args.data)
        trainer.check_performance_targets(results)
    else:
        # Setup model
        trainer.setup_model(args.weights)
        
        # Start training
        print("Starting waste detection model training...")
        print(f"Configuration: {args.config}")
        print(f"Dataset: {args.data}")
        print(f"Classes: {WASTE_CLASS_NAMES}")
        
        results = trainer.train(args.data)
        
        # Find best weights
        project_dir = Path(trainer.config['hardware']['project'])
        name = trainer.config['hardware']['name']
        weights_dir = project_dir / name / "weights"
        best_weights = weights_dir / "best.pt"
        
        if best_weights.exists():
            print(f"\nValidating best model: {best_weights}")
            val_results = trainer.validate_model(str(best_weights), args.data)
            trainer.check_performance_targets(val_results)
        else:
            print("Warning: Best weights not found for validation")


if __name__ == "__main__":
    main()