"""
Visualization utilities for waste detection results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd

from utils import CLASS_COLORS, WASTE_CLASS_NAMES


class WasteDetectionVisualizer:
    """Handles visualization of detection results and training metrics"""
    
    def __init__(self):
        self.colors = CLASS_COLORS
        self.class_names = WASTE_CLASS_NAMES
        
    def draw_detection_box(self, image: np.ndarray, box: List[float], 
                          class_id: int, confidence: float, 
                          class_names: List[str] = None) -> np.ndarray:
        """
        Draw detection box on image
        
        Args:
            image: Input image (BGR format)
            box: Bounding box [x1, y1, x2, y2]
            class_id: Class index
            confidence: Detection confidence
            class_names: List of class names
            
        Returns:
            Image with drawn detection box
        """
        class_names = class_names or self.class_names
        
        if class_id >= len(class_names):
            return image
            
        class_name = class_names[class_id]
        color = self.colors.get(class_name, (255, 255, 255))
        
        x1, y1, x2, y2 = map(int, box)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def visualize_detections(self, image_path: str, detections: List[Dict],
                           save_path: str = None, show: bool = True) -> np.ndarray:
        """
        Visualize all detections on an image
        
        Args:
            image_path: Path to input image
            detections: List of detection dictionaries with keys:
                       'box', 'class_id', 'confidence'
            save_path: Path to save visualization
            show: Whether to display the image
            
        Returns:
            Image with visualized detections
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Draw each detection
        for det in detections:
            image = self.draw_detection_box(
                image, det['box'], det['class_id'], det['confidence']
            )
        
        # Add title
        title = f"Detections: {len(detections)} objects found"
        cv2.putText(image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"Visualization saved to: {save_path}")
        
        if show:
            cv2.imshow("Waste Detection Results", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return image
    
    def plot_training_metrics(self, results_dir: str, save_path: str = None):
        """
        Plot training metrics from YOLOv5 results
        
        Args:
            results_dir: Directory containing training results
            save_path: Path to save the plot
        """
        results_path = Path(results_dir)
        results_csv = results_path / "results.csv"
        
        if not results_csv.exists():
            print(f"Results file not found: {results_csv}")
            return
        
        # Load training results
        df = pd.read_csv(results_csv)
        df = df.strip()  # Remove leading/trailing whitespace from column names
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16)
        
        # Plot metrics
        metrics = [
            ('train/box_loss', 'Training Box Loss'),
            ('train/obj_loss', 'Training Objectness Loss'),
            ('train/cls_loss', 'Training Classification Loss'),
            ('val/box_loss', 'Validation Box Loss'),
            ('val/obj_loss', 'Validation Objectness Loss'), 
            ('val/cls_loss', 'Validation Classification Loss')
        ]
        
        for i, (metric, title) in enumerate(metrics):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            if metric in df.columns:
                ax.plot(df['epoch'], df[metric])
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, f'Metric {metric}\nnot found', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training metrics plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                             save_path: str = None):
        """
        Plot confusion matrix for model evaluation
        
        Args:
            confusion_matrix: Confusion matrix array
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to: {save_path}")
        
        plt.show()
    
    def plot_class_distribution(self, class_counts: Dict[str, int], 
                               save_path: str = None):
        """
        Plot class distribution in dataset
        
        Args:
            class_counts: Dictionary with class names and counts
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = [self.colors.get(cls, (128, 128, 128)) for cls in classes]
        
        # Convert BGR to RGB for matplotlib
        colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]
        
        bars = plt.bar(classes, counts, color=colors_rgb)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Waste Classes')
        plt.ylabel('Number of Instances')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add total count
        total = sum(counts)
        plt.text(0.02, 0.98, f'Total: {total} instances', 
                transform=plt.gca().transAxes, va='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to: {save_path}")
        
        plt.show()
    
    def create_detection_report(self, results_dir: str, output_path: str = None):
        """
        Create comprehensive detection report with visualizations
        
        Args:
            results_dir: Directory containing training/validation results
            output_path: Path to save the report
        """
        results_path = Path(results_dir)
        
        # Create report directory
        report_dir = Path(output_path) if output_path else results_path / "report"
        report_dir.mkdir(exist_ok=True)
        
        print(f"Generating detection report in: {report_dir}")
        
        # Generate individual plots
        try:
            # Training metrics
            self.plot_training_metrics(results_dir, 
                                     report_dir / "training_metrics.png")
            
            # Look for confusion matrix
            confusion_matrix_path = results_path / "confusion_matrix.png"
            if confusion_matrix_path.exists():
                print(f"Confusion matrix found: {confusion_matrix_path}")
            
            print("Detection report generated successfully!")
            
        except Exception as e:
            print(f"Error generating report: {e}")


def main():
    """Demo visualization functions"""
    visualizer = WasteDetectionVisualizer()
    
    # Create sample data for demonstration
    sample_detections = [
        {'box': [100, 100, 200, 200], 'class_id': 0, 'confidence': 0.95},
        {'box': [250, 150, 350, 250], 'class_id': 1, 'confidence': 0.88},
        {'box': [400, 50, 500, 180], 'class_id': 2, 'confidence': 0.76}
    ]
    
    # Sample class distribution
    sample_class_counts = {
        'plastic_bottle': 120,
        'paper_box': 95,
        'metal_can': 87,
        'glass_bottle': 76,
        'paper': 103
    }
    
    # Plot class distribution
    visualizer.plot_class_distribution(sample_class_counts)
    
    print("Visualization demo completed!")


if __name__ == "__main__":
    main()