"""
Performance metrics calculation for waste detection evaluation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class WasteDetectionMetrics:
    """Calculate comprehensive metrics for waste detection performance"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box1: [x1, y1, x2, y2] format
            box2: [x1, y1, x2, y2] format
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Calculate intersection area
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_ap(self, precisions: np.ndarray, recalls: np.ndarray) -> float:
        """
        Calculate Average Precision (AP) using precision-recall curve
        
        Args:
            precisions: Precision values
            recalls: Recall values
            
        Returns:
            Average Precision value
        """
        # Add sentinel values
        mrec = np.concatenate(([0.], recalls, [1.]))
        mpre = np.concatenate(([0.], precisions, [0.]))
        
        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        # Find where recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # Calculate AP
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
    
    def calculate_map(self, predictions: List[Dict], ground_truth: List[Dict],
                     iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate mean Average Precision (mAP) for object detection
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            Dictionary with mAP metrics
        """
        ap_per_class = {}
        
        for class_id, class_name in enumerate(self.class_names):
            # Filter predictions and ground truth for this class
            class_predictions = [p for p in predictions if p['class_id'] == class_id]
            class_gt = [gt for gt in ground_truth if gt['class_id'] == class_id]
            
            if not class_gt:
                ap_per_class[class_name] = 0.0
                continue
            
            # Sort predictions by confidence
            class_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Track true positives and false positives
            tp = np.zeros(len(class_predictions))
            fp = np.zeros(len(class_predictions))
            
            # Mark ground truth as not detected
            gt_detected = [False] * len(class_gt)
            
            for pred_idx, prediction in enumerate(class_predictions):
                best_iou = 0.0
                best_gt_idx = -1
                
                # Find best matching ground truth
                for gt_idx, gt in enumerate(class_gt):
                    if gt_detected[gt_idx]:
                        continue
                    
                    iou = self.calculate_iou(prediction['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Check if detection is positive
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    tp[pred_idx] = 1.0
                    gt_detected[best_gt_idx] = True
                else:
                    fp[pred_idx] = 1.0
            
            # Calculate precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recall = tp_cumsum / len(class_gt)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            
            # Calculate AP
            ap = self.calculate_ap(precision, recall)
            ap_per_class[class_name] = ap
        
        # Calculate overall mAP
        map_value = np.mean(list(ap_per_class.values()))
        
        return {
            'mAP': map_value,
            'AP_per_class': ap_per_class
        }
    
    def calculate_classification_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict:
        """
        Calculate classification metrics (precision, recall, F1)
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Dictionary with classification metrics
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        # Calculate per-class metrics
        report = classification_report(
            y_true, y_pred, 
            labels=range(self.num_classes),
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'confusion_matrix': cm,
            'classification_report': report,
            'overall_accuracy': report['accuracy'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_precision': report['weighted avg']['precision'],
            'weighted_recall': report['weighted avg']['recall'],
            'weighted_f1': report['weighted avg']['f1-score']
        }
    
    def calculate_detection_speed_metrics(self, inference_times: List[float]) -> Dict:
        """
        Calculate inference speed metrics
        
        Args:
            inference_times: List of inference times in seconds
            
        Returns:
            Dictionary with speed metrics
        """
        times = np.array(inference_times)
        
        return {
            'mean_inference_time': float(np.mean(times)),
            'std_inference_time': float(np.std(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'median_inference_time': float(np.median(times)),
            'p95_inference_time': float(np.percentile(times, 95)),
            'p99_inference_time': float(np.percentile(times, 99)),
            'mean_fps': float(1.0 / np.mean(times)),
            'max_fps': float(1.0 / np.min(times)),
            'min_fps': float(1.0 / np.max(times))
        }
    
    def plot_precision_recall_curve(self, precisions: np.ndarray, recalls: np.ndarray,
                                   class_name: str, save_path: str = None):
        """Plot precision-recall curve for a specific class"""
        plt.figure(figsize=(8, 6))
        
        plt.plot(recalls, precisions, 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve: {class_name}')
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        # Calculate and display AP
        ap = self.calculate_ap(precisions, recalls)
        plt.text(0.02, 0.98, f'AP = {ap:.3f}', transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                             normalize: bool = True, save_path: str = None):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(10, 8))
        
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            cm = confusion_matrix
            fmt = 'd'
            title = 'Confusion Matrix'
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_metrics_report(self, metrics: Dict, save_path: str = None) -> str:
        """Generate a comprehensive metrics report"""
        report = []
        report.append("="*60)
        report.append("WASTE DETECTION PERFORMANCE REPORT")
        report.append("="*60)
        
        # Detection metrics
        if 'mAP' in metrics:
            report.append(f"\nDetection Metrics:")
            report.append(f"  mAP@0.5: {metrics['mAP']:.4f}")
            
            if 'AP_per_class' in metrics:
                report.append(f"\nPer-Class Average Precision:")
                for class_name, ap in metrics['AP_per_class'].items():
                    report.append(f"  {class_name:15}: {ap:.4f}")
        
        # Classification metrics
        if 'overall_accuracy' in metrics:
            report.append(f"\nClassification Metrics:")
            report.append(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
            report.append(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
            report.append(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
            report.append(f"  Macro F1:         {metrics['macro_f1']:.4f}")
        
        # Speed metrics
        if 'mean_inference_time' in metrics:
            report.append(f"\nInference Speed Metrics:")
            report.append(f"  Mean inference time: {metrics['mean_inference_time']*1000:.2f}ms")
            report.append(f"  Mean FPS:           {metrics['mean_fps']:.1f}")
            report.append(f"  P95 inference time: {metrics['p95_inference_time']*1000:.2f}ms")
            report.append(f"  P99 inference time: {metrics['p99_inference_time']*1000:.2f}ms")
        
        report.append("="*60)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


def main():
    """Demo metrics calculation"""
    from utils import WASTE_CLASS_NAMES
    
    # Initialize metrics calculator
    metrics_calc = WasteDetectionMetrics(WASTE_CLASS_NAMES)
    
    # Demo IoU calculation
    box1 = [100, 100, 200, 200]
    box2 = [150, 150, 250, 250]
    iou = metrics_calc.calculate_iou(box1, box2)
    print(f"IoU between boxes: {iou:.3f}")
    
    # Demo speed metrics
    inference_times = np.random.normal(0.033, 0.005, 100)  # ~30 FPS
    speed_metrics = metrics_calc.calculate_detection_speed_metrics(inference_times)
    
    print("\nSpeed Metrics Demo:")
    for key, value in speed_metrics.items():
        if 'time' in key:
            print(f"  {key}: {value*1000:.2f}ms")
        else:
            print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()