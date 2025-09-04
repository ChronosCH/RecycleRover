"""
废品检测模型评估脚本
提供综合评估指标和分析
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("警告: ultralytics不可用。请安装: pip install ultralytics")

from utils import WASTE_CLASS_NAMES
from utils.visualization import WasteDetectionVisualizer


class WasteDetectionEvaluator:
    """废品检测模型的综合评估"""
    
    def __init__(self, model_path: str, data_config: str):
        """
        初始化评估器
        
        参数:
            model_path: 训练模型权重的路径
            data_config: 数据集配置文件路径
        """
        self.model_path = model_path
        self.data_config = data_config
        self.model = None
        self.visualizer = WasteDetectionVisualizer()
        
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("评估需要ultralytics包")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def evaluate_model(self, split: str = 'test', save_results: bool = True) -> Dict:
        """
        Evaluate model on specified dataset split
        
        Args:
            split: Dataset split to evaluate ('test', 'val', or 'train')
            save_results: Whether to save detailed results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"Evaluating model on {split} split...")
        
        # Run validation
        results = self.model.val(
            data=self.data_config,
            split=split,
            imgsz=640,
            batch=1,
            device='cpu',  # Use CPU for consistent results
            plots=True,
            save_json=True,
            verbose=True,
            project='runs/eval',
            name=f'{split}_evaluation'
        )
        
        # Extract metrics
        metrics = {
            'map_50': float(results.box.map50),
            'map_50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'split': split,
            'model_path': self.model_path
        }
        
        # Per-class metrics
        if hasattr(results.box, 'ap_class_index'):
            per_class_metrics = {}
            for i, class_idx in enumerate(results.box.ap_class_index):
                if class_idx < len(WASTE_CLASS_NAMES):
                    class_name = WASTE_CLASS_NAMES[class_idx]
                    per_class_metrics[class_name] = {
                        'ap_50': float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
                        'ap_50_95': float(results.box.ap[i]) if i < len(results.box.ap) else 0.0
                    }
            metrics['per_class'] = per_class_metrics
        
        # Print summary
        self.print_evaluation_summary(metrics)
        
        # Save results if requested
        if save_results:
            output_dir = Path('runs/eval') / f'{split}_evaluation'
            self.save_evaluation_results(metrics, output_dir)
        
        return metrics
    
    def print_evaluation_summary(self, metrics: Dict):
        """Print formatted evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Dataset Split: {metrics['split']}")
        print(f"Model: {metrics['model_path']}")
        print(f"mAP@0.5: {metrics['map_50']:.4f}")
        print(f"mAP@0.5:0.95: {metrics['map_50_95']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        if 'per_class' in metrics:
            print("\nPer-Class Results (mAP@0.5):")
            print("-" * 40)
            for class_name, class_metrics in metrics['per_class'].items():
                ap50 = class_metrics['ap_50']
                print(f"  {class_name:15}: {ap50:.4f}")
        
        print("="*60)
    
    def save_evaluation_results(self, metrics: Dict, output_dir: Path):
        """Save evaluation results to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Evaluation results saved to: {output_dir}")
    
    def compare_models(self, model_paths: List[str], split: str = 'test') -> pd.DataFrame:
        """
        Compare multiple models on the same dataset
        
        Args:
            model_paths: List of paths to model weights
            split: Dataset split to evaluate
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_path in model_paths:
            if not os.path.exists(model_path):
                print(f"Warning: Model not found: {model_path}")
                continue
            
            print(f"\nEvaluating model: {model_path}")
            
            # Load model
            try:
                temp_model = YOLO(model_path)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                continue
            
            # Evaluate
            eval_results = temp_model.val(
                data=self.data_config,
                split=split,
                imgsz=640,
                batch=1,
                device='cpu',
                verbose=False
            )
            
            # Store results
            model_name = Path(model_path).stem
            result = {
                'model': model_name,
                'model_path': model_path,
                'map_50': float(eval_results.box.map50),
                'map_50_95': float(eval_results.box.map),
                'precision': float(eval_results.box.mp),
                'recall': float(eval_results.box.mr)
            }
            results.append(result)
        
        # Create comparison DataFrame
        df = pd.DataFrame(results)
        
        if not df.empty:
            # Sort by mAP@0.5
            df = df.sort_values('map_50', ascending=False)
            
            # Print comparison table
            print("\n" + "="*80)
            print("MODEL COMPARISON")
            print("="*80)
            print(df.to_string(index=False, float_format='%.4f'))
            print("="*80)
        
        return df
    
    def analyze_failure_cases(self, split: str = 'test', threshold: float = 0.5) -> Dict:
        """
        Analyze failure cases and misclassifications
        
        Args:
            split: Dataset split to analyze
            threshold: Confidence threshold for analysis
            
        Returns:
            Dictionary with failure analysis
        """
        print(f"Analyzing failure cases on {split} split...")
        
        # This would require detailed prediction analysis
        # For now, provide a framework for future implementation
        
        analysis = {
            'low_confidence_detections': [],
            'missed_detections': [],
            'false_positives': [],
            'class_confusion': {}
        }
        
        print("Failure case analysis framework ready.")
        print("Note: Detailed implementation requires access to ground truth data.")
        
        return analysis
    
    def generate_performance_report(self, output_dir: str = 'evaluation_report'):
        """Generate comprehensive performance report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating performance report in: {output_path}")
        
        # Evaluate on test set
        test_metrics = self.evaluate_model('test', save_results=False)
        
        # Evaluate on validation set for comparison
        val_metrics = self.evaluate_model('val', save_results=False)
        
        # Create performance comparison
        comparison_data = {
            'Metric': ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall'],
            'Validation': [
                val_metrics['map_50'],
                val_metrics['map_50_95'],
                val_metrics['precision'],
                val_metrics['recall']
            ],
            'Test': [
                test_metrics['map_50'],
                test_metrics['map_50_95'],
                test_metrics['precision'],
                test_metrics['recall']
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_file = output_path / 'performance_comparison.csv'
        df_comparison.to_csv(comparison_file, index=False)
        
        # Plot comparison
        self.plot_performance_comparison(df_comparison, output_path / 'performance_comparison.png')
        
        # Generate per-class analysis if available
        if 'per_class' in test_metrics:
            self.plot_per_class_performance(
                test_metrics['per_class'], 
                output_path / 'per_class_performance.png'
            )
        
        # Save detailed metrics
        with open(output_path / 'test_metrics.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        with open(output_path / 'val_metrics.json', 'w') as f:
            json.dump(val_metrics, f, indent=2)
        
        print(f"Performance report generated successfully in: {output_path}")
    
    def plot_performance_comparison(self, df: pd.DataFrame, save_path: str):
        """Plot performance comparison between validation and test sets"""
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(df['Metric']))
        width = 0.35
        
        plt.bar(x - width/2, df['Validation'], width, label='Validation', alpha=0.8)
        plt.bar(x + width/2, df['Test'], width, label='Test', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance: Validation vs Test')
        plt.xticks(x, df['Metric'])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (val, test) in enumerate(zip(df['Validation'], df['Test'])):
            plt.text(i - width/2, val + 0.01, f'{val:.3f}', ha='center', va='bottom')
            plt.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_per_class_performance(self, per_class_metrics: Dict, save_path: str):
        """Plot per-class performance metrics"""
        classes = list(per_class_metrics.keys())
        ap50_scores = [per_class_metrics[cls]['ap_50'] for cls in classes]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(classes, ap50_scores, alpha=0.8)
        
        # Color bars based on performance
        for bar, score in zip(bars, ap50_scores):
            if score >= 0.9:
                bar.set_color('green')
            elif score >= 0.7:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.xlabel('Waste Classes')
        plt.ylabel('mAP@0.5')
        plt.title('Per-Class Detection Performance')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Add target line
        plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, 
                   label='Target (95%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate waste detection model")
    parser.add_argument('--model', required=True, help='Path to model weights')
    parser.add_argument('--data', default='config/waste_data.yaml',
                       help='Path to dataset configuration')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--compare', nargs='+', help='Compare multiple models')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive performance report')
    parser.add_argument('--output', default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if data config exists
    if not os.path.exists(args.data):
        print(f"Error: Dataset configuration not found: {args.data}")
        return
    
    if args.compare:
        # Model comparison mode
        if not os.path.exists(args.model):
            print(f"Error: Primary model not found: {args.model}")
            return
        
        evaluator = WasteDetectionEvaluator(args.model, args.data)
        all_models = [args.model] + args.compare
        
        comparison_df = evaluator.compare_models(all_models, args.split)
        
        # Save comparison results
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        
    else:
        # Single model evaluation
        if not os.path.exists(args.model):
            print(f"Error: Model not found: {args.model}")
            return
        
        evaluator = WasteDetectionEvaluator(args.model, args.data)
        
        if args.report:
            evaluator.generate_performance_report(args.output)
        else:
            metrics = evaluator.evaluate_model(args.split)


if __name__ == "__main__":
    main()