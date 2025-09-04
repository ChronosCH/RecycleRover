"""
废品检测模型训练脚本
使用YOLOv5训练自定义废品检测模型
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch

# 如果可用，将yolov5添加到路径
try:
    import ultralytics
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("警告: ultralytics不可用。请安装: pip install ultralytics")

from utils import WASTE_CLASS_NAMES


class WasteDetectionTrainer:
    """处理废品检测模型的训练"""
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.model = None
        
    def load_config(self) -> dict:
        """从YAML文件加载训练配置"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_model(self, pretrained_weights: str = None):
        """初始化YOLOv5模型"""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("训练需要ultralytics包")
        
        weights = pretrained_weights or self.config['model']['pretrained_weights']
        
        # 创建模型
        self.model = YOLO(weights)
        
        print(f"模型已使用权重初始化: {weights}")
        print(f"模型参数: {sum(p.numel() for p in self.model.model.parameters()):,}")
        
    def train(self, data_config: str = "config/waste_data.yaml"):
        """训练废品检测模型"""
        if self.model is None:
            self.setup_model()
        
        # 从配置获取训练参数
        training_config = self.config['training']
        model_config = self.config['model']
        hardware_config = self.config['hardware']
        
        # 开始训练
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
            save_period=10,  # 每10个周期保存检查点
            cache=True,      # 缓存图像以加快训练速度
            # 额外优化设置
            amp=True,        # 自动混合精度
            single_cls=False, # 多类别
            rect=False,      # 矩形训练
            cos_lr=True,     # 余弦学习率调度器
            close_mosaic=10, # 在最后10个周期禁用马赛克
        )
        
        print("训练完成!")
        return results
    
    def validate_model(self, weights_path: str, data_config: str = "config/waste_data.yaml"):
        """验证训练模型性能"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"模型权重文件未找到: {weights_path}")
        
        # 加载训练好的模型权重
        model = YOLO(weights_path)
        
        # 验证
        results = model.val(
            data=data_config,
            imgsz=self.config['model']['input_size'],
            batch=1,  # 验证时使用batch=1
            device=self.config['hardware']['device'],
            plots=True,
            save_json=True,
            verbose=True
        )
        
        # 打印结果
        print("\n验证结果:")
        print(f"mAP@0.5: {results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {results.box.map:.4f}")
        
        # 各类别结果
        if hasattr(results.box, 'ap_class_index'):
            print("\n各类别 mAP@0.5:")
            for i, class_idx in enumerate(results.box.ap_class_index):
                if class_idx < len(WASTE_CLASS_NAMES):
                    class_name = WASTE_CLASS_NAMES[class_idx]
                    ap50 = results.box.ap50[i] if i < len(results.box.ap50) else 0
                    print(f"  {class_name}: {ap50:.4f}")
        
        return results
    
    def check_performance_targets(self, results):
        """检查模型是否达到性能目标"""
        targets = self.config['targets']
        
        map50 = results.box.map50
        target_map50 = targets['map_50']
        
        print(f"\n性能检查:")
        print(f"目标 mAP@0.5: {target_map50}")
        print(f"实际 mAP@0.5: {map50:.4f}")
        
        if map50 >= target_map50:
            print("✅ 性能目标已达成!")
            return True
        else:
            print("❌ 性能目标未达成。建议考虑:")
            print("  - 收集更多训练数据")
            print("  - 提升数据质量和标注准确性")
            print("  - 调整超参数")
            print("  - 增加训练轮次")
            return False


def main():
    parser = argparse.ArgumentParser(description="训练废品检测模型")
    parser.add_argument('--config', default='config/training_config.yaml',
                       help='训练配置文件路径')
    parser.add_argument('--data', default='config/waste_data.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--weights', default=None,
                       help='预训练权重文件路径')
    parser.add_argument('--validate-only', action='store_true',
                       help='仅验证现有模型')
    parser.add_argument('--model-path', default=None,
                       help='用于验证的模型路径')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件未找到: {args.config}")
        return
    
    if not os.path.exists(args.data):
        print(f"错误: 数据集配置文件未找到: {args.data}")
        print("请运行: python dataset/data_utils.py --create-yaml")
        return
    
    # 初始化训练器
    trainer = WasteDetectionTrainer(args.config)
    
    if args.validate_only:
        if not args.model_path:
            print("错误: 验证模式需要提供 --model-path 参数")
            return
        
        results = trainer.validate_model(args.model_path, args.data)
        trainer.check_performance_targets(results)
    else:
        # 设置模型
        trainer.setup_model(args.weights)
        
        # 开始训练
        print("开始废品检测模型训练...")
        print(f"配置文件: {args.config}")
        print(f"数据集: {args.data}")
        print(f"类别: {WASTE_CLASS_NAMES}")
        
        results = trainer.train(args.data)
        
        # 查找最佳权重
        project_dir = Path(trainer.config['hardware']['project'])
        name = trainer.config['hardware']['name']
        weights_dir = project_dir / name / "weights"
        best_weights = weights_dir / "best.pt"
        
        if best_weights.exists():
            print(f"\n验证最佳模型: {best_weights}")
            val_results = trainer.validate_model(str(best_weights), args.data)
            trainer.check_performance_targets(val_results)
        else:
            print("警告: 未找到最佳权重文件用于验证")


if __name__ == "__main__":
    main()