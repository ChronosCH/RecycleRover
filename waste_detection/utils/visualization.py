"""
废品检测结果可视化工具
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
    """处理检测结果和训练指标的可视化"""
    
    def __init__(self):
        self.colors = CLASS_COLORS
        self.class_names = WASTE_CLASS_NAMES
        
    def draw_detection_box(self, image: np.ndarray, box: List[float], 
                          class_id: int, confidence: float, 
                          class_names: List[str] = None) -> np.ndarray:
        """
        在图像上绘制检测框
        
        参数:
            image: 输入图像 (BGR格式)
            box: 边界框 [x1, y1, x2, y2]
            class_id: 类别索引
            confidence: 检测置信度
            class_names: 类别名称列表
            
        返回:
            绘制了检测框的图像
        """
        class_names = class_names or self.class_names
        
        if class_id >= len(class_names):
            return image
            
        class_name = class_names[class_id]
        color = self.colors.get(class_name, (255, 255, 255))
        
        x1, y1, x2, y2 = map(int, box)
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # 绘制标签文本
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def visualize_detections(self, image_path: str, detections: List[Dict],
                           save_path: str = None, show: bool = True) -> np.ndarray:
        """
        在图像上可视化所有检测结果
        
        参数:
            image_path: 输入图像路径
            detections: 检测结果字典列表，包含键值:
                       'box', 'class_id', 'confidence'
            save_path: 保存可视化结果的路径
            show: 是否显示图像
            
        返回:
            带有可视化检测结果的图像
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 绘制每个检测结果
        for det in detections:
            image = self.draw_detection_box(
                image, det['box'], det['class_id'], det['confidence']
            )
        
        # 添加标题
        title = f"检测结果: 发现 {len(detections)} 个物体"
        cv2.putText(image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"可视化结果已保存至: {save_path}")
        
        if show:
            cv2.imshow("废品检测结果", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return image
    
    def plot_training_metrics(self, results_dir: str, save_path: str = None):
        """
        绘制YOLOv5训练指标
        
        参数:
            results_dir: 训练结果目录
            save_path: 保存图表的路径
        """
        results_path = Path(results_dir)
        results_csv = results_path / "results.csv"
        
        if not results_csv.exists():
            print(f"结果文件未找到: {results_csv}")
            return
        
        # 加载训练结果
        df = pd.read_csv(results_csv)
        df = df.strip()  # 移除列名的前后空格
        
        # 创建子图布局
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('训练指标', fontsize=16)
        
        # 绘制指标
        metrics = [
            ('train/box_loss', '训练边界框损失'),
            ('train/obj_loss', '训练目标性损失'),
            ('train/cls_loss', '训练分类损失'),
            ('val/box_loss', '验证边界框损失'),
            ('val/obj_loss', '验证目标性损失'), 
            ('val/cls_loss', '验证分类损失')
        ]
        
        for i, (metric, title) in enumerate(metrics):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            if metric in df.columns:
                ax.plot(df['epoch'], df[metric])
                ax.set_title(title)
                ax.set_xlabel('轮次')
                ax.set_ylabel('损失')
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, f'指标 {metric}\n未找到', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练指标图表已保存至: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                             save_path: str = None):
        """
        绘制模型评估的混淆矩阵
        
        参数:
            confusion_matrix: 混淆矩阵数组
            save_path: 保存图表的路径
        """
        plt.figure(figsize=(10, 8))
        
        # 标准化混淆矩阵
        cm_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # 创建热力图
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title('标准化混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵图表已保存至: {save_path}")
        
        plt.show()
    
    def plot_class_distribution(self, class_counts: Dict[str, int], 
                               save_path: str = None):
        """
        绘制数据集中的类别分布
        
        参数:
            class_counts: 包含类别名称和数量的字典
            save_path: 保存图表的路径
        """
        plt.figure(figsize=(12, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = [self.colors.get(cls, (128, 128, 128)) for cls in classes]
        
        # 将BGR转换为RGB格式用于matplotlib
        colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]
        
        bars = plt.bar(classes, counts, color=colors_rgb)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title('数据集中的类别分布')
        plt.xlabel('废品类别')
        plt.ylabel('实例数量')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # 添加总数统计
        total = sum(counts)
        plt.text(0.02, 0.98, f'总计: {total} 个实例', 
                transform=plt.gca().transAxes, va='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"类别分布图表已保存至: {save_path}")
        
        plt.show()
    
    def create_detection_report(self, results_dir: str, output_path: str = None):
        """
        创建包含可视化的综合检测报告
        
        参数:
            results_dir: 包含训练/验证结果的目录
            output_path: 保存报告的路径
        """
        results_path = Path(results_dir)
        
        # 创建报告目录
        report_dir = Path(output_path) if output_path else results_path / "report"
        report_dir.mkdir(exist_ok=True)
        
        print(f"正在生成检测报告至: {report_dir}")
        
        # 生成各个图表
        try:
            # 训练指标
            self.plot_training_metrics(results_dir, 
                                     report_dir / "training_metrics.png")
            
            # 查找混淆矩阵
            confusion_matrix_path = results_path / "confusion_matrix.png"
            if confusion_matrix_path.exists():
                print(f"发现混淆矩阵: {confusion_matrix_path}")
            
            print("检测报告生成成功!")
            
        except Exception as e:
            print(f"生成报告时出错: {e}")


def main():
    """演示可视化功能"""
    visualizer = WasteDetectionVisualizer()
    
    # 创建演示用的样本数据
    sample_detections = [
        {'box': [100, 100, 200, 200], 'class_id': 0, 'confidence': 0.95},
        {'box': [250, 150, 350, 250], 'class_id': 1, 'confidence': 0.88},
        {'box': [400, 50, 500, 180], 'class_id': 2, 'confidence': 0.76}
    ]
    
    # 样本类别分布
    sample_class_counts = {
        'plastic_bottle': 120,
        'paper_box': 95,
        'metal_can': 87,
        'glass_bottle': 76,
        'paper': 103
    }
    
    # 绘制类别分布
    visualizer.plot_class_distribution(sample_class_counts)
    
    print("可视化演示完成!")


if __name__ == "__main__":
    main()