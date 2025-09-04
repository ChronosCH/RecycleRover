"""
废品检测数据集工具
处理数据集准备、标注和管理
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

from utils import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, WASTE_CLASS_NAMES


class WasteDatasetManager:
    """管理废品检测数据集的准备和组织"""
    
    def __init__(self, dataset_root: str = "dataset"):
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / "images" 
        self.labels_dir = self.dataset_root / "labels"
        
        # 如果目录不存在则创建
        for split in ['train', 'val', 'test']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    def prepare_dataset_structure(self):
        """创建标准数据集目录结构"""
        print("设置数据集目录结构...")
        
        # 为每个目录创建README文件
        readme_content = {
            'train': "训练图像和标签（数据集的80%）",
            'val': "验证图像和标签（数据集的10%）", 
            'test': "测试图像和标签（数据集的10%）"
        }
        
        for split, description in readme_content.items():
            readme_path = self.images_dir / split / "README.md"
            with open(readme_path, 'w') as f:
                f.write(f"# {split.capitalize()} 数据集\n\n{description}\n")
                
        print("数据集结构创建成功!")
        return True
    
    def split_dataset(self, source_images_dir: str, source_labels_dir: str):
        """
        Splits dataset into train/val/test based on configured ratios
        
        Args:
            source_images_dir: Directory containing all images
            source_labels_dir: Directory containing all labels
        """
        source_images = Path(source_images_dir)
        source_labels = Path(source_labels_dir)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(source_images.glob(f"*{ext}"))
            image_files.extend(source_images.glob(f"*{ext.upper()}"))
        
        # Shuffle for random split
        random.shuffle(image_files)
        
        total_images = len(image_files)
        train_count = int(total_images * TRAIN_RATIO)
        val_count = int(total_images * VAL_RATIO)
        
        # Split files
        splits = {
            'train': image_files[:train_count],
            'val': image_files[train_count:train_count + val_count],
            'test': image_files[train_count + val_count:]
        }
        
        print(f"Splitting {total_images} images:")
        print(f"  Train: {len(splits['train'])} images")
        print(f"  Val: {len(splits['val'])} images") 
        print(f"  Test: {len(splits['test'])} images")
        
        # Copy files to respective directories
        for split_name, files in splits.items():
            for img_file in files:
                # Copy image
                dest_img = self.images_dir / split_name / img_file.name
                shutil.copy2(img_file, dest_img)
                
                # Copy corresponding label if exists
                label_file = source_labels / (img_file.stem + '.txt')
                if label_file.exists():
                    dest_label = self.labels_dir / split_name / label_file.name
                    shutil.copy2(label_file, dest_label)
        
        print("Dataset split completed successfully!")
    
    def validate_annotations(self, split: str = None) -> Dict[str, int]:
        """
        Validates YOLO format annotations
        
        Args:
            split: Specific split to validate, or None for all splits
            
        Returns:
            Dictionary with validation statistics
        """
        splits_to_check = [split] if split else ['train', 'val', 'test']
        stats = {
            'total_images': 0,
            'total_labels': 0,
            'missing_labels': 0,
            'invalid_annotations': 0,
            'class_distribution': {cls: 0 for cls in WASTE_CLASS_NAMES}
        }
        
        for split_name in splits_to_check:
            img_dir = self.images_dir / split_name
            label_dir = self.labels_dir / split_name
            
            image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            stats['total_images'] += len(image_files)
            
            for img_file in image_files:
                label_file = label_dir / (img_file.stem + '.txt')
                
                if not label_file.exists():
                    stats['missing_labels'] += 1
                    continue
                
                stats['total_labels'] += 1
                
                # Validate label file format
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                stats['invalid_annotations'] += 1
                                continue
                            
                            class_id = int(parts[0])
                            if 0 <= class_id < len(WASTE_CLASS_NAMES):
                                stats['class_distribution'][WASTE_CLASS_NAMES[class_id]] += 1
                            else:
                                stats['invalid_annotations'] += 1
                                
                except Exception as e:
                    stats['invalid_annotations'] += 1
                    print(f"Error reading {label_file}: {e}")
        
        return stats
    
    def print_dataset_stats(self):
        """Prints comprehensive dataset statistics"""
        stats = self.validate_annotations()
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total Images: {stats['total_images']}")
        print(f"Total Label Files: {stats['total_labels']}")
        print(f"Missing Labels: {stats['missing_labels']}")
        print(f"Invalid Annotations: {stats['invalid_annotations']}")
        
        print("\nClass Distribution:")
        print("-" * 30)
        total_objects = sum(stats['class_distribution'].values())
        for class_name, count in stats['class_distribution'].items():
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"  {class_name:15}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\nTotal Objects: {total_objects}")
        print("="*50)


def create_data_yaml(output_path: str = "config/waste_data.yaml"):
    """Creates the data.yaml file required by YOLOv5"""
    data_config = {
        'path': '../dataset',
        'train': 'images/train',
        'val': 'images/val', 
        'test': 'images/test',
        'nc': len(WASTE_CLASS_NAMES),
        'names': WASTE_CLASS_NAMES
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Created data configuration file: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Waste Detection Dataset Utilities")
    parser.add_argument('--prepare', action='store_true', 
                       help='Prepare dataset directory structure')
    parser.add_argument('--split', nargs=2, metavar=('IMAGES_DIR', 'LABELS_DIR'),
                       help='Split dataset from source directories')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset annotations')
    parser.add_argument('--stats', action='store_true',
                       help='Print dataset statistics')
    parser.add_argument('--create-yaml', action='store_true',
                       help='Create YOLOv5 data.yaml file')
    
    args = parser.parse_args()
    
    dataset_manager = WasteDatasetManager()
    
    if args.prepare:
        dataset_manager.prepare_dataset_structure()
    
    if args.split:
        dataset_manager.split_dataset(args.split[0], args.split[1])
    
    if args.validate or args.stats:
        dataset_manager.print_dataset_stats()
    
    if args.create_yaml:
        create_data_yaml()
    
    if not any(vars(args).values()):
        print("No action specified. Use --help for available options.")


if __name__ == "__main__":
    main()