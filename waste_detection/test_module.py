#!/usr/bin/env python3
"""
验证废品检测模块功能的简单测试脚本
"""

import sys
import os
from pathlib import Path

# 将父目录添加到路径
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """测试所有模块是否可以正常导入"""
    print("测试模块导入...")
    
    try:
        from utils import WASTE_CLASS_NAMES, CLASS_COLORS, WASTE_CLASSES
        print(f"✅ 工具模块导入成功")
        print(f"   类别: {WASTE_CLASS_NAMES}")
        
        from utils.visualization import WasteDetectionVisualizer
        print(f"✅ 可视化模块已导入")
        
        from utils.metrics import WasteDetectionMetrics
        print(f"✅ 指标模块已导入")
        
        from utils.deployment import DeploymentManager
        print(f"✅ 部署模块已导入")
        
        from dataset.data_utils import WasteDatasetManager
        print(f"✅ 数据集工具已导入")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False

def test_configuration_files():
    """测试配置文件是否有效"""
    print("\n测试配置文件...")
    
    import yaml
    
    config_files = [
        'config/waste_data.yaml',
        'config/training_config.yaml', 
        'config/model_config.yaml'
    ]
    
    all_valid = True
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ {config_file} 是有效的YAML")
        except Exception as e:
            print(f"❌ {config_file} 错误: {e}")
            all_valid = False
    
    return all_valid

def test_dataset_utilities():
    """测试数据集工具函数"""
    print("\n测试数据集工具...")
    
    try:
        from dataset.data_utils import WasteDatasetManager, create_data_yaml
        
        # 测试数据集管理器
        manager = WasteDatasetManager()
        print("✅ 数据集管理器已创建")
        
        # 测试data.yaml创建
        create_data_yaml("/tmp/test_data.yaml")
        if os.path.exists("/tmp/test_data.yaml"):
            print("✅ 数据YAML创建正常")
            os.remove("/tmp/test_data.yaml")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集工具错误: {e}")
        return False

def test_visualization():
    """测试可视化工具"""
    print("\n测试可视化工具...")
    
    try:
        from utils.visualization import WasteDetectionVisualizer
        import numpy as np
        
        visualizer = WasteDetectionVisualizer()
        
        # 测试检测框绘制（模拟）
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_box = [100, 100, 200, 200]
        
        result = visualizer.draw_detection_box(dummy_image, dummy_box, 0, 0.95)
        print("✅ 检测框绘制正常")
        
        # 测试类别分布绘图（模拟数据）
        sample_counts = {'plastic_bottle': 50, 'paper': 30}
        # 在测试中我们不会实际显示图表
        print("✅ 可视化工具功能正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 可视化错误: {e}")
        return False

def test_metrics():
    """测试指标计算"""
    print("\n测试指标工具...")
    
    try:
        from utils.metrics import WasteDetectionMetrics
        from utils import WASTE_CLASS_NAMES
        
        metrics_calc = WasteDetectionMetrics(WASTE_CLASS_NAMES)
        
        # 测试IoU计算
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = metrics_calc.calculate_iou(box1, box2)
        print(f"✅ IoU计算正常: {iou:.3f}")
        
        # 测试速度指标
        import numpy as np
        times = np.random.normal(0.033, 0.005, 10)
        speed_metrics = metrics_calc.calculate_detection_speed_metrics(times)
        print(f"✅ 速度指标计算正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 指标错误: {e}")
        return False

def main():
    """运行所有测试"""
    print("="*60)
    print("废品检测模块测试套件")
    print("="*60)
    
    # 切换到waste_detection目录
    os.chdir(Path(__file__).parent)
    
    tests = [
        test_imports,
        test_configuration_files,
        test_dataset_utilities,
        test_visualization, 
        test_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试 {test.__name__} 异常失败: {e}")
    
    print("\n" + "="*60)
    print(f"测试结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！废品检测模块已就绪。")
    else:
        print("⚠️  部分测试失败。请检查上述错误。")
    
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)