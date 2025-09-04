"""
废品检测模块工具包
"""

__version__ = "1.0.0"
__author__ = "RecycleRover Team"

# 废品检测类别定义
WASTE_CLASSES = {
    0: 'plastic_bottle',
    1: 'paper_box', 
    2: 'metal_can',
    3: 'glass_bottle',
    4: 'paper'
}

WASTE_CLASS_NAMES = list(WASTE_CLASSES.values())

# 可视化颜色映射 (OpenCV BGR格式)
CLASS_COLORS = {
    'plastic_bottle': (255, 0, 0),    # 蓝色
    'paper_box': (0, 255, 0),         # 绿色
    'metal_can': (0, 0, 255),         # 红色
    'glass_bottle': (255, 255, 0),    # 青色
    'paper': (255, 0, 255)            # 品红色
}

# 模型配置常量
DEFAULT_IMG_SIZE = 640
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

# 数据集分割比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1