# 废品检测模块

本模块为RecycleRover项目实现了基于YOLOv5的计算机视觉系统，用于检测和分类可回收废品材料。

## 概述

废品检测系统可以识别5类可回收材料：
1. `plastic_bottle` - 塑料瓶
2. `paper_box` - 纸盒/纸板
3. `metal_can` - 金属罐（铝/钢）
4. `glass_bottle` - 玻璃瓶
5. `paper` - 废纸/报纸

## 目标性能
- **准确率**: >95% 检测准确率
- **模型**: YOLOv5s（轻量化，<15MB）
- **部署**: 针对NVIDIA Jetson TX2优化

## 目录结构

```
waste_detection/
├── README.md                 # 本文件
├── requirements.txt          # Python依赖包
├── dataset/                  # 数据集管理
│   ├── images/              # 图像数据
│   │   ├── train/           # 训练图像（80%）
│   │   ├── val/             # 验证图像（10%）
│   │   └── test/            # 测试图像（10%）
│   ├── labels/              # YOLO格式标注文件
│   │   ├── train/           # 训练标签
│   │   ├── val/             # 验证标签
│   │   └── test/            # 测试标签
│   └── data_utils.py        # 数据集工具
├── config/                   # 配置文件
│   ├── waste_data.yaml      # 数据集配置
│   ├── model_config.yaml    # 模型配置
│   └── training_config.yaml # 训练参数配置
├── models/                   # 模型定义和权重
│   └── yolov5s_waste.yaml   # 自定义YOLOv5s配置
├── scripts/                  # 训练和推理脚本
│   ├── train.py             # 训练脚本
│   ├── evaluate.py          # 评估脚本
│   ├── inference.py         # 实时推理
│   └── export.py            # 模型导出（ONNX/TensorRT）
├── utils/                    # 工具函数
│   ├── __init__.py
│   ├── visualization.py     # 结果可视化
│   ├── metrics.py           # 性能指标计算
│   └── deployment.py        # 部署工具
└── deployment/               # 部署配置
    ├── jetson_setup.md      # Jetson TX2安装指南
    ├── tensorrt_config.py   # TensorRT优化配置
    └── inference_service.py # 部署服务脚本
```

## 快速开始

1. **环境设置**
   ```bash
   cd waste_detection
   pip install -r requirements.txt
   ```

2. **数据集准备**
   ```bash
   python dataset/data_utils.py --prepare
   ```

3. **训练**
   ```bash
   python scripts/train.py --config config/training_config.yaml
   ```

4. **评估**
   ```bash
   python scripts/evaluate.py --weights models/best.pt
   ```

5. **推理**
   ```bash
   python scripts/inference.py --source 0  # 摄像头
   python scripts/inference.py --source path/to/image.jpg
   ```

## 与RecycleRover的集成

废品检测模块通过以下方式与主RecycleRover系统集成：
- **ROS接口**: 将检测结果发布到ROS话题
- **MQTT通信**: 向云端平台发送检测数据
- **实时处理**: 处理无人车视觉系统的摄像头画面

## 性能目标

- **推理速度**: 在Jetson TX2上>30 FPS
- **模型大小**: <15MB用于部署
- **准确率**: >95% mAP@0.5
- **功耗效率**: 针对移动部署优化