# 快速开始指南 - 废品检测模块

本指南为RecycleRover废品检测系统提供快速设置说明。

## 🚀 快速设置

### 1. 环境设置
```bash
cd waste_detection

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Windows系统: venv\Scripts\activate

# 安装依赖包
pip install -r requirements.txt
```

### 2. 准备数据集
```bash
# 创建数据集结构
python dataset/data_utils.py --prepare

# 创建YOLO配置
python dataset/data_utils.py --create-yaml

# 如果您有现有数据，请分割它：
# python dataset/data_utils.py --split /path/to/images /path/to/labels
```

### 3. 训练模型
```bash
# 使用默认配置开始训练
python scripts/train.py

# 或使用自定义配置：
python scripts/train.py --config config/training_config.yaml --data config/waste_data.yaml
```

### 4. 评估模型
```bash
# 评估训练好的模型
python scripts/evaluate.py --model runs/train/waste_detection/weights/best.pt

# 生成综合报告
python scripts/evaluate.py --model runs/train/waste_detection/weights/best.pt --report
```

### 5. 导出用于部署
```bash
# 导出到所有格式
python scripts/export.py --model runs/train/waste_detection/weights/best.pt --format all

# 为Jetson导出TensorRT
python scripts/export.py --model runs/train/waste_detection/weights/best.pt --format tensorrt --precision fp16
```

### 6. 运行推理
```bash
# 使用摄像头测试
python scripts/inference.py --model exported_models/best.engine --source 0

# 使用图像测试
python scripts/inference.py --model exported_models/best.engine --source test_image.jpg

# 性能基准测试
python scripts/inference.py --model exported_models/best.engine --source test_image.jpg --benchmark
```

## 🎯 主要特性

- **5类检测**: plastic_bottle, paper_box, metal_can, glass_bottle, paper
- **高性能**: >95% mAP目标，在Jetson TX2上>30 FPS
- **多种导出格式**: PyTorch, ONNX, TensorRT
- **生产就绪**: MQTT集成，部署服务
- **完整工具**: 训练、评估、可视化、部署

## 📊 预期结果

使用充足数据训练后（每类1000+图像），预期：
- **mAP@0.5**: >95%
- **推理速度**: 在Jetson TX2上30+ FPS
- **模型大小**: <15MB（TensorRT）
- **内存使用**: <2GB

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   - 在`config/training_config.yaml`中减少批量大小
   - 使用更小的输入尺寸（320而不是640）

2. **性能低下**
   - 收集更多样化的训练数据
   - 提高标注质量
   - 调整超参数

3. **部署问题**
   - 查看Jetson设置指南：`deployment/jetson_setup.md`
   - 验证TensorRT安装
   - 使用部署工具：`python utils/deployment.py --check`

### 获取帮助

1. 检查`config/`中的配置文件
2. 查看`runs/train/`目录中的日志
3. 使用可视化工具进行调试
4. 运行测试：`python test_module.py`

## 🎓 高级用法

### 自定义训练
```bash
# 在config/training_config.yaml中修改超参数
# 调整数据增强设置
# 在config/model_config.yaml中更改模型架构
```

### 生产部署
```bash
# 创建部署包
python utils/deployment.py --package . /path/to/output

# 运行生产服务
python deployment/inference_service.py --config deployment/service_config.json
```

### 模型优化
```bash
# 为Jetson TX2优化
python deployment/tensorrt_config.py

# 对不同格式进行基准测试
python scripts/export.py --model best.pt --format all --benchmark
```

本系统提供从数据准备到生产部署的完整废品检测解决方案！🎉