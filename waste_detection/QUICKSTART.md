# Quick Start Guide - Waste Detection Module

This guide provides quick setup instructions for the RecycleRover waste detection system.

## 🚀 Quick Setup

### 1. Environment Setup
```bash
cd waste_detection

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
# Create dataset structure
python dataset/data_utils.py --prepare

# Create YOLO configuration
python dataset/data_utils.py --create-yaml

# If you have existing data, split it:
# python dataset/data_utils.py --split /path/to/images /path/to/labels
```

### 3. Train Model
```bash
# Start training with default configuration
python scripts/train.py

# Or with custom config:
python scripts/train.py --config config/training_config.yaml --data config/waste_data.yaml
```

### 4. Evaluate Model
```bash
# Evaluate trained model
python scripts/evaluate.py --model runs/train/waste_detection/weights/best.pt

# Generate comprehensive report
python scripts/evaluate.py --model runs/train/waste_detection/weights/best.pt --report
```

### 5. Export for Deployment
```bash
# Export to all formats
python scripts/export.py --model runs/train/waste_detection/weights/best.pt --format all

# Export TensorRT for Jetson
python scripts/export.py --model runs/train/waste_detection/weights/best.pt --format tensorrt --precision fp16
```

### 6. Run Inference
```bash
# Test with webcam
python scripts/inference.py --model exported_models/best.engine --source 0

# Test with image
python scripts/inference.py --model exported_models/best.engine --source test_image.jpg

# Benchmark performance
python scripts/inference.py --model exported_models/best.engine --source test_image.jpg --benchmark
```

## 🎯 Key Features

- **5-Class Detection**: plastic_bottle, paper_box, metal_can, glass_bottle, paper
- **High Performance**: >95% mAP target, >30 FPS on Jetson TX2
- **Multiple Export Formats**: PyTorch, ONNX, TensorRT
- **Production Ready**: MQTT integration, deployment service
- **Comprehensive Tools**: Training, evaluation, visualization, deployment

## 📊 Expected Results

After training with sufficient data (1000+ images per class), expect:
- **mAP@0.5**: >95%
- **Inference Speed**: 30+ FPS on Jetson TX2
- **Model Size**: <15MB (TensorRT)
- **Memory Usage**: <2GB

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `config/training_config.yaml`
   - Use smaller input size (320 instead of 640)

2. **Low Performance**
   - Collect more diverse training data
   - Improve annotation quality
   - Adjust hyperparameters

3. **Deployment Issues**
   - Check Jetson setup guide: `deployment/jetson_setup.md`
   - Verify TensorRT installation
   - Use deployment utilities: `python utils/deployment.py --check`

### Getting Help

1. Check configuration files in `config/`
2. Review logs in `runs/train/` directory
3. Use visualization tools for debugging
4. Run tests: `python test_module.py`

## 🎓 Advanced Usage

### Custom Training
```bash
# Modify hyperparameters in config/training_config.yaml
# Adjust data augmentation settings
# Change model architecture in config/model_config.yaml
```

### Production Deployment
```bash
# Create deployment package
python utils/deployment.py --package . /path/to/output

# Run production service
python deployment/inference_service.py --config deployment/service_config.json
```

### Model Optimization
```bash
# Optimize for Jetson TX2
python deployment/tensorrt_config.py

# Benchmark different formats
python scripts/export.py --model best.pt --format all --benchmark
```

This system provides a complete solution for waste detection from data preparation to production deployment! 🎉