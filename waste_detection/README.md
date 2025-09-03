# Waste Detection Module

This module implements a YOLOv5-based computer vision system for detecting and classifying recyclable waste materials in the RecycleRover project.

## Overview

The waste detection system can identify 5 categories of recyclable materials:
1. `plastic_bottle` - Plastic bottles
2. `paper_box` - Paper boxes/cardboard
3. `metal_can` - Metal cans (aluminum/steel)
4. `glass_bottle` - Glass bottles
5. `paper` - Waste paper/newspapers

## Target Performance
- **Accuracy**: >95% detection accuracy
- **Model**: YOLOv5s (lightweight, <15MB)
- **Deployment**: Optimized for NVIDIA Jetson TX2

## Directory Structure

```
waste_detection/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── dataset/                  # Dataset management
│   ├── images/              # Image data
│   │   ├── train/           # Training images (80%)
│   │   ├── val/             # Validation images (10%)
│   │   └── test/            # Test images (10%)
│   ├── labels/              # YOLO format annotations
│   │   ├── train/           # Training labels
│   │   ├── val/             # Validation labels
│   │   └── test/            # Test labels
│   └── data_utils.py        # Dataset utilities
├── config/                   # Configuration files
│   ├── waste_data.yaml      # Dataset configuration
│   ├── model_config.yaml    # Model configuration
│   └── training_config.yaml # Training parameters
├── models/                   # Model definitions and weights
│   └── yolov5s_waste.yaml   # Custom YOLOv5s configuration
├── scripts/                  # Training and inference scripts
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   ├── inference.py         # Real-time inference
│   └── export.py            # Model export (ONNX/TensorRT)
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── visualization.py     # Result visualization
│   ├── metrics.py           # Performance metrics
│   └── deployment.py        # Deployment utilities
└── deployment/               # Deployment configurations
    ├── jetson_setup.md      # Jetson TX2 setup guide
    ├── tensorrt_config.py   # TensorRT optimization
    └── inference_service.py # Deployment service
```

## Quick Start

1. **Environment Setup**
   ```bash
   cd waste_detection
   pip install -r requirements.txt
   ```

2. **Dataset Preparation**
   ```bash
   python dataset/data_utils.py --prepare
   ```

3. **Training**
   ```bash
   python scripts/train.py --config config/training_config.yaml
   ```

4. **Evaluation**
   ```bash
   python scripts/evaluate.py --weights models/best.pt
   ```

5. **Inference**
   ```bash
   python scripts/inference.py --source 0  # webcam
   python scripts/inference.py --source path/to/image.jpg
   ```

## Integration with RecycleRover

The waste detection module integrates with the main RecycleRover system through:
- **ROS Interface**: Publishes detection results to ROS topics
- **MQTT Communication**: Sends detection data to cloud platform
- **Real-time Processing**: Processes camera feed from rover's vision system

## Performance Targets

- **Inference Speed**: >30 FPS on Jetson TX2
- **Model Size**: <15MB for deployment
- **Accuracy**: >95% mAP@0.5
- **Power Efficiency**: Optimized for mobile deployment