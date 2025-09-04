# Jetson TX2 废品检测系统安装指南

本指南提供在NVIDIA Jetson TX2上设置废品检测系统的详细步骤。

## 前提条件

- NVIDIA Jetson TX2开发套件
- MicroSD卡（推荐64GB+）
- 用于刷机的主机电脑
- 网络连接

## 1. 系统设置

### 1.1 刷入JetPack

1. **下载NVIDIA SDK Manager**
   - 访问 https://developer.nvidia.com/nvidia-sdk-manager
   - 在主机电脑上下载并安装SDK Manager

2. **刷入JetPack 4.6.1**（推荐用于兼容性）
   ```bash
   # 将Jetson TX2设置为恢复模式
   # 通过USB线缆连接到主机
   # 运行SDK Manager并按照说明操作
   ```

3. **在Jetson TX2上完成初始设置**

### 1.2 系统配置

1. **Increase swap space** (important for compilation)
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

2. **Set power mode** for maximum performance
   ```bash
   sudo nvpmodel -m 0  # Max performance mode
   sudo jetson_clocks   # Enable max clocks
   ```

3. **Install system dependencies**
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-dev python3-setuptools
   sudo apt install -y cmake build-essential pkg-config
   sudo apt install -y libjpeg-dev libtiff5-dev libpng-dev
   sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
   sudo apt install -y libgtk2.0-dev libcanberra-gtk-module
   sudo apt install -y libxvidcore-dev libx264-dev
   sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
   ```

## 2. Python Environment Setup

### 2.1 Create Virtual Environment

```bash
# Install virtual environment
sudo pip3 install virtualenv

# Create environment for waste detection
cd /home/$USER
python3 -m venv waste_detection_env
source waste_detection_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2.2 Install PyTorch for Jetson

```bash
# Download PyTorch wheel for Jetson (JetPack 4.6.1)
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install PyTorch
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install torchvision
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python setup.py install --user
cd ..
```

### 2.3 Install OpenCV

```bash
# OpenCV should be pre-installed with JetPack
# Verify installation
python3 -c "import cv2; print(cv2.__version__)"

# If not installed, build from source (time-consuming)
# Or use: pip install opencv-python
```

## 3. Waste Detection Installation

### 3.1 Clone Repository

```bash
git clone <repository-url>
cd RecycleRover/waste_detection
```

### 3.2 Install Dependencies

```bash
# Activate virtual environment
source ~/waste_detection_env/bin/activate

# Install basic requirements
pip install numpy matplotlib pillow pyyaml tqdm requests scipy pandas seaborn

# Install ultralytics (for YOLOv5)
pip install ultralytics

# Install ONNX runtime for inference optimization
pip install onnxruntime
```

### 3.3 Install TensorRT Python API

```bash
# TensorRT should be included with JetPack
# Install Python bindings
cd /usr/src/tensorrt/samples/python/python_plugin
make clean && make
sudo cp build/libnvplugins.so /usr/lib/aarch64-linux-gnu/

# Verify TensorRT installation
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

## 4. Model Deployment

### 4.1 Transfer Model Files

```bash
# Copy trained model from development machine
scp user@dev-machine:/path/to/best.pt ~/RecycleRover/waste_detection/models/

# Or download from cloud storage
wget <model-download-url> -O models/waste_detection_best.pt
```

### 4.2 Convert Model to TensorRT

```bash
cd ~/RecycleRover/waste_detection

# Export to ONNX first
python scripts/export.py --model models/waste_detection_best.pt --format onnx

# Convert to TensorRT engine
python scripts/export.py --model models/waste_detection_best.pt --format tensorrt --precision fp16
```

### 4.3 Test Inference

```bash
# Test with sample image
python scripts/inference.py --model models/waste_detection_best.engine --source test_image.jpg

# Test with webcam
python scripts/inference.py --model models/waste_detection_best.engine --source 0

# Benchmark performance
python scripts/inference.py --model models/waste_detection_best.engine --source test_image.jpg --benchmark
```

## 5. Performance Optimization

### 5.1 TensorRT Optimization Settings

```python
# Optimal TensorRT settings for Jetson TX2
TENSORRT_CONFIG = {
    'precision': 'fp16',           # Use FP16 for speed
    'max_batch_size': 1,           # Single image inference
    'workspace_size': '1GB',       # Limit workspace for TX2
    'enable_dla': False,           # TX2 doesn't have DLA
    'strict_type_constraints': True
}
```

### 5.2 Inference Optimization

```bash
# Set CPU performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable GUI for headless operation (optional)
sudo systemctl set-default multi-user.target

# Monitor system resources
sudo apt install htop iotop
htop  # Monitor CPU/memory usage during inference
```

## 6. Integration Setup

### 6.1 ROS Integration (if needed)

```bash
# Install ROS Melodic (compatible with Ubuntu 18.04)
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-melodic-desktop-base

# Setup ROS environment
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install additional ROS packages
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
sudo rosdep init
rosdep update
```

### 6.2 MQTT Communication

```bash
pip install paho-mqtt

# Test MQTT connection
python3 -c "import paho.mqtt.client as mqtt; print('MQTT client available')"
```

## 7. Autostart Configuration

### 7.1 Create Service File

```bash
sudo nano /etc/systemd/system/waste-detection.service
```

```ini
[Unit]
Description=Waste Detection Service
After=network.target

[Service]
Type=simple
User=nvidia
WorkingDirectory=/home/nvidia/RecycleRover/waste_detection
Environment=PATH=/home/nvidia/waste_detection_env/bin
ExecStart=/home/nvidia/waste_detection_env/bin/python scripts/inference.py --model models/waste_detection_best.engine --source 0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 7.2 Enable Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable waste-detection.service
sudo systemctl start waste-detection.service

# Check status
sudo systemctl status waste-detection.service
```

## 8. Troubleshooting

### 8.1 Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size or image resolution
# Monitor GPU memory: watch -n 1 nvidia-smi
```

**Slow Inference**
```bash
# Ensure TensorRT engine is used
# Check power mode: sudo nvpmodel -q
# Verify clocks: sudo jetson_clocks --show
```

**Import Errors**
```bash
# Check virtual environment activation
# Verify package installation: pip list
# Check CUDA/cuDNN installation: python -c "import torch; print(torch.cuda.is_available())"
```

### 8.2 Performance Monitoring

```bash
# Monitor system resources
sudo apt install jetson-stats
sudo -H pip install jetson-stats
jtop  # Real-time system monitoring

# Monitor inference performance
python scripts/inference.py --model models/best.engine --source 0 --benchmark
```

## 9. Deployment Checklist

- [ ] JetPack 4.6.1 installed and configured
- [ ] Python virtual environment created
- [ ] PyTorch and dependencies installed
- [ ] TensorRT working and model converted
- [ ] Inference speed >30 FPS achieved
- [ ] ROS integration configured (if needed)
- [ ] MQTT communication tested
- [ ] Autostart service configured
- [ ] System monitoring tools installed
- [ ] Performance benchmarks documented

## Performance Targets for Jetson TX2

- **Inference Speed**: >30 FPS @ 640x640 input
- **Model Size**: <15MB (TensorRT engine)
- **Memory Usage**: <2GB total system memory
- **Power Consumption**: <15W during inference
- **Accuracy**: >95% mAP@0.5 (same as training)

## Support

For technical support and troubleshooting:
1. Check system logs: `journalctl -u waste-detection.service`
2. Monitor performance: `jtop`
3. Review NVIDIA developer documentation
4. Contact development team