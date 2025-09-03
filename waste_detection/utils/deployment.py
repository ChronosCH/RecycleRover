"""
Deployment utilities for waste detection system
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union
import platform
import psutil

from utils import WASTE_CLASS_NAMES


class DeploymentManager:
    """Manages deployment of waste detection system"""
    
    def __init__(self, target_platform: str = "jetson_tx2"):
        """
        Initialize deployment manager
        
        Args:
            target_platform: Target deployment platform
        """
        self.target_platform = target_platform
        self.system_info = self.get_system_info()
        
    def get_system_info(self) -> Dict:
        """Get system information for deployment planning"""
        info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
        }
        
        # Check for CUDA
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            info['cuda_available'] = result.returncode == 0
            if info['cuda_available']:
                # Parse GPU info from nvidia-smi
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'CUDA Version' in line:
                        cuda_version = line.split('CUDA Version: ')[1].split()[0]
                        info['cuda_version'] = cuda_version
                        break
        except FileNotFoundError:
            info['cuda_available'] = False
        
        # Check for TensorRT
        try:
            import tensorrt
            info['tensorrt_available'] = True
            info['tensorrt_version'] = tensorrt.__version__
        except ImportError:
            info['tensorrt_available'] = False
        
        return info
    
    def check_deployment_requirements(self) -> Dict[str, bool]:
        """Check if deployment requirements are met"""
        requirements = {
            'python_version': self.system_info['python_version'] >= '3.6',
            'cuda_available': self.system_info.get('cuda_available', False),
            'tensorrt_available': self.system_info.get('tensorrt_available', False),
            'sufficient_memory': self.system_info['memory_gb'] >= 4.0,
            'sufficient_storage': self.check_storage_space(),
        }
        
        # Platform-specific requirements
        if self.target_platform == "jetson_tx2":
            requirements.update({
                'arm_architecture': self.system_info['machine'].startswith('aarch64'),
                'jetpack_installed': self.check_jetpack_installation(),
            })
        
        return requirements
    
    def check_storage_space(self, required_gb: float = 10.0) -> bool:
        """Check if sufficient storage space is available"""
        try:
            usage = shutil.disk_usage('/')
            free_gb = usage.free / (1024**3)
            return free_gb >= required_gb
        except:
            return False
    
    def check_jetpack_installation(self) -> bool:
        """Check if JetPack is properly installed"""
        jetpack_indicators = [
            '/usr/local/cuda/bin/nvcc',
            '/usr/src/tensorrt',
            '/etc/nv_tegra_release'
        ]
        
        return any(os.path.exists(path) for path in jetpack_indicators)
    
    def install_dependencies(self, requirements_file: str = "requirements.txt") -> bool:
        """Install Python dependencies"""
        if not os.path.exists(requirements_file):
            print(f"Requirements file not found: {requirements_file}")
            return False
        
        try:
            print(f"Installing dependencies from {requirements_file}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', requirements_file
            ], check=True, capture_output=True, text=True)
            
            print("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def optimize_for_jetson(self, config: Dict) -> Dict:
        """Optimize configuration for Jetson platform"""
        optimized_config = config.copy()
        
        # Memory optimizations
        optimized_config.setdefault('model', {})
        optimized_config['model']['batch_size'] = 1  # Single inference
        optimized_config['model']['workers'] = 2  # Limit workers
        
        # Inference optimizations
        optimized_config.setdefault('inference', {})
        optimized_config['inference']['precision'] = 'fp16'
        optimized_config['inference']['use_tensorrt'] = True
        optimized_config['inference']['optimize_dla'] = False  # TX2 doesn't have DLA
        
        # Camera settings
        optimized_config.setdefault('camera', {})
        optimized_config['camera']['resolution'] = [1280, 720]  # Balanced resolution
        optimized_config['camera']['fps'] = 30
        
        # System settings
        optimized_config.setdefault('system', {})
        optimized_config['system']['power_mode'] = 'MAXN'  # Maximum performance
        optimized_config['system']['enable_jetson_clocks'] = True
        
        return optimized_config
    
    def create_deployment_package(self, source_dir: str, output_path: str,
                                 include_models: bool = True) -> bool:
        """Create deployment package"""
        print(f"Creating deployment package from {source_dir}...")
        
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Source directory not found: {source_dir}")
            return False
        
        # Create temporary directory for package
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "waste_detection_deployment"
            package_dir.mkdir()
            
            # Copy essential files
            essential_dirs = ['config', 'scripts', 'utils', 'deployment']
            essential_files = ['requirements.txt', 'README.md']
            
            for dir_name in essential_dirs:
                src_dir = source_path / dir_name
                if src_dir.exists():
                    shutil.copytree(src_dir, package_dir / dir_name)
            
            for file_name in essential_files:
                src_file = source_path / file_name
                if src_file.exists():
                    shutil.copy2(src_file, package_dir / file_name)
            
            # Copy models if requested
            if include_models:
                models_dir = source_path / 'models'
                if models_dir.exists():
                    shutil.copytree(models_dir, package_dir / 'models')
            
            # Create deployment script
            self.create_deployment_script(package_dir)
            
            # Create archive
            shutil.make_archive(output_path, 'zip', temp_dir, 'waste_detection_deployment')
            
        print(f"Deployment package created: {output_path}.zip")
        return True
    
    def create_deployment_script(self, package_dir: Path):
        """Create automated deployment script"""
        script_content = f'''#!/bin/bash
# Waste Detection System Deployment Script
# Generated for {self.target_platform}

set -e

echo "Waste Detection System Deployment"
echo "================================="

# Check system requirements
echo "Checking system requirements..."

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python3 not found"
    exit 1
fi

# Check CUDA (if available)
nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "CUDA detected"
else
    echo "Warning: CUDA not detected"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p runs
mkdir -p tmp

# Set executable permissions
chmod +x scripts/*.py
chmod +x deployment/*.py

# Platform-specific setup
if [ "{self.target_platform}" = "jetson_tx2" ]; then
    echo "Configuring for Jetson TX2..."
    
    # Set power mode
    sudo nvpmodel -m 0 || echo "Warning: Could not set power mode"
    
    # Enable max clocks
    sudo jetson_clocks || echo "Warning: Could not enable jetson_clocks"
    
    # Install additional Jetson packages
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-dev
fi

echo "Deployment completed successfully!"
echo ""
echo "To start the service:"
echo "  python3 deployment/inference_service.py"
echo ""
echo "To test inference:"
echo "  python3 scripts/inference.py --model models/best.engine --source 0"
'''
        
        script_path = package_dir / 'deploy.sh'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
    
    def verify_deployment(self, deployment_dir: str) -> Dict[str, bool]:
        """Verify deployment integrity"""
        deployment_path = Path(deployment_dir)
        
        checks = {
            'directory_exists': deployment_path.exists(),
            'config_files': (deployment_path / 'config').exists(),
            'scripts_exist': (deployment_path / 'scripts').exists(),
            'requirements_file': (deployment_path / 'requirements.txt').exists(),
            'deployment_tools': (deployment_path / 'deployment').exists(),
        }
        
        # Check if model files exist
        models_dir = deployment_path / 'models'
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pt')) + list(models_dir.glob('*.engine'))
            checks['model_files'] = len(model_files) > 0
        else:
            checks['model_files'] = False
        
        # Check Python imports
        try:
            sys.path.insert(0, str(deployment_path))
            import utils
            checks['utils_import'] = True
        except ImportError:
            checks['utils_import'] = False
        finally:
            if str(deployment_path) in sys.path:
                sys.path.remove(str(deployment_path))
        
        return checks
    
    def generate_deployment_report(self, save_path: str = None) -> str:
        """Generate deployment readiness report"""
        requirements = self.check_deployment_requirements()
        
        report = []
        report.append("="*60)
        report.append("DEPLOYMENT READINESS REPORT")
        report.append("="*60)
        
        report.append(f"\nTarget Platform: {self.target_platform}")
        report.append(f"Current System:")
        for key, value in self.system_info.items():
            report.append(f"  {key}: {value}")
        
        report.append(f"\nRequirement Checks:")
        all_passed = True
        for requirement, passed in requirements.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report.append(f"  {requirement}: {status}")
            if not passed:
                all_passed = False
        
        report.append(f"\nOverall Status: {'✅ READY' if all_passed else '❌ NOT READY'}")
        
        if not all_passed:
            report.append(f"\nRecommendations:")
            if not requirements.get('cuda_available', True):
                report.append("  - Install NVIDIA drivers and CUDA toolkit")
            if not requirements.get('tensorrt_available', True):
                report.append("  - Install TensorRT for inference optimization")
            if not requirements.get('sufficient_memory', True):
                report.append("  - Increase system memory or enable swap")
            if not requirements.get('jetpack_installed', True):
                report.append("  - Install NVIDIA JetPack for Jetson platforms")
        
        report.append("="*60)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


def main():
    """Demo deployment utilities"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deployment utilities")
    parser.add_argument('--platform', default='jetson_tx2', 
                       help='Target deployment platform')
    parser.add_argument('--check', action='store_true',
                       help='Check deployment requirements')
    parser.add_argument('--package', nargs=2, metavar=('SOURCE', 'OUTPUT'),
                       help='Create deployment package')
    parser.add_argument('--verify', help='Verify deployment directory')
    
    args = parser.parse_args()
    
    manager = DeploymentManager(args.platform)
    
    if args.check:
        report = manager.generate_deployment_report()
        print(report)
    
    elif args.package:
        source_dir, output_path = args.package
        success = manager.create_deployment_package(source_dir, output_path)
        if not success:
            return 1
    
    elif args.verify:
        checks = manager.verify_deployment(args.verify)
        print("Deployment Verification:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {check}: {status}")
    
    else:
        print("Use --help for available options")


if __name__ == "__main__":
    main()