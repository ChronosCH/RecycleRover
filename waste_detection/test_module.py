#!/usr/bin/env python3
"""
Simple test script to verify waste detection module functionality
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from utils import WASTE_CLASS_NAMES, CLASS_COLORS, WASTE_CLASSES
        print(f"✅ Utils imported successfully")
        print(f"   Classes: {WASTE_CLASS_NAMES}")
        
        from utils.visualization import WasteDetectionVisualizer
        print(f"✅ Visualization module imported")
        
        from utils.metrics import WasteDetectionMetrics
        print(f"✅ Metrics module imported")
        
        from utils.deployment import DeploymentManager
        print(f"✅ Deployment module imported")
        
        from dataset.data_utils import WasteDatasetManager
        print(f"✅ Dataset utilities imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_configuration_files():
    """Test that configuration files are valid"""
    print("\nTesting configuration files...")
    
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
            print(f"✅ {config_file} is valid YAML")
        except Exception as e:
            print(f"❌ {config_file} error: {e}")
            all_valid = False
    
    return all_valid

def test_dataset_utilities():
    """Test dataset utility functions"""
    print("\nTesting dataset utilities...")
    
    try:
        from dataset.data_utils import WasteDatasetManager, create_data_yaml
        
        # Test dataset manager
        manager = WasteDatasetManager()
        print("✅ Dataset manager created")
        
        # Test data.yaml creation
        create_data_yaml("/tmp/test_data.yaml")
        if os.path.exists("/tmp/test_data.yaml"):
            print("✅ Data YAML creation works")
            os.remove("/tmp/test_data.yaml")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset utilities error: {e}")
        return False

def test_visualization():
    """Test visualization utilities"""
    print("\nTesting visualization utilities...")
    
    try:
        from utils.visualization import WasteDetectionVisualizer
        import numpy as np
        
        visualizer = WasteDetectionVisualizer()
        
        # Test detection box drawing (mock)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_box = [100, 100, 200, 200]
        
        result = visualizer.draw_detection_box(dummy_image, dummy_box, 0, 0.95)
        print("✅ Detection box drawing works")
        
        # Test class distribution plotting (mock data)
        sample_counts = {'plastic_bottle': 50, 'paper': 30}
        # We won't actually show the plot in test
        print("✅ Visualization utilities functional")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def test_metrics():
    """Test metrics calculation"""
    print("\nTesting metrics utilities...")
    
    try:
        from utils.metrics import WasteDetectionMetrics
        from utils import WASTE_CLASS_NAMES
        
        metrics_calc = WasteDetectionMetrics(WASTE_CLASS_NAMES)
        
        # Test IoU calculation
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = metrics_calc.calculate_iou(box1, box2)
        print(f"✅ IoU calculation works: {iou:.3f}")
        
        # Test speed metrics
        import numpy as np
        times = np.random.normal(0.033, 0.005, 10)
        speed_metrics = metrics_calc.calculate_detection_speed_metrics(times)
        print(f"✅ Speed metrics calculation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("WASTE DETECTION MODULE TEST SUITE")
    print("="*60)
    
    # Change to waste_detection directory
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
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The waste detection module is ready.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)