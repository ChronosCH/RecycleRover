"""
Waste Detection Module Utilities
"""

__version__ = "1.0.0"
__author__ = "RecycleRover Team"

# Waste detection class definitions
WASTE_CLASSES = {
    0: 'plastic_bottle',
    1: 'paper_box', 
    2: 'metal_can',
    3: 'glass_bottle',
    4: 'paper'
}

WASTE_CLASS_NAMES = list(WASTE_CLASSES.values())

# Color map for visualization (BGR format for OpenCV)
CLASS_COLORS = {
    'plastic_bottle': (255, 0, 0),    # Blue
    'paper_box': (0, 255, 0),         # Green
    'metal_can': (0, 0, 255),         # Red
    'glass_bottle': (255, 255, 0),    # Cyan
    'paper': (255, 0, 255)            # Magenta
}

# Model configuration constants
DEFAULT_IMG_SIZE = 640
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

# Dataset split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1