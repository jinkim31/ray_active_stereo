from phase_shift_pattern import PhaseShiftPattern
from util import *
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ray_calibration import *

# Arguments
decode_maps_dir = "../results/intrinsic_decode_maps/192.168.0.2"
result_dir = "../results/intrinsics/192.168.0.2"
lcd_resolution = [3840, 2160]

# Load images
decode_maps = []
decode_masks = []
for subdir in os.listdir(os.path.join(decode_maps_dir, 'decode_maps')):
    x = os.path.join(decode_maps_dir, 'decode_maps', subdir, 'x')
    y = os.path.join(decode_maps_dir, 'decode_maps', subdir, 'y')
    decode_maps.append([x, y])
    decode_masks.append([x > 0, y>0])

# Calibrate
ray_calibrate(decode_maps, decode_masks, target_size=lcd_resolution)
