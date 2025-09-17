from phase_shift_pattern import PhaseShiftPattern
from util import *
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Arguments
calib_images_dir = "../datasets/intrinsic_images/192.168.0.2"
result_dir = "../results/intrinsic_decode_maps/192.168.0.2"
lcd_resolution = [3840, 2160]

# Load images
images = []
for subdir in os.listdir(calib_images_dir):
    subdir_images = [
        cv2.imread(os.path.join(calib_images_dir, subdir, filename), cv2.IMREAD_GRAYSCALE)
        for filename in os.listdir(os.path.join(calib_images_dir, subdir))]
    images.append(subdir_images)

for i, subdir_images in enumerate(images):
    # Decode
    phase_shift_pattern = PhaseShiftPattern(lcd_resolution)
    x, y = phase_shift_pattern.decode(subdir_images)

    # Save decode maps
    decode_map_dir = os.path.join(result_dir, 'decode_maps', f'{i:02d}')
    os.makedirs(decode_map_dir, exist_ok=True)
    np.save(os.path.join(decode_map_dir, 'x.npy'), x)
    np.save(os.path.join(decode_map_dir, 'y.npy'), y)

    # Save plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    im0 = axes[0, 0].imshow(subdir_images[0], cmap='gray')
    axes[0, 0].set_title('First Image')
    im1 = axes[0, 1].imshow(subdir_images[1], cmap='gray')
    axes[0, 1].set_title('Second Image')
    im2 = axes[1, 0].imshow(x, cmap='jet')
    axes[1, 0].set_title('Decode Map X')
    cbar2 = plt.colorbar(im2, ax=axes[1, 0])
    cbar2.set_label('px')
    im3 = axes[1, 1].imshow(y, cmap='jet')
    axes[1, 1].set_title('Decode Map Y')
    cbar3 = plt.colorbar(im3, ax=axes[1, 1])
    cbar3.set_label('px')
    plt.tight_layout()
    os.makedirs(os.path.join(result_dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(result_dir, 'plots', f'{i:02d}.png'))
    plt.close()

