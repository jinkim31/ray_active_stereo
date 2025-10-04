from phase_shift_pattern import PhaseShiftPattern
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Arguments
calib_images_dir = '../datasets/intrinsic/192.168.1.200'
result_dir = '../results/intrinsic_decode_maps/192.168.1.200'
lcd_resolution = [3840, 2160]

# Load images
images = []
for subdir in sorted(os.listdir(calib_images_dir)):
    if subdir.startswith('.'):
        continue
    subdir_images = [
        cv2.imread(os.path.join(calib_images_dir, subdir, filename), cv2.IMREAD_GRAYSCALE)
        for filename in sorted(os.listdir(os.path.join(calib_images_dir, subdir))) if filename.endswith(".png")]
    images.append(subdir_images)

for i, subdir_images in enumerate(images):
    # Decode
    phase_shift_pattern = PhaseShiftPattern(lcd_resolution)
    decode_maps = phase_shift_pattern.decode(subdir_images) # 한 pose에 대한 decode map. [H, W, 2(x,y순서)]. 0~1

    # Save decode maps
    decode_map_dir = os.path.join(result_dir, 'decode_maps')
    os.makedirs(decode_map_dir, exist_ok=True)
    np.save(os.path.join(decode_map_dir, f'{i:02d}.npy'), np.stack(decode_maps, axis=-1))

    # Save plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    im0 = axes[0, 0].imshow(subdir_images[0], cmap='gray')
    axes[0, 0].set_title('First Image')
    im1 = axes[0, 1].imshow(subdir_images[1], cmap='gray')
    axes[0, 1].set_title('Second Image')
    im2 = axes[1, 0].imshow(decode_maps[..., 0], cmap='jet')
    axes[1, 0].set_title('Decode Map X')
    cbar2 = plt.colorbar(im2, ax=axes[1, 0])
    cbar2.set_label('px')
    im3 = axes[1, 1].imshow(decode_maps[..., 1], cmap='jet')
    axes[1, 1].set_title('Decode Map Y')
    cbar3 = plt.colorbar(im3, ax=axes[1, 1])
    cbar3.set_label('px')
    plt.tight_layout()
    os.makedirs(os.path.join(result_dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(result_dir, 'plots', f'{i:02d}.png'))
    plt.close()

