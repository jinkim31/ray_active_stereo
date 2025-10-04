import cv2
import gray_code
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

import transform
import util


# Parameters
projector_resolution = [1920, 1080]
dataset_dir = '../datasets/measurement/bright'
intrinsic_dir = '../results/intrinsic_results'
extrinsic_dir = '../results/extrinsic_results'
result_dir = '../results/measurement_results'


if __name__ == '__main__':

    # Handler parameters
    cam_names = sorted(os.listdir(dataset_dir))
    n_cams = len(cam_names)
    os.makedirs(result_dir, exist_ok=True)

    # Decode graycode
    def decode_graycode():

        # Load images
        images = [[] for _ in range(n_cams)]
        for i_cam, cam_name in enumerate(cam_names):
            for file_path in sorted(glob.glob(os.path.join(dataset_dir, cam_name, '*.png'))):
                images[i_cam].append(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE))
        image_size = images[0][0].shape[::-1]

        decode_maps = []
        graycode = cv2.structured_light_GrayCodePattern.create(*projector_resolution)
        for i_cam, cam_name in enumerate(cam_names):
            decode_maps.append(gray_code.decode(graycode, images[i_cam]))

            # Save plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            im1 = axes[0].imshow(decode_maps[-1][0], cmap='jet', aspect='auto')
            axes[0].set_title('X')
            fig.colorbar(im1, ax=axes[0])  # add colorbar
            im2 = axes[1].imshow(decode_maps[-1][1], cmap='jet', aspect='auto')
            axes[1].set_title('Y')
            fig.colorbar(im2, ax=axes[1])  # add colorbar
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, f'{cam_name}_decode_map.png'))
            plt.close()

        return decode_maps, image_size
    decode_maps, image_size = util.pickle_get('gray_decode_map.pkl', decode_graycode)

    # Find correspondences
    cam_0_points, cam_1_points = util.pickle_get(
        'correspondences.pkl', lambda: gray_code.stereo_match(decode_maps[0], decode_maps[1]))

    # Load ray parameters
    ray_parameters = []
    for cam_name in cam_names:
        parameters = np.load(os.path.join(intrinsic_dir, cam_name, 'parameters.npy'))
        parameters = parameters.reshape(*image_size[::-1], 4)
        ray_parameters.append(parameters)
    print(np.shape(ray_parameters))

    # Find rays
    cam_0_xs, cam_0_ys = cam_0_points[:, 0], cam_0_points[:, 1]
    cam_0_rays = ray_parameters[0][cam_0_ys, cam_0_xs]
    cam_0_rays_start = np.concat([cam_0_rays[:, :2], np.zeros([len(cam_0_rays), 1])], axis=1)
    cam_0_rays_direction = np.concat([cam_0_rays[:, 2:], np.ones([len(cam_0_rays), 1])], axis=1)

    cam_1_xs, cam_1_ys = cam_1_points[:, 0], cam_1_points[:, 1]
    cam_1_rays_from_1 = ray_parameters[0][cam_1_ys, cam_1_xs]
    cam_1_rays_start_from_1 = np.concat([cam_1_rays_from_1[:, :2], np.zeros([len(cam_1_rays_from_1), 1])], axis=1)
    cam_1_rays_direction_from_1 = np.concat([cam_1_rays_from_1[:, 2:], np.ones([len(cam_1_rays_from_1), 1])], axis=1)

    # Load extrinsic
    transform_cam_0_to_1 = np.load(os.path.join(extrinsic_dir, 'extrinsic.npy'))
    print(transform_cam_0_to_1)

    # Map camera 1 ray to camera 0 frame
    cam_1_rays_start = transform.transform_point(transform_cam_0_to_1, cam_1_rays_start_from_1)
    cam_1_rays_direction = cam_1_rays_direction_from_1 @ transform_cam_0_to_1[:3, :3].T


    def quiver_ray(ax, starts, directions, color):
        starts = starts[::10000]
        directions = directions[::10000]
        ax.quiver(
            starts[:, 0], starts[:, 1], starts[:, 2],
            directions[:, 0],directions[:, 1], directions[:, 2],
            length=1000.0, normalize=True, color=color)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    quiver_ray(ax, cam_0_rays_start, cam_0_rays_direction, 'r')
    quiver_ray(ax, cam_1_rays_start, cam_1_rays_direction, 'b')
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-100, 1900)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    

    # 3D reconstruct
    mid, cp1, cp2 = transform.closest_points_between_rays(
        cam_0_rays_start,
        cam_0_rays_direction,
        cam_1_rays_start,
        cam_1_rays_direction)
    np.savetxt("pointcloud.xyz", mid, fmt="%.6f", delimiter=" ")
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mid[:,0], mid[:,1], mid[:,2])
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-100, 1900)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    """

