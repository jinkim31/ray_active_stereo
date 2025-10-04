import cv2
import numpy as np
import os
import glob
import torch
import transform
import util
import matplotlib.pyplot as plt

# Parameters
dataset_dir = '../datasets/extrinsic'
intrinsic_dir = '../results/intrinsic_results'
result_dir = '../results/extrinsic_results'
checkerboard_shape = [10, 7]
checkerboard_corners_3d = np.array([
    (x, y, 0.0)
    for y in range(checkerboard_shape[1])
    for x in range(checkerboard_shape[0])]).astype(np.float32) * 35.0

if __name__ == '__main__':

    # Handler parameters
    cam_names = sorted(os.listdir(dataset_dir))
    n_cams = len(cam_names)
    os.makedirs(result_dir, exist_ok=True)

    # Get checkerboard corners
    def find_checkerboard_corners():
        # Find checkerboard corners
        checkerboard_corners = [ [] for _ in range(n_cams) ]
        image_size = None
        for cam_index, cam_name in enumerate(cam_names):
            for file_path in sorted(glob.glob(os.path.join(dataset_dir, cam_name, '*.png'))):
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                image_size = image.shape[::-1]
                image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                ret, corners = cv2.findChessboardCorners(image, checkerboard_shape, None)
                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
                    checkerboard_corners[cam_index].append(corners.squeeze()*2)
                else:
                    checkerboard_corners[cam_index].append(None)

        # Sanitize corner array
        checkerboard_corners = np.array(util.sanitize_lists(checkerboard_corners)).astype(np.float32)
        print('Corners', checkerboard_corners.shape)

        return checkerboard_corners, image_size
    checkerboard_corners, image_size = util.pickle_get('checkerboard_corners.pkl', find_checkerboard_corners)

    # Perform pinhole calibration to find checkerboard poses
    def get_checkerboard_initial_poses():
        checkerboard_poses = []
        for cam_index in range(n_cams):
            _, _, _, rs, ts = cv2.calibrateCamera(
                np.tile(checkerboard_corners_3d, (len(checkerboard_corners[cam_index]), 1, 1)),
                checkerboard_corners[cam_index],
                image_size, None, None)
            checkerboard_poses.append(np.squeeze(np.concat([rs, ts], axis=1)))
        return np.array(checkerboard_poses)
    checkerboard_poses = util.pickle_get('checkerboard_poses.pkl', get_checkerboard_initial_poses) # [n_cam, n_checkerboards, 6]

    # Compute pose between cameras
    n_checkerboards = checkerboard_poses.shape[1]
    pose_camera_0_to_1_candidates = []
    for checkerboard_index in range(n_checkerboards):
        transform_camera_0_to_checkerboard = transform.pose_vector_to_transform_mat(checkerboard_poses[0, checkerboard_index]) # T^0_C
        transform_camera_1_to_checkerboard = transform.pose_vector_to_transform_mat(checkerboard_poses[1, checkerboard_index]) # T^1_C
        transform_camera_0_to_1 = transform_camera_0_to_checkerboard @ transform.invert_transform_mat(transform_camera_1_to_checkerboard) # T^0_1 = T^0_C @ inv(T^1_C)
        pose_camera_0_to_1 = transform.transform_mat_to_pose_vector(transform_camera_0_to_1)
        pose_camera_0_to_1_candidates.append(pose_camera_0_to_1)
    pose_camera_0_to_1_candidates = np.array(pose_camera_0_to_1_candidates)
    pose_camera_0_to_1 = np.mean(pose_camera_0_to_1_candidates, axis=0)

    # Load ray parameters
    ray_parameters = []
    for cam_name in cam_names:
        parameters = np.load(os.path.join(intrinsic_dir, cam_name, 'parameters.npy'))
        parameters = parameters.reshape(*image_size[::-1], 4)
        ray_parameters.append(parameters)
    print(np.shape(ray_parameters))

    # Optimize pose
    pose_camera_0_to_1 = torch.nn.Parameter(torch.tensor(pose_camera_0_to_1))
    optimizer = torch.optim.Adam([pose_camera_0_to_1], lr=1e-3)

    for epoch in range(100000):
        loss = torch.tensor(0.0)
        for i_checkerboard in range(n_checkerboards):

            # Sample rays
            cam_0_xs, cam_0_ys = np.round(checkerboard_corners[0][i_checkerboard][:, 0]).astype(int), np.round(checkerboard_corners[0][i_checkerboard][:, 1]).astype(int)
            cam_0_rays = ray_parameters[0][cam_0_ys, cam_0_xs]
            cam_0_rays_start = np.concat([cam_0_rays[:, :2], np.zeros([len(cam_0_rays), 1])], axis=1)
            cam_0_rays_direction = np.concat([cam_0_rays[:, 2:], np.ones([len(cam_0_rays), 1])], axis=1)
            cam_1_xs, cam_1_ys = np.round(checkerboard_corners[1][i_checkerboard][:, 0]).astype(int), np.round(checkerboard_corners[1][i_checkerboard][:, 1]).astype(int)
            cam_1_rays_from_1 = ray_parameters[0][cam_1_ys, cam_1_xs]
            cam_1_rays_start_from_1 = np.concat([cam_1_rays_from_1[:, :2], np.zeros([len(cam_1_rays_from_1), 1])], axis=1)
            cam_1_rays_direction_from_1 = np.concat([cam_1_rays_from_1[:, 2:], np.ones([len(cam_1_rays_from_1), 1])], axis=1)

            # Compute transform from camera to 1
            transform_cam_0_to_1 = transform.pose_vector_to_transform_mat_torch(pose_camera_0_to_1)

            # Map camera 1 rays to camera 0 frame
            cam_1_rays_start = transform.transform_point_torch(transform_cam_0_to_1, torch.tensor(cam_1_rays_start_from_1))
            cam_1_rays_direction = torch.tensor(cam_1_rays_direction_from_1) @ transform_cam_0_to_1[:3, :3].T

            # 3D reconstruct
            mid, cp1, cp2 = transform.closest_points_between_rays_torch(
                torch.tensor(cam_0_rays_start),
                torch.tensor(cam_0_rays_direction),
                cam_1_rays_start,
                cam_1_rays_direction)

            # Register known checkerboard geometry to reconstructed points
            _, _, error = transform.kabsch_register_points(mid.float(), torch.tensor(checkerboard_corners_3d))

            # Add to loss
            loss += error ** 2

        print(f'Epoch {epoch}: {loss}')

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save occasionally
        if epoch % 1000 == 0:
            transform_camera_0_to_1 = transform.pose_vector_to_transform_mat(pose_camera_0_to_1.detach().numpy())
            print(transform_camera_0_to_1)
            np.save(os.path.join(result_dir, 'extrinsic.npy'), transform_camera_0_to_1)
