import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import glob
import argparse
import ray_calibration
import random
import os

decode_map_dir = '../results/intrinsic_decode_maps/192.168.1.210/decode_maps'
result_dir = '../results/intrinsic_results/192.168.1.210'

def get_rt(decode_maps):
    # Arguments
    lcd_resolution = [3840, 2160]
    lcd_size = [700, 400]

    # Convert pixel index to meters
    decode_maps = np.stack([
        decode_maps[..., 0] / lcd_resolution[0] * lcd_size[0],
        decode_maps[..., 1] / lcd_resolution[1] * lcd_size[1],  # use lcd_size[1] for Y
    ], axis=-1)

    # Intrinsic calibration data
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    for decode_map in decode_maps:
        # Sample valid points
        valid_mask = ~np.isnan(decode_map).any(axis=-1)
        valid_points = np.argwhere(valid_mask)
        choice = np.random.choice(len(valid_points), size=100, replace=False)
        sampled_points = valid_points[choice]

        # 3D world points (x, y, z=0 since LCD is planar)
        sampled_3d_points = decode_map[sampled_points[:, 0], sampled_points[:, 1], :]
        sampled_3d_points = np.hstack([sampled_3d_points, np.zeros((sampled_3d_points.shape[0], 1))])

        # 2D image points (swap row,col → x=col, y=row)
        sampled_2d_points = sampled_points[:, [1, 0]]

        objpoints.append(sampled_3d_points.astype(np.float32))  # (N,3)
        imgpoints.append(sampled_2d_points.astype(np.float32))  # (N,2)

    print(f'{len(imgpoints)} images found')

    # Run calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (4024, 3036), None, None
    )

    return np.array(rvecs).squeeze(), np.array(tvecs).squeeze()



if __name__ == '__main__':

    os.makedirs(result_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--image_folder', type=str, default='cam_img',
                        help='Path to folder containing checkerboard images.')
    parser.add_argument('--pattern_rows', type=int, default=10, help='Checkerboard rows (inner corners).')
    parser.add_argument('--pattern_cols', type=int, default=7, help='Checkerboard columns (inner corners).')
    parser.add_argument('--square_size', type=float, default=30.0,
                        help='Size of a square in the checkerboard (in your preferred unit).')
    parser.add_argument('--cameradata_path', default='cameradata.h5')
    parser.add_argument('--pointdata_path', default='pointdata.h5')
    parser.add_argument('--optimization_result_path', default='optimization_results.h5')
    parser.add_argument('--cam_resolution', type=int, nargs=2, default=(4024, 3036),
                        help='Camera resolution (width height).')
    parser.add_argument('--target_size', type=int, nargs=2, default=(400, 700),
                        help='Camera resolution (width height).')
    args = parser.parse_args()

    lcd_resolution = [2160, 3840]

    # model = ForwardBSplineSurface(4, (3036 , 4024) , np.linspace(0,1,180) , np.linspace(0,1,120))
    # ctrl_pts , rvec , tvec  = load_optimization_from_h5(args.optimization_result_path)
    # K , _ , _ , _ = cameradata_func("cameradata.h5" , is_write=False)
    # model.set_extrinsic(rvec , tvec)
    # model.set_initctrl_pts(ctrl_pts)
    # decoded_maps , decoded_mask =model.forward_calulation(args.target_size)
    # print("decoded_map  & mask are calculated")
    '''
    decode_map 들의 path 를 가져옴 

    target_size는 parser 에 있음 (세로 가로)    
    LCD resolution은 [세로 가로] 순서

    rvec tvec  은 N,3 이 어야하지만 , 
    N,3,1로 되 어있어서 squeeze로 N,3


    '''

    decode_maps_list = []
    paths = sorted(glob.glob(os.path.join(decode_map_dir, '*.npy')))  # decode map path


    for path in paths:
        pose = np.load(path)
        print(path, pose.shape)
        decode_maps_list.append(pose)

    decode_maps = np.stack(decode_maps_list, axis=0)

    rvec, tvec = get_rt(decode_maps)

    decode_x = decode_maps[..., 0] / lcd_resolution[1]
    decode_y = decode_maps[..., 1] / lcd_resolution[0]
    decode_maps = np.stack([decode_x, decode_y], axis=-1)
    decode_masks = (~np.isnan(decode_maps[..., 0])) * 1
    mask = np.isnan(decode_maps)
    decode_maps[mask] = 0

    # total ray_parameters가 전체 캘리브레이션 결과
    # transforms_camera_to_target는 3,4의 변환 행렬
    ray_parameters, total_ray_parameters, transforms_camera_to_target = ray_calibration.ray_calibrate(
        decode_maps, decode_masks,
        args.target_size
        , rvec, tvec, learning_rate=1e-5,
        stop_loss=1e-8, device='cuda',
        show_plot=False)
    plt = ray_calibration.plot(ray_parameters, transforms_camera_to_target, args.target_size, ray_decimation_rate=4000)
    plt.show()

    np.save(os.path.join(result_dir, 'parameters.npy'), total_ray_parameters)
    np.save(os.path.join(result_dir, 'lcd_transforms.npy'), transforms_camera_to_target)
