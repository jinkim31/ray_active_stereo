import numpy as np
import cv2
import torch


def pose_vector_to_transform_mat(pose):
    rvec = np.array(pose[:3], dtype=float).reshape(3, 1)
    tvec = np.array(pose[3:], dtype=float).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def rodrigues_to_rotation_mat_torch(rvec):
    """
    Convert a rotation vector (3,) to a rotation matrix (3,3) using Rodrigues formula.
    rvec: tensor of shape (3,)
    """
    theta = torch.norm(rvec)
    if theta < 1e-8:
        return torch.eye(3, dtype=rvec.dtype, device=rvec.device)

    k = rvec / theta
    K = torch.zeros((3, 3), dtype=rvec.dtype, device=rvec.device)
    K[0, 1] = -k[2]
    K[0, 2] = k[1]
    K[1, 0] = k[2]
    K[1, 2] = -k[0]
    K[2, 0] = -k[1]
    K[2, 1] = k[0]

    I = torch.eye(3, dtype=rvec.dtype, device=rvec.device)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


def pose_vector_to_transform_mat_torch(pose):
    """
    Convert 6D pose vector (rvec[3], tvec[3]) to 4x4 transform matrix.
    pose: tensor of shape (6,)
    """
    rvec = pose[:3]
    tvec = pose[3:]
    R = rodrigues_to_rotation_mat_torch(rvec)
    T = torch.eye(4, dtype=pose.dtype, device=pose.device)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T

def invert_transform_mat(mat):
    R = mat[:3, :3]
    t = mat[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_mat_to_pose_vector(mat):
    R = mat[:3, :3]
    t = mat[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    rvec = rvec.flatten()
    pose6 = np.hstack([rvec, t])
    return pose6

def transform_point(transform, points):
    points_h = np.hstack([points, np.ones((len(points), 1))])  # shape [N, 4]
    points_transformed = (transform @ points_h.T).T  # shape [N, 4]
    points_transformed = points_transformed[:, :3] / points_transformed[:, 3:4]  # shape [N, 3]
    return points_transformed

def transform_point_torch(transform, points):
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=1)  # [N, 4]
    points_transformed_h = (transform @ points_h.T).T  # [N, 4]
    points_transformed = points_transformed_h[:, :3] / points_transformed_h[:, 3:4]
    return points_transformed



def closest_points_between_rays(p1_list, d1_list, p2_list, d2_list):
    """
    Compute midpoints of closest points between two sets of rays.

    p1_list, d1_list: [N,3] arrays for first camera rays
    p2_list, d2_list: [N,3] arrays for second camera rays

    Returns:
        midpoints: [N,3] array of midpoints
        cp1_list: [N,3] points on first rays
        cp2_list: [N,3] points on second rays
    """
    p1 = np.array(p1_list, dtype=float)
    d1 = np.array(d1_list, dtype=float)
    p2 = np.array(p2_list, dtype=float)
    d2 = np.array(d2_list, dtype=float)

    # Normalize directions
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
    d2 /= np.linalg.norm(d2, axis=1, keepdims=True)

    r = p1 - p2
    a = np.sum(d1 * d1, axis=1)  # should be 1 after normalization
    b = np.sum(d1 * d2, axis=1)
    c = np.sum(d2 * d2, axis=1)  # should be 1 after normalization
    d = np.sum(d1 * r, axis=1)
    e = np.sum(d2 * r, axis=1)

    denom = a * c - b * b
    # Handle parallel rays
    t1 = np.where(denom != 0, (b * e - c * d) / denom, 0)
    t2 = np.where(denom != 0, (a * e - b * d) / denom, e / c)

    cp1_list = p1 + d1 * t1[:, None]
    cp2_list = p2 + d2 * t2[:, None]
    midpoints = (cp1_list + cp2_list) / 2

    return midpoints, cp1_list, cp2_list


def closest_points_between_rays_torch(p1_list, d1_list, p2_list, d2_list):
    """
    Compute midpoints of closest points between two sets of rays in PyTorch.

    p1_list, d1_list: [N,3] tensors for first camera rays
    p2_list, d2_list: [N,3] tensors for second camera rays

    Returns:
        midpoints: [N,3] tensor of midpoints
        cp1_list: [N,3] points on first rays
        cp2_list: [N,3] points on second rays
    """
    # Normalize directions
    d1 = d1_list / d1_list.norm(dim=1, keepdim=True)
    d2 = d2_list / d2_list.norm(dim=1, keepdim=True)

    r = p1_list - p2_list
    a = torch.sum(d1 * d1, dim=1)  # should be 1
    b = torch.sum(d1 * d2, dim=1)
    c = torch.sum(d2 * d2, dim=1)  # should be 1
    d = torch.sum(d1 * r, dim=1)
    e = torch.sum(d2 * r, dim=1)

    denom = a * c - b * b
    eps = 1e-8  # avoid division by zero

    # Compute t1, t2, handle parallel rays
    t1 = torch.where(denom.abs() > eps, (b * e - c * d) / denom, torch.zeros_like(denom))
    t2 = torch.where(denom.abs() > eps, (a * e - b * d) / denom, e / c)

    cp1_list = p1_list + d1 * t1[:, None]
    cp2_list = p2_list + d2 * t2[:, None]
    midpoints = (cp1_list + cp2_list) / 2

    return midpoints, cp1_list, cp2_list


def kabsch_register_points(A, B):
    """
    Compute rotation R and translation t to align point cloud A to B
    A, B: [N,3] tensors of corresponding points

    Returns:
        R: [3,3] rotation matrix
        t: [3] translation vector
        error: mean squared distance after transformation
    """
    assert A.shape == B.shape
    N = A.shape[0]

    # Compute centroids
    centroid_A = A.mean(dim=0)
    centroid_B = B.mean(dim=0)

    # Center the clouds
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute covariance matrix
    H = AA.T @ BB
    U, S, V = torch.linalg.svd(H, full_matrices=True)
    R = V @ U.T

    # Reflection case
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T

    t = centroid_B - R @ centroid_A

    # Transform A and compute mean squared error
    A_transformed = (R @ A.T).T + t
    error = torch.mean(torch.norm(A_transformed - B, dim=1) ** 2)

    return R, t, error


