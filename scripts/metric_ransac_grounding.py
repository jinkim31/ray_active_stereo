import numpy as np
from sklearn.linear_model import RANSACRegressor


def fit_plane_ransac(points, min_samples=3, residual_threshold=0.01):
    """
    Fit a plane to 3D points using RANSAC.
    points: (N,3) numpy array
    """
    X = points[:, :2]  # x,y
    y = points[:, 2]  # z

    ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold)
    ransac.fit(X, y)

    # Plane: z = a*x + b*y + c
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    normal = np.array([a, b, -1.0])
    normal /= np.linalg.norm(normal)

    return normal, c


def align_to_xy_plane(points):
    # 1. Fit plane
    normal, c = fit_plane_ransac(points)

    # 2. Find rotation matrix: rotate plane normal -> [0,0,1]
    target_normal = np.array([0, 0, 1])
    v = np.cross(normal, target_normal)
    s = np.linalg.norm(v)
    if s < 1e-10:
        R = np.eye(3)  # already aligned
    else:
        c_dot = np.dot(normal, target_normal)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c_dot) / (s ** 2))

    # 3. Rotate points
    rotated = points @ R.T

    # 4. Translate so plane passes through z=0
    z_offset = np.mean(rotated[:, 2])
    rotated[:, 2] -= z_offset

    return rotated, R, z_offset



points = np.loadtxt('../clouds/as_pinhole_raw.xyz')
aligned_points, R, z_off = align_to_xy_plane(points)
np.savetxt('../clouds/as_pinhole.xyz', aligned_points)