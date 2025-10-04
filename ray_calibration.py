import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def ray_calibrate(decode_maps, decode_masks, target_size, rvec, tvec,
                  learning_rate=5e-2, stop_loss=1e-1, show_plot=False,
                  verbose=True, device='cpu'):
    Num_mask = np.sum(decode_masks, axis=0)

    itv_mask = np.zeros_like(Num_mask)
    itv_mask[::3, ::3] = 1

    rowcol = np.where((Num_mask >= 3) & (itv_mask == 1))

    X_local = decode_maps[:, rowcol[0], rowcol[1], :] * target_size[::-1]

    nonzero_mask = (np.sum(X_local, axis=2) != 0)
    nonzero_mask = torch.tensor(nonzero_mask, device=device)
    X_local = np.concatenate([X_local, np.zeros((X_local.shape[0], X_local.shape[1], 1), dtype=X_local.dtype)], axis=-1)

    X_local = torch.tensor(X_local, dtype=torch.float64, device=device)

    image_size = decode_maps.shape[1:3]  # (H, W)
    n_targets = decode_maps.shape[0]

    r0 = torch.tensor(rvec[0, :], device=device, dtype=torch.float64)
    t0 = torch.tensor(tvec[0, :], device=device, dtype=torch.float64)

    rvec = rvec[1:, :]
    tvec = tvec[1:, :]

    rvec = torch.tensor(rvec, dtype=torch.float64, device=device)
    tvec = torch.tensor(tvec, dtype=torch.float64, device=device)
    rvec_opt = nn.Parameter(rvec.clone())
    tvec_opt = nn.Parameter(tvec.clone())

    ray_params = torch.zeros((image_size[0], image_size[1], 4), dtype=torch.float64, device=device)
    ray_params[..., 0] = 0.0  # x offset
    ray_params[..., 1] = 0.0  # y offset
    ray_params[..., 2] = 0  # direction x
    ray_params[..., 3] = 0  # direction y
    # ray_parameters = init_ray_parameters_from_K(K, image_size, device)
    # ray_parameters = nn.Parameter(torch.zeros(*image_size, 4).to(device, dtype=torch.float64)) # [480, 640, 4]. x, y, u, v

    optimizer = optim.Adam([rvec_opt, tvec_opt], lr=learning_rate)

    # batch_size = 71  # or 1, depending on GPU capacity
    for epoch in range(500):
        loss = torch.tensor(0.0, dtype=torch.float64, device=device)
        r0_ = r0.unsqueeze(0)
        rvec_new = torch.cat([r0_, rvec_opt], 0)
        t0_ = t0.unsqueeze(0)
        tvec_new = torch.cat([t0_, tvec_opt], 0)
        R = rodrigues_to_rotation_matrix(rvec_new)  # (N, 3, 3)
        T = tvec_new.unsqueeze(2)  # (N, 3, 1)
        transform_matrix = torch.cat([R, T], dim=2)  # (N, 3, 4)

        X_global = X_local @ R.transpose(1, 2) + T.transpose(1, 2)
        # X_global[zero_mask, :] = torch.nan

        x = X_global[:, :, 0]  # N(33) X M
        y = X_global[:, :, 1]  # N(33) X M
        z = X_global[:, :, 2]  # N(33) X M

        count = nonzero_mask.sum(dim=0).clamp(min=1)
        x_mean = (x * nonzero_mask).sum(dim=0) / count
        y_mean = (y * nonzero_mask).sum(dim=0) / count
        z_mean = (z * nonzero_mask).sum(dim=0) / count
        x_ = x - x_mean
        y_ = y - y_mean
        z_ = z - z_mean

        z2_mean = (z_ ** 2 * nonzero_mask).sum(dim=0) / count

        um = (x_ * z_ * nonzero_mask).sum(dim=0) / count / z2_mean
        vm = (y_ * z_ * nonzero_mask).sum(dim=0) / count / z2_mean

        x0 = x_mean - um * z_mean
        y0 = y_mean - vm * z_mean

        Errx = abs(X_global[:, :, 2] * um + x0 - X_global[:, :, 0])
        Erry = abs(X_global[:, :, 2] * vm + y0 - X_global[:, :, 1])
        loss += ((Errx * nonzero_mask).sum(dim=0) / count + (Erry * nonzero_mask).sum(dim=0) / count).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            print(f'Epoch {epoch}: Loss = {loss.item():.12f}')

        # if show_plot and epoch % 10 == 0:
        #     plot(ray_parameters, transform_matrix)
        if loss <= stop_loss:
            break
    print(x0)
    ray_parameters = torch.stack([x0, y0, um, vm], dim=0).T.cpu()
    with torch.no_grad():
        rowcol2 = np.where((Num_mask >= 3))
        X_local = decode_maps[:, rowcol2[0], rowcol2[1], :] * target_size[::-1]
        nonzero_mask = (np.sum(X_local, axis=2) != 0)
        nonzero_mask = torch.tensor(nonzero_mask, device=device)
        X_local = np.concatenate([X_local, np.zeros((X_local.shape[0], X_local.shape[1], 1), dtype=X_local.dtype)],
                                 axis=-1)
        X_local = torch.tensor(X_local, dtype=torch.float64, device=device)
        r0_ = r0.unsqueeze(0)
        rvec_new = torch.cat([r0_, rvec_opt], 0)
        t0_ = t0.unsqueeze(0)
        tvec_new = torch.cat([t0_, tvec_opt], 0)
        R = rodrigues_to_rotation_matrix(rvec_new)  # (N, 3, 3)
        T = tvec_new.unsqueeze(2)  # (N, 3, 1)
        X_global = X_local @ R.transpose(1, 2) + T.transpose(1, 2)
        # X_global[zero_mask, :] = torch.nan

        x = X_global[:, :, 0]  # N(33) X M
        y = X_global[:, :, 1]  # N(33) X M
        z = X_global[:, :, 2]  # N(33) X M

        X_global = X_global.to('cpu')
        count = nonzero_mask.sum(dim=0).clamp(min=1)
        x_mean = (x * nonzero_mask).sum(dim=0) / count
        y_mean = (y * nonzero_mask).sum(dim=0) / count
        z_mean = (z * nonzero_mask).sum(dim=0) / count
        x_ = x - x_mean
        y_ = y - y_mean
        z_ = z - z_mean
        x_, y_, z_ = x_.to('cpu'), y_.to('cpu'), z_.to('cpu')
        x_mean, y_mean, z_mean = x_mean.to('cpu'), y_mean.to('cpu'), z_mean.to('cpu')
        count, nonzero_mask = count.to('cpu'), nonzero_mask.to('cpu')
        z2_mean = (z_ ** 2 * nonzero_mask).sum(dim=0) / count

        um = (x_ * z_ * nonzero_mask).sum(dim=0) / count / z2_mean
        vm = (y_ * z_ * nonzero_mask).sum(dim=0) / count / z2_mean

        x0 = x_mean - um * z_mean
        y0 = y_mean - vm * z_mean

        ray_parameters_total = torch.stack([x0, y0, um, vm], dim=0).T.cpu()

    return ray_parameters.detach().cpu().numpy(), ray_parameters_total.detach().cpu().numpy(), transform_matrix.detach().cpu()


def rodrigues_to_rotation_matrix(rvec):
    theta = torch.norm(rvec, dim=1, keepdim=True).clamp(min=1e-8)
    r = rvec / theta
    rx, ry, rz = r[:, 0:1], r[:, 1:2], r[:, 2:3]

    zeros = torch.zeros_like(rx)
    K = torch.cat([
        torch.cat([zeros, -rz, ry], dim=1).unsqueeze(1),
        torch.cat([rz, zeros, -rx], dim=1).unsqueeze(1),
        torch.cat([-ry, rx, zeros], dim=1).unsqueeze(1)
    ], dim=1)  # (N, 3, 3)

    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype).unsqueeze(0)
    R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta).unsqueeze(-1)) * (K @ K)
    return R


def init_ray_parameters_from_K(K, image_size, device='cpu'):
    H, W = image_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float64, device=device),
        torch.arange(W, dtype=torch.float64, device=device),
        indexing='ij'
    )

    d_x = (grid_x - cx) / fx
    d_y = (grid_y - cy) / fy

    ray_params = torch.zeros((H, W, 4), dtype=torch.float64, device=device)
    ray_params[..., 0] = 0.0  # x offset
    ray_params[..., 1] = 0.0  # y offset
    ray_params[..., 2] = d_x  # direction x
    ray_params[..., 3] = d_y  # direction y

    return nn.Parameter(ray_params)


def plot(ray_parameters, transform_matrix, target_size, ray_decimation_rate=1):
    ray_parameters_flat = ray_parameters.reshape(-1, 4)
    ray_parameters_flat = ray_parameters_flat[
        np.random.choice(len(ray_parameters_flat), int(len(ray_parameters_flat) / ray_decimation_rate), replace=False)]
    target_size = (target_size[1], target_size[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-10, 1000)
    ax.set_ylim(-10, 1000)
    ax.set_zlim(-10, 1000)

    ax.quiver(
        ray_parameters_flat[:, 0],
        ray_parameters_flat[:, 1],
        np.zeros(len(ray_parameters_flat)),
        ray_parameters_flat[:, 2],
        ray_parameters_flat[:, 3],
        np.ones(len(ray_parameters_flat)),
        length=1500.0, linewidth=0.2, arrow_length_ratio=0.1, alpha=0.4
    )

    target_vertices_local = np.array([
        [0, 0, 0, 1],
        [target_size[0], 0, 0, 1],
        [target_size[0], target_size[1], 0, 1],
        [0, target_size[1], 0, 1],
    ])
    target_vertices_global = np.einsum('pj,bij->bpi', target_vertices_local, transform_matrix)

    for target_vertices in target_vertices_global:
        ax.add_collection3d(Poly3DCollection(
            [target_vertices[:, :3]], facecolors='magenta', edgecolors='black', linewidths=1, alpha=0.7))

    return plt