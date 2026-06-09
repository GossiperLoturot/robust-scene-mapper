import evo.core.trajectory
import evo.tools.plot
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize


def align_path_from_extrinsics(
    extrinsics_target: np.ndarray,
    extrinsics_ref: np.ndarray,
) -> tuple[evo.core.trajectory.PosePath3D, evo.core.trajectory.PosePath3D, float]:
    pose_ref = np.linalg.inv(extrinsics_ref)
    pose_target = np.linalg.inv(extrinsics_target)
    path_ref = evo.core.trajectory.PosePath3D(poses_se3=pose_ref)
    path_target = evo.core.trajectory.PosePath3D(poses_se3=pose_target)
    _, _, scale = path_target.align(path_ref, correct_scale=True)
    return path_target, path_ref, scale


def plot_path(
    path: str,
    path_target: evo.core.trajectory.PosePath3D,
    path_ref: evo.core.trajectory.PosePath3D
):
    fig = plt.figure(figsize=(8, 8))
    ax = evo.tools.plot.prepare_axis(fig, plot_mode=evo.tools.plot.PlotMode.xyz)
    evo.tools.plot.traj(ax=ax, plot_mode=evo.tools.plot.PlotMode.xyz, traj=path_target, color="blue")
    evo.tools.plot.traj(ax=ax, plot_mode=evo.tools.plot.PlotMode.xyz, traj=path_ref, color="red")
    fig.axes.append(ax)
    fig.savefig(path)


# https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/utils/export/glb.py#L205
def depth_to_world_point(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (u, v, 1.0) in (H * W, 3)

    points_all, colors_all = [], []
    for i in range(N):
        d = depth[i]  # depth value in (H, W)
        valid = np.isfinite(d) & (d > 0)  # boolean in (H, W)
        valid &= conf[i] >= conf_thr
        valid &= mask[i] > 0
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)  # depth value in (H * W)
        vidx = np.flatnonzero(valid.reshape(-1))  # int in (H, W)

        K_inv = np.linalg.inv(K[i])  # float in (3, 3) intrisics
        c2w = np.linalg.inv(ext_w2c[i])  # float in (4, 4) camera to world

        rays = K_inv @ pix[vidx].T  # (x, y, z) in (3, M) perspective rays direction
        Xc = rays * d_flat[vidx][None, :]  # (3, M) rays
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M, 3) points
        colors = image[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M, 3) colors

        points_all.append(Xw)
        colors_all.append(colors)

    if len(points_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(points_all, 0), np.concatenate(colors_all, 0)


def transform_for_optimize_plane(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    cpts = points - centroid

    def distance_from_plane_mae(params):
        theta, phi, d = params
        a = np.sin(theta) * np.cos(phi)
        b = np.sin(theta) * np.sin(phi)
        c = np.cos(theta)
        distances = np.abs(a * cpts[:, 0] + b * cpts[:, 1] + c * cpts[:, 2] + d)
        return np.mean(distances)
    res = scipy.optimize.minimize(distance_from_plane_mae, [0.0, 0.0, 0.0], method="Nelder-Mead")
    theta_opt, phi_opt, d_opt = res.x

    # plane equation ax + by + cz + d = 0
    a = np.sin(theta_opt) * np.cos(phi_opt)
    b = np.sin(theta_opt) * np.sin(phi_opt)
    c = np.cos(theta_opt)
    normal = np.array([a, b, c])
    rot, _ = scipy.spatial.transform.Rotation.align_vectors([0, 0, 1], normal)
    rot_matrix = rot.as_matrix()

    # translate plane to origin
    t_origin = centroid - d_opt * normal
    t = -rot_matrix @ t_origin

    # create transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rot_matrix
    matrix[:3, 3] = t
    return matrix
