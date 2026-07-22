import os
import shutil
import tempfile

import cv2
import evo.core.trajectory
import evo.tools.plot
import matplotlib.pyplot as plt
import numpy as np
import pycolmap


def undistort_image_dir(model_dir: str, image_dir: str, rgb: bool) -> np.ndarray:
    with tempfile.TemporaryDirectory() as temp_dir:
        undistort_dir = os.path.join(temp_dir, "undistort")
        os.makedirs(undistort_dir, exist_ok=True)
        pycolmap.undistort_images(undistort_dir, model_dir, image_dir)
        undistort_image_dir = os.path.join(undistort_dir, "images")

        # read highres mask
        images = list[np.ndarray]()
        for filename in sorted(os.listdir(undistort_image_dir)):
            image_path = os.path.join(undistort_image_dir, filename)
            if rgb:
                image = cv2.imread(image_path)
                assert isinstance(image, np.ndarray), f"Failed to read image: {filename}"
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                assert isinstance(image, np.ndarray), f"Failed to read image: {filename}"
                images.append(image)
        images = np.array(images)

        # cleanup
        shutil.rmtree(undistort_dir, ignore_errors=True)

        return images


def align_path_from_extrinsics(
    extrinsics_target: np.ndarray,
    extrinsics_ref: np.ndarray,
) -> tuple[evo.core.trajectory.PosePath3D, evo.core.trajectory.PosePath3D, np.ndarray, np.ndarray, float]:
    pose_ref = np.linalg.inv(extrinsics_ref)
    pose_target = np.linalg.inv(extrinsics_target)
    path_ref = evo.core.trajectory.PosePath3D(poses_se3=pose_ref)
    path_target = evo.core.trajectory.PosePath3D(poses_se3=pose_target)
    rotate, translate, scale = path_target.align(path_ref, correct_scale=True)
    return path_target, path_ref, rotate, translate, scale


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


def depth_to_plane(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    N, H, W, _ = image.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (u, v, 1.0) in (H * W, 3)

    points_all, colors_all = [], []
    for i in range(N):
        valid = (depth[i] < 10.0) & (depth[i] > 0)  # boolean in (H, W)
        valid &= conf[i] >= conf_thr
        valid &= mask[i] > 0
        if not np.any(valid):
            continue
        valid_idx = np.flatnonzero(valid.reshape(-1))  # int in (H, W)

        K_inv = np.linalg.inv(K[i])  # float in (3, 3) intrisics
        c2w = np.linalg.inv(ext_w2c[i])  # float in (4, 4) camera to world

        rays = K_inv @ pix[valid_idx].T  # (x, y, z) in (3, M) perspective rays direction
        camera_w, rays_w = c2w[:3, 3], c2w[:3, :3] @ rays  # (3, M) perspective rays direction in world
        d = -camera_w[2] / rays_w[2, :]  # (M,) depth along the rays
        pts_w = (camera_w[:, np.newaxis] + d[np.newaxis, :] * rays_w).T  # (M, 3) points
        colors = image[i].reshape(-1, 3)[valid_idx]  # (M, 3) colors

        points_all.append(pts_w)  # (M, 3)
        colors_all.append(colors)  # (M, 3)

    if len(points_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(points_all, 0), np.concatenate(colors_all, 0)
