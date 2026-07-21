import os
import shutil
import tempfile

import cv2
import evo.core.trajectory
import evo.tools.plot
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import rich.progress
import torch

import context


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


# https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/utils/export/glb.py#L205
def depth_to_world_point(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
    near_clip: float = 1.0,
    far_clip: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (u, v, 1.0) in (H * W, 3)

    points_all, colors_all, ray_all = [], [], []
    for i in range(N):
        d = depth[i]  # depth value in (H, W)
        valid = np.isfinite(d) & (d > 0)  # boolean in (H, W)
        valid &= conf[i] >= conf_thr
        valid &= mask[i] > 0
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)  # depth value in (H * W)
        valid_idx = np.flatnonzero(valid.reshape(-1))  # int in (H, W)

        K_inv = np.linalg.inv(K[i])  # float in (3, 3) intrisics
        c2w = np.linalg.inv(ext_w2c[i])  # float in (4, 4) camera to world

        rays = K_inv @ pix[valid_idx].T  # (x, y, z) in (3, M) perspective rays direction
        pts_c = rays * d_flat[valid_idx][np.newaxis, :]  # (3, M) rays
        pts_c_hom = np.vstack([pts_c, np.ones((1, pts_c.shape[1]))])
        pts_w = (c2w @ pts_c_hom)[:3].T.astype(np.float32)  # (M, 3) points
        colors = image[i].reshape(-1, 3)[valid_idx].astype(np.uint8)  # (M, 3) colors

        # ray (original code)
        ray_n = rays * near_clip
        ray_f = rays * far_clip
        ray_n_hom = np.vstack([ray_n, np.ones((1, ray_n.shape[1]))])
        ray_f_hom = np.vstack([ray_f, np.ones((1, ray_f.shape[1]))])
        ray_n_w = (c2w @ ray_n_hom)[:3].T.astype(np.float32)  # (M, 3) points
        ray_f_w = (c2w @ ray_f_hom)[:3].T.astype(np.float32)  # (M, 3) points

        points_all.append(pts_w)  # (M, 3)
        colors_all.append(colors)  # (M, 3)
        ray_all.append(np.stack([ray_n_w, ray_f_w], axis=1))  # (M, 2, 3)

    if len(points_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 2, 3), dtype=np.uint8)

    return np.concatenate(points_all, 0), np.concatenate(colors_all, 0), np.concatenate(ray_all, 0)


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


# points align planar and guided points.
def alignment_model(
    xyz: np.ndarray,  # (N, 3)
    rgb: np.ndarray,  # (N, 3)
    ray: np.ndarray,  # (N, 2, 3)
    xyz_guide: np.ndarray,  # (N, 3)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = context.Context()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert xyz.shape[0] == rgb.shape[0] == ray.shape[0]
    assert xyz.shape[1] == 3 and rgb.shape[1] == 3 and (ray.shape[1] == 2 and ray.shape[2] == 3)
    assert xyz.dtype == np.float32 and rgb.dtype == np.uint8 and ray.dtype == np.float32

    N, _ = xyz.shape
    ctx.logger.info(f"alignment: {N} points.")

    # build faiss index
    res = faiss.StandardGpuResources()
    index_self = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(3))
    index_self.add(xyz)

    # find self nearest neighbors
    K = 256
    _, net_self = index_self.search(xyz, K)  # (N, K)
    net_self = torch.from_numpy(net_self).long().to(dev)  # (N, K)
    ctx.logger.info(f"computed nearest neighbors: {K} points.")
    del index_self, res

    # compute ray directions
    raydirs = ray[:, 1] - ray[:, 0]  # (N, 3)
    raydirs = raydirs / np.linalg.norm(raydirs, axis=1, keepdims=True)  # (N, 3)
    raydirs = torch.from_numpy(raydirs).float().to(dev)  # (N, 3)

    EPOCH, ALPHA, B = 32, -0.4, 4096
    normals = torch.zeros((N, 3), dtype=torch.float32, device=dev)  # (N, 3)
    current_xyz = torch.from_numpy(xyz).float().to(dev)  # (N, 3)
    for epoch in rich.progress.track(range(EPOCH), description="alignment model", console=ctx.console):
        neighbors = current_xyz[net_self]  # (N, K, 3)

        # compute local center and normals
        means = torch.mean(neighbors, dim=1)  # (N, 3)
        diff = neighbors - means.unsqueeze(1)  # (N, K, 3)
        cov = torch.bmm(diff.transpose(1, 2), diff) / (K - 1)  # (N, 3, 3)
        # batch eigen decomposition for large N
        normals_list = []
        for i in range(0, N, B):
            _, eigenvectors = torch.linalg.eigh(cov[i:i + B])  # (B, 3, 3)
            normals_list.append(eigenvectors[:, :, 0])  # (B, 3)
        normals = torch.cat(normals_list, dim=0)  # (N, 3)

        normals = torch.where(normals[:, [1]] > 0, -normals, normals)

        # compute patch (local plane)
        n_dot_r = torch.sum(normals * raydirs, dim=1, keepdims=True)  # (N, 1)
        d_dot_n = torch.sum((current_xyz - means) * normals, dim=1, keepdims=True)  # (N, 1)
        patch_term = ALPHA * d_dot_n * n_dot_r  # (N, 1)
        ctx.logger.info(f"[{1 + epoch}/{EPOCH}] displacement error: {torch.mean(patch_term)}")

        # apply
        current_xyz = current_xyz + patch_term * raydirs  # (N, 3)
    xyz = current_xyz.cpu().numpy()
    normals = normals.cpu().numpy()

    return xyz, rgb, normals
