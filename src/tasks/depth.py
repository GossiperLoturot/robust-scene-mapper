import os
import shutil
import subprocess
import tempfile

import cv2
import luigi
import numpy as np
import open3d
import pycolmap

import context
import tasks.object_masking
import tasks.reconstruction
import tasks.video_sampling
import utils.depth
import utils.task
import utils.object_masking


class DepthTask(luigi.Task):
    resources = { "gpu_vol": 1 }

    input_path: luigi.StrParameter = luigi.StrParameter()
    fps: luigi.IntParameter = luigi.IntParameter()
    width: luigi.IntParameter = luigi.IntParameter()
    height: luigi.IntParameter = luigi.IntParameter()
    max_keypoints: luigi.IntParameter = luigi.IntParameter()
    depth_confidence: luigi.FloatParameter = luigi.FloatParameter()
    width_confidence: luigi.FloatParameter = luigi.FloatParameter()
    init_frame_width: luigi.IntParameter = luigi.IntParameter()
    init_frame_height: luigi.IntParameter = luigi.IntParameter()
    init_focal_length: luigi.FloatParameter = luigi.FloatParameter()

    def requires(self):
        video_sampling = tasks.video_sampling.VideoSamplingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )
        reconstruction = tasks.reconstruction.ReconstructionTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            max_keypoints=self.max_keypoints,
            width_confidence=self.width_confidence,
            depth_confidence=self.depth_confidence,
            init_frame_width=self.init_frame_width,
            init_frame_height=self.init_frame_height,
            init_focal_length=self.init_focal_length,
        )
        return [video_sampling, reconstruction]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[video_sampling], [reconstruction]] = self.input()
            image_dir = os.path.join(video_sampling.read(), "images")
            model_dir = os.path.join(reconstruction.read(), "model")

            undistort_dir = os.path.join(temp_dir, "undistort")
            os.makedirs(undistort_dir, exist_ok=True)
            pycolmap.undistort_images(undistort_dir, model_dir, image_dir)

            depth_dir = os.path.join(temp_dir, "depth")
            ctx.logger.info("estimate depth by Depth Anything 3")
            with subprocess.Popen(
                [
                    "uv", "run", "da3", "auto", undistort_dir,
                    "--export-dir", depth_dir,
                    "--export-format", "npz",
                    "--process-res", "256",
                    "--no-align-to-input-ext-scale",
                ], cwd="deps/depth-anything-3",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            ) as proc:
                if proc.stdout:
                    for line in proc.stdout:
                        ctx.console.print(line, end="")
                if proc.wait() != 0:
                    raise RuntimeError("failed to estimate depth")
            depth_path = os.path.join(depth_dir, "exports", "npz", "results.npz")

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(depth_path, output.open())


class DepthAlignTask(luigi.Task):
    input_path: luigi.StrParameter = luigi.StrParameter()
    fps: luigi.IntParameter = luigi.IntParameter()
    width: luigi.IntParameter = luigi.IntParameter()
    height: luigi.IntParameter = luigi.IntParameter()
    max_keypoints: luigi.IntParameter = luigi.IntParameter()
    depth_confidence: luigi.FloatParameter = luigi.FloatParameter()
    width_confidence: luigi.FloatParameter = luigi.FloatParameter()
    init_frame_width: luigi.IntParameter = luigi.IntParameter()
    init_frame_height: luigi.IntParameter = luigi.IntParameter()
    init_focal_length: luigi.FloatParameter = luigi.FloatParameter()
    align_confidence: luigi.FloatParameter = luigi.FloatParameter()
    highres_width: luigi.IntParameter = luigi.IntParameter()
    highres_height: luigi.IntParameter = luigi.IntParameter()
    downsample_resolution: luigi.FloatParameter = luigi.FloatParameter()
    clipping_radius: luigi.FloatParameter = luigi.FloatParameter()

    def requires(self):
        object_masking = tasks.object_masking.ObjectMaskingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            mask_categories=tuple(utils.object_masking.PLANAR_CATEGORIES),
        )
        reconstruction = tasks.reconstruction.ReconstructionTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            max_keypoints=self.max_keypoints,
            width_confidence=self.width_confidence,
            depth_confidence=self.depth_confidence,
            init_frame_width=self.init_frame_width,
            init_frame_height=self.init_frame_height,
            init_focal_length=self.init_focal_length,
        )
        depth = DepthTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            max_keypoints=self.max_keypoints,
            width_confidence=self.width_confidence,
            depth_confidence=self.depth_confidence,
            init_frame_width=self.init_frame_width,
            init_frame_height=self.init_frame_height,
            init_focal_length=self.init_focal_length,
        )
        highres_video_sampling = tasks.video_sampling.VideoSamplingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.highres_width,
            height=self.highres_height,
        )
        highres_object_masking = tasks.object_masking.ObjectMaskingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.highres_width,
            height=self.highres_height,
            mask_categories=tuple(utils.object_masking.PLANAR_CATEGORIES),
        )
        return [object_masking, reconstruction, depth, highres_video_sampling, highres_object_masking]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[object_masking], [reconstruction], [depth], [highres_video_sampling], [highres_object_masking]] = self.input()
            mask_dir = os.path.join(object_masking.read(), "masks")
            model_dir = os.path.join(reconstruction.read(), "model")
            depth_path = os.path.join(depth.read(), "results.npz")
            highres_image_dir = os.path.join(highres_video_sampling.read(), "images")
            highres_mask_dir = os.path.join(highres_object_masking.read(), "masks")

            # extract reference model
            extrinsics_ref = []
            model = pycolmap.Reconstruction(model_dir)
            for image_id in sorted(model.images):
                image = model.image(image_id)
                extrinsics_ref.append(np.concat([image.cam_from_world().matrix(), [[0, 0, 0, 1]]]))
            extrinsics_ref = np.array(extrinsics_ref)
            # undistort mask
            mask = utils.depth.undistort_image_dir(model_dir, mask_dir, rgb=False)

            # extract estimated model
            depth_output = np.load(depth_path)
            image_est = depth_output["image"]
            depth_est = depth_output["depth"]
            conf_est = depth_output["conf"]
            extrinsics_est = depth_output["extrinsics"]
            intrinsics_est = depth_output["intrinsics"]

            # check image and mask consistency
            assert image_est.shape[0] == mask.shape[0], f"image length {image_est.shape[0]} does not match mask length {mask.shape[0]}"
            # resize mask to match downsampled image for depth estimation
            mask_est = []
            for i in range(mask.shape[0]):
                m = cv2.resize(mask[i], (image_est.shape[2], image_est.shape[1]), interpolation=cv2.INTER_AREA)
                mask_est.append(m)
            mask_est = np.array(mask_est)
            mask_est = (mask_est > 127).astype(np.uint8)

            # align model
            path_ref, path_est, scale = utils.depth.align_path_from_extrinsics(extrinsics_ref, extrinsics_est)
            depth_ref = depth_est / scale
            ctx.logger.info(f"align scale: {scale}")
            # debug aligned trajectory
            fig_path = os.path.join(temp_dir, "trajectory.png")
            utils.depth.plot_path(fig_path, path_ref, path_est)

            # use plane constraint model
            points, _ = utils.depth.depth_to_world_point(depth_ref, intrinsics_est, extrinsics_ref, image_est, mask_est, conf_est, self.align_confidence)
            world_to_plane = utils.depth.transform_for_optimize_plane(points)  # z = 0 plane
            extrinsics_plane = extrinsics_ref @ np.linalg.inv(world_to_plane)
            # write params
            params_path = os.path.join(temp_dir, "params.npz")
            with open(params_path, "wb") as f:
                np.savez_compressed(f, allow_pickle=False, world_to_plane=world_to_plane, scale=scale)

            # create highres model
            highres_model_dir = os.path.join(temp_dir, "highres_model")
            os.makedirs(highres_model_dir, exist_ok=True)
            highres_model = pycolmap.Reconstruction(model_dir)
            utils.depth.resize_model(highres_model, self.highres_width, self.highres_height)
            highres_model.write(highres_model_dir)
            # undistort highres image
            highres_image = utils.depth.undistort_image_dir(highres_model_dir, highres_image_dir, rgb=True)
            # undistort highres mask
            highres_mask = utils.depth.undistort_image_dir(highres_model_dir, highres_mask_dir, rgb=False)

            # erode mask to remove boundary
            highres_mask = (highres_mask > 127).astype(np.uint8)
            kernel = np.ones((8, 8), np.uint8)
            for i in range(highres_mask.shape[0]):
                highres_mask[i] = cv2.erode(highres_mask[i], kernel, iterations=1)

            # make highres bev
            intrinsics_highres = intrinsics_est.copy()
            intrinsics_highres[:, 0, 0] *= highres_image.shape[2] / image_est.shape[2]
            intrinsics_highres[:, 1, 1] *= highres_image.shape[1] / image_est.shape[1]
            intrinsics_highres[:, 0, 2] *= highres_image.shape[2] / image_est.shape[2]
            intrinsics_highres[:, 1, 2] *= highres_image.shape[1] / image_est.shape[1]
            points, colors = utils.depth.raycast_to_world_point(intrinsics_highres, extrinsics_plane, highres_image, highres_mask)
            points, colors = utils.depth.voxel_downsample(points, colors, self.downsample_resolution / scale)
            points, colors = utils.depth.clipping_sphere(points, colors, np.array([0, 0, 0]), self.clipping_radius / scale)

            # write points as PLY format
            object_path = os.path.join(temp_dir, "object.ply")
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points)
            pcd.colors = open3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            open3d.io.write_point_cloud(object_path, pcd, write_ascii=False, compressed=True)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(fig_path, output.open())
            shutil.move(params_path, output.open())
            shutil.move(object_path, output.open())


class DepthAlignV2Task(luigi.Task):
    input_path: luigi.StrParameter = luigi.StrParameter()
    fps: luigi.IntParameter = luigi.IntParameter()
    width: luigi.IntParameter = luigi.IntParameter()
    height: luigi.IntParameter = luigi.IntParameter()
    max_keypoints: luigi.IntParameter = luigi.IntParameter()
    depth_confidence: luigi.FloatParameter = luigi.FloatParameter()
    width_confidence: luigi.FloatParameter = luigi.FloatParameter()
    init_frame_width: luigi.IntParameter = luigi.IntParameter()
    init_frame_height: luigi.IntParameter = luigi.IntParameter()
    init_focal_length: luigi.FloatParameter = luigi.FloatParameter()
    align_confidence: luigi.FloatParameter = luigi.FloatParameter()
    downsample_resolution: luigi.FloatParameter = luigi.FloatParameter()
    clipping_radius: luigi.FloatParameter = luigi.FloatParameter()

    def requires(self):
        object_masking = tasks.object_masking.ObjectMaskingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            mask_categories=tuple(utils.object_masking.PLANAR_CATEGORIES),
        )
        reconstruction = tasks.reconstruction.ReconstructionTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            max_keypoints=self.max_keypoints,
            width_confidence=self.width_confidence,
            depth_confidence=self.depth_confidence,
            init_frame_width=self.init_frame_width,
            init_frame_height=self.init_frame_height,
            init_focal_length=self.init_focal_length,
        )
        depth = DepthTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            max_keypoints=self.max_keypoints,
            width_confidence=self.width_confidence,
            depth_confidence=self.depth_confidence,
            init_frame_width=self.init_frame_width,
            init_frame_height=self.init_frame_height,
            init_focal_length=self.init_focal_length,
        )
        return [object_masking, reconstruction, depth]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[object_masking], [reconstruction], [depth]] = self.input()
            mask_dir = os.path.join(object_masking.read(), "masks")
            model_dir = os.path.join(reconstruction.read(), "model")
            depth_path = os.path.join(depth.read(), "results.npz")

            # extract reference model
            extrinsics_ref = []
            model = pycolmap.Reconstruction(model_dir)
            for image_id in sorted(model.images):
                image = model.image(image_id)
                extrinsics_ref.append(np.concat([image.cam_from_world().matrix(), [[0, 0, 0, 1]]]))
            extrinsics_ref = np.array(extrinsics_ref)
            # undistort mask
            mask = utils.depth.undistort_image_dir(model_dir, mask_dir, rgb=False)

            # extract estimated model
            depth_output = np.load(depth_path)
            image_est = depth_output["image"]
            depth_est = depth_output["depth"]
            conf_est = depth_output["conf"]
            extrinsics_est = depth_output["extrinsics"]
            intrinsics_est = depth_output["intrinsics"]

            # check image and mask consistency
            assert image_est.shape[0] == mask.shape[0], f"image length {image_est.shape[0]} does not match mask length {mask.shape[0]}"
            # resize mask to match downsampled image for depth estimation
            mask_est = []
            for i in range(mask.shape[0]):
                m = cv2.resize(mask[i], (image_est.shape[2], image_est.shape[1]), interpolation=cv2.INTER_AREA)
                mask_est.append(m)
            mask_est = np.array(mask_est)
            mask_est = (mask_est > 127).astype(np.uint8)

            # align model
            path_ref, path_est, scale = utils.depth.align_path_from_extrinsics(extrinsics_ref, extrinsics_est)
            depth_ref = depth_est / scale
            ctx.logger.info(f"align scale: {scale}")
            # debug aligned trajectory
            fig_path = os.path.join(temp_dir, "trajectory.png")
            utils.depth.plot_path(fig_path, path_ref, path_est)

            # projection dense depth
            idx = slice(None, None)
            points, colors = utils.depth.depth_to_world_point(depth_ref[idx], intrinsics_est[idx], extrinsics_ref[idx], image_est[idx], mask_est[idx], conf_est[idx], self.align_confidence)
            normals = utils.depth.compute_normals(points)
            ctx.logger.info(f"points: {points.shape} {points.dtype}, colors: {colors.shape} {colors.dtype}, normals: {normals.shape} {normals.dtype}")

            # write points as PLY format
            object_path = os.path.join(temp_dir, "object.ply")
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points)
            pcd.colors = open3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            pcd.normals = open3d.utility.Vector3dVector(normals)
            open3d.io.write_point_cloud(object_path, pcd, write_ascii=False, compressed=True)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(fig_path, output.open())
            shutil.move(object_path, output.open())
