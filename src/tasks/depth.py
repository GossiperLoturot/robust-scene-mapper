import os
import shutil
import subprocess
import tempfile

import cv2
import luigi
import numpy as np
import pycolmap
import trimesh

import context
import tasks.object_masking
import tasks.reconstruction
import tasks.video_sampling
import utils.depth
import utils.task


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
    align_confidence: luigi.FloatParameter = luigi.FloatParameter()
    highres_width: luigi.IntParameter = luigi.IntParameter()
    highres_height: luigi.IntParameter = luigi.IntParameter()

    def requires(self):
        object_masking = tasks.object_masking.ObjectMaskingTask(
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
        )
        return [object_masking, reconstruction, depth, highres_video_sampling, highres_object_masking]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[object_masking], [reconstruction], [depth], [highres_video_sampling], [highres_object_masking]] = self.input()
            mask_dir = os.path.join(object_masking.read(), "planar_masks")
            model_dir = os.path.join(reconstruction.read(), "model")
            depth_path = os.path.join(depth.read(), "results.npz")
            highres_image_dir = os.path.join(highres_video_sampling.read(), "images")
            highres_mask_dir = os.path.join(highres_object_masking.read(), "planar_masks")

            # undistort mask
            undistort_dir = os.path.join(temp_dir, "undistort")
            os.makedirs(undistort_dir, exist_ok=True)
            pycolmap.undistort_images(undistort_dir, model_dir, mask_dir)
            undistort_mask_dir = os.path.join(undistort_dir, "images")
            # read mask
            mask = []
            for filename in sorted(os.listdir(undistort_mask_dir)):
                mask_path = os.path.join(undistort_mask_dir, filename)
                mask.append(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
            mask = np.array(mask)
            # cleanup
            shutil.rmtree(undistort_dir, ignore_errors=True)

            # extract reference model
            extrinsics_ref = []
            model = pycolmap.Reconstruction(model_dir)
            for image_id in sorted(model.images):
                image = model.image(image_id)
                extrinsics_ref.append(np.concat([image.cam_from_world().matrix(), [[0, 0, 0, 1]]]))
            extrinsics_ref = np.array(extrinsics_ref)

            # extract estimated model
            depth_output = np.load(depth_path)
            image_est = depth_output["image"]
            depth_est = depth_output["depth"]
            conf_est = depth_output["conf"]
            extrinsics_est = depth_output["extrinsics"]
            intrinsics_est = depth_output["intrinsics"]

            # check image and mask consistency
            assert image_est.shape[0] == mask.shape[0], f"image length {image_est.shape[0]} does not match mask length {mask.shape[0]}"
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

            # use planar constraint model
            points, colors = utils.depth.depth_to_world_point(depth_ref, intrinsics_est, extrinsics_ref, image_est, mask_est, conf_est, self.align_confidence)
            matrix = utils.depth.transform_for_optimize_plane(points)  # z = 0 plane
            extrinsics_plane = extrinsics_ref @ np.linalg.inv(matrix)

            # create highres model
            highres_model = pycolmap.Reconstruction(model_dir)
            for camera_id in highres_model.cameras:
                camera = highres_model.cameras[camera_id]
                camera.width = self.highres_width
                camera.height = self.highres_height
                camera.params[0] *= self.highres_width / self.width
                camera.params[1] *= self.highres_height / self.height
                camera.params[2] *= self.highres_width / self.width
                camera.params[3] *= self.highres_height / self.height
            highres_model_dir = os.path.join(temp_dir, "highres_model")
            os.makedirs(highres_model_dir, exist_ok=True)
            highres_model.write(highres_model_dir)

            # undistort highres image
            undistort_dir = os.path.join(temp_dir, "undistort")
            os.makedirs(undistort_dir, exist_ok=True)
            pycolmap.undistort_images(undistort_dir, highres_model_dir, highres_image_dir)
            undistort_highres_image_dir = os.path.join(undistort_dir, "images")
            # read highres image
            highres_image = []
            for filename in sorted(os.listdir(undistort_highres_image_dir)):
                image_path = os.path.join(undistort_highres_image_dir, filename)
                highres_image.append(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
            highres_image = np.array(highres_image)
            # cleanup
            shutil.rmtree(undistort_dir, ignore_errors=True)

            # undistort highres mask
            undistort_dir = os.path.join(temp_dir, "undistort")
            os.makedirs(undistort_dir, exist_ok=True)
            pycolmap.undistort_images(undistort_dir, highres_model_dir, highres_mask_dir)
            undistort_highres_mask_dir = os.path.join(undistort_dir, "images")
            # read highres mask
            highres_mask = []
            for filename in sorted(os.listdir(undistort_highres_mask_dir)):
                mask_path = os.path.join(undistort_highres_mask_dir, filename)
                highres_mask.append(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
            highres_mask = np.array(highres_mask)
            # cleanup
            shutil.rmtree(undistort_dir, ignore_errors=True)

            # erode mask to remove boundary
            highres_mask = (highres_mask > 127).astype(np.uint8)
            kernel = np.ones((8, 8), np.uint8)
            for i in range(highres_mask.shape[0]):
                highres_mask[i] = cv2.erode(highres_mask[i], kernel, iterations=1)

            # make highres bev
            intrinsics_est[:, 0, 0] *= highres_image.shape[2] / image_est.shape[2]
            intrinsics_est[:, 1, 1] *= highres_image.shape[1] / image_est.shape[1]
            intrinsics_est[:, 0, 2] *= highres_image.shape[2] / image_est.shape[2]
            intrinsics_est[:, 1, 2] *= highres_image.shape[1] / image_est.shape[1]
            points, colors = utils.depth.raycast_to_world_point(intrinsics_est, extrinsics_plane, highres_image, highres_mask)

            # write points as glTF 2.0 format
            object_path = os.path.join(temp_dir, "object.glb")
            scene = trimesh.Scene()
            point_cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
            scene.add_geometry(point_cloud)
            scene.export(object_path)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(fig_path, output.open())
            shutil.move(object_path, output.open())
