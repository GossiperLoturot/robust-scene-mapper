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
        return [object_masking, reconstruction, depth]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[object_masking], [reconstruction], [depth]] = self.input()
            mask_dir = os.path.join(object_masking.read(), "planar_masks")
            model_dir = os.path.join(reconstruction.read(), "model")
            depth_path = os.path.join(depth.read(), "results.npz")

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
            mask_resized = []
            for i in range(mask.shape[0]):
                mask_resized.append(cv2.resize(mask[i], (image_est.shape[2], image_est.shape[1]), interpolation=cv2.INTER_AREA))
            mask_resized = np.array(mask_resized)
            mask_resized = (mask_resized == 255).astype(np.uint8)

            # align model
            path_ref, path_est, scale = utils.depth.align_path_from_extrinsics(extrinsics_ref, extrinsics_est)
            depth_ref = depth_est / scale

            # debug aligned trajectory
            fig_path = os.path.join(temp_dir, "trajectory.png")
            utils.depth.plot_path(fig_path, path_ref, path_est)

            # use planar constraint model
            points, colors = utils.depth.depth_to_world_point(depth_ref, intrinsics_est, extrinsics_ref, image_est, mask_resized, conf_est, self.align_confidence)
            matrix = utils.depth.transform_for_optimize_plane(points)
            points = points @ matrix[:3, :3].T + matrix[:3, 3]

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
