import os
import shutil
import tempfile

import luigi
import numpy as np
import open3d
import pycolmap
import cv2

import context
import tasks.depth
import tasks.multiview_stereo
import tasks.object_masking
import tasks.reconstruction
import utils.task
import utils.object_masking
import utils.alignment


class AlignmentTask(luigi.Task):
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
    highres_width: luigi.IntParameter = luigi.IntParameter()
    highres_height: luigi.IntParameter = luigi.IntParameter()
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
        depth = tasks.depth.DepthTask(
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
        stereo_fusion = tasks.multiview_stereo.StereoFusionTask(
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
            highres_width=self.highres_width,
            highres_height=self.highres_height,
            mask_categories=tuple(utils.object_masking.STATIC_CATEGORIES),
        )
        return [object_masking, reconstruction, depth, stereo_fusion]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[object_masking], [reconstruction], [depth], [stereo_fusion]] = self.input()
            mask_dir = os.path.join(object_masking.read(), "masks")
            model_dir = os.path.join(reconstruction.read(), "model")
            depth_path = os.path.join(depth.read(), "results.npz")
            fused_path = os.path.join(stereo_fusion.read(), "fused.ply")

            # extract reference model
            extrinsics_guide = []
            model = pycolmap.Reconstruction(model_dir)
            for image_id in sorted(model.images):
                image = model.image(image_id)
                extrinsics_guide.append(np.concat([image.cam_from_world().matrix(), [[0, 0, 0, 1]]]))
            extrinsics_guide = np.array(extrinsics_guide)
            # undistort mask
            mask = utils.alignment.undistort_image_dir(model_dir, mask_dir, rgb=False)

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

            # align model (guided to estimated quantinity)
            path_guide, path_est, rotate_g2e, translate_g2e, scale_g2e = utils.alignment.align_path_from_extrinsics(extrinsics_guide, extrinsics_est)
            ctx.logger.info(f"guide to est scale: {scale_g2e}")
            # debug aligned trajectory
            fig_path = os.path.join(temp_dir, "trajectory.png")
            utils.alignment.plot_path(fig_path, path_guide, path_est)
            # align guide point cloud
            pcd_guide = open3d.io.read_point_cloud(fused_path)
            pcd_guide.scale(scale_g2e, [0.0, 0.0, 0.0])
            pcd_guide.rotate(rotate_g2e, center=[0.0, 0.0, 0.0])
            pcd_guide.translate(translate_g2e)
            points_guide = np.asarray(pcd_guide.points)

            # projection dense depth
            points, colors, rays = utils.alignment.depth_to_world_point(depth_est, intrinsics_est, extrinsics_est, image_est, mask_est, conf_est, self.align_confidence)
            ctx.logger.info(f"points: {points.shape} {points.dtype}, colors: {colors.shape} {colors.dtype}")
            # alignment!
            points, colors, normals = utils.alignment.alignment_model(points, colors, rays, points_guide)

            # write points as PLY format
            object_path = os.path.join(temp_dir, "object.ply")
            pcd_est = open3d.geometry.PointCloud()
            pcd_est.points = open3d.utility.Vector3dVector(points)
            pcd_est.colors = open3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            pcd_est.normals = open3d.utility.Vector3dVector(normals)
            pcd = (pcd_est + pcd_guide).voxel_down_sample(voxel_size=self.downsample_resolution)
            open3d.io.write_point_cloud(object_path, pcd, write_ascii=False, compressed=True)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(fig_path, output.open())
            shutil.move(object_path, output.open())
