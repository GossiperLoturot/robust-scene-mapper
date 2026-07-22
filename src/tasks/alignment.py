import os
import shutil
import tempfile

import cv2
import luigi
import numpy as np
import open3d
import open3d as o3d
import pycolmap
import scipy

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
        stereo_fusion_guide = tasks.multiview_stereo.StereoFusionTask(
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
            mask_categories=tuple(utils.object_masking.PLANAR_CATEGORIES),
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
        return [object_masking, reconstruction, depth, stereo_fusion_guide, stereo_fusion]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[object_masking], [reconstruction], [depth], [planar_stereo_fusion], [stereo_fusion]] = self.input()
            mask_dir = os.path.join(object_masking.read(), "masks")
            model_dir = os.path.join(reconstruction.read(), "model")
            depth_path = os.path.join(depth.read(), "results.npz")
            fused_guide_path = os.path.join(planar_stereo_fusion.read(), "fused.ply")
            fused_path = os.path.join(stereo_fusion.read(), "fused.ply")

            # extract reference model
            extrinsics_guide = []
            model = pycolmap.Reconstruction(model_dir)
            for image_id in sorted(model.images):
                image = model.image(image_id)
                extrinsics_guide.append(np.concat([image.cam_from_world().matrix(), [[0, 0, 0, 1]]]))
            extrinsics_guide = np.array(extrinsics_guide)
            # extract estimated model
            depth_output = np.load(depth_path)
            extrinsics_est = depth_output["extrinsics"]
            # align model (guided to estimated quantinity)
            path_guide, path_est, rotate_g2e, translate_g2e, scale_g2e = utils.alignment.align_path_from_extrinsics(
                extrinsics_guide,
                extrinsics_est
            )
            ctx.logger.info(f"guide to est scale: {scale_g2e}")
            # debug aligned trajectory
            fig_path = os.path.join(temp_dir, "trajectory.png")
            utils.alignment.plot_path(fig_path, path_guide, path_est)

            # detect plane segment
            pcd_guide = open3d.io.read_point_cloud(fused_guide_path)
            pcd_guide.scale(scale_g2e, [0.0, 0.0, 0.0])
            pcd_guide.rotate(rotate_g2e, center=[0.0, 0.0, 0.0])
            pcd_guide.translate(translate_g2e)
            model, _ = pcd_guide.segment_plane(distance_threshold=0.100, ransac_n=3, num_iterations=10000)
            # align plane normal to z-axis
            a, b, c, _ = model
            normal = np.array([a, b, c])
            normal /= np.linalg.norm(normal)
            rot, _ = scipy.spatial.transform.Rotation.align_vectors([[0.0, 0.0, 1.0]], [normal])
            rot_mat = rot.as_matrix()
            t = -rot_mat @ pcd_guide.get_center()
            hom_mat = np.eye(4)
            hom_mat[:3, :3] = rot_mat
            hom_mat[:3, 3] = t

            # transform reference model
            pcd_ref = open3d.io.read_point_cloud(fused_path)
            pcd_ref.scale(scale_g2e, [0.0, 0.0, 0.0])
            pcd_ref.rotate(rotate_g2e, center=[0.0, 0.0, 0.0])
            pcd_ref.translate(translate_g2e)
            pcd_ref.transform(hom_mat)

            # read depth
            depth = depth_output["depth"]
            image_est = depth_output["image"]
            conf_est = depth_output["conf"]
            intrinsics_est = depth_output["intrinsics"]
            # read mask
            mask = utils.alignment.undistort_image_dir(model_dir, mask_dir, rgb=False)
            mask_est = []
            for i in range(mask.shape[0]):
                m = cv2.resize(mask[i], (image_est.shape[2], image_est.shape[1]), interpolation=cv2.INTER_AREA)
                mask_est.append(m)
            mask_est = np.array(mask_est)
            mask_est = (mask_est > 127).astype(np.uint8)
            # projection (z = 0) to plane
            points, colors = utils.alignment.depth_to_plane(
                depth,
                intrinsics_est,
                extrinsics_est @ np.linalg.inv(hom_mat),
                image_est,
                mask_est,
                conf_est,
                self.align_confidence
            )
            pcd_est = open3d.geometry.PointCloud()
            pcd_est.points = open3d.utility.Vector3dVector(points)
            pcd_est.colors = open3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

            # write points as PLY format
            object_path = os.path.join(temp_dir, "object.ply")
            pcd = pcd_ref + pcd_est
            min_bound = np.array([-30.0, -30.0, -30.0])
            max_bound = np.array([ 30.0,  30.0,  30.0])
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            pcd = pcd.crop(bbox)
            pcd = pcd.voxel_down_sample(voxel_size=self.downsample_resolution)
            open3d.io.write_point_cloud(object_path, pcd, write_ascii=False, compressed=True)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(fig_path, output.open())
            shutil.move(object_path, output.open())
