import os
import shutil
import tempfile

import sklearn.linear_model
import sklearn.preprocessing
import luigi
import numpy as np
import open3d as o3d
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

    ransac_threshold: luigi.FloatParameter = luigi.FloatParameter()
    max_depth: luigi.FloatParameter = luigi.FloatParameter()
    downsample_res: luigi.FloatParameter = luigi.FloatParameter()

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
            image_est = depth_output["image"]
            extrinsics_est = depth_output["extrinsics"]
            intrinsics_est = depth_output["intrinsics"]
            # align model (guided to estimated quantinity)
            path_guide, path_est, rotate_g2e, translate_g2e, scale_g2e = utils.alignment.align_path_from_extrinsics(extrinsics_guide, extrinsics_est)
            ctx.logger.info(f"guide to est scale: {scale_g2e}")
            # debug aligned trajectory
            fig_path = os.path.join(temp_dir, "trajectory.png")
            utils.alignment.plot_path(fig_path, path_guide, path_est)

            # detect quadric surface
            pcd_guide = o3d.io.read_point_cloud(fused_guide_path)
            pcd_guide.scale(scale_g2e, [0.0, 0.0, 0.0])
            pcd_guide.rotate(rotate_g2e, center=[0.0, 0.0, 0.0])
            pcd_guide.translate(translate_g2e)
            # quadric surface fitting using RANSAC
            poly = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
            xyz_guide = np.asarray(pcd_guide.points)
            x, y, z = xyz_guide[:, 0], xyz_guide[:, 1], xyz_guide[:, 2]
            X, Y = poly.fit_transform(np.c_[x, z]), y
            ransac = sklearn.linear_model.RANSACRegressor(
                estimator=sklearn.linear_model.LinearRegression(),
                min_samples=6,  # 5 coef + 1 bias
                residual_threshold=self.ransac_threshold,
                max_trials=100000,
                random_state=42,
            )
            ransac.fit(X, Y)
            inlier_mask = ransac.inlier_mask_
            ctx.logger.info(f"ransac inlier ratio: {np.sum(inlier_mask) / len(inlier_mask)}")
            # # [DEBUG] show fitting surface
            # xx, zz = np.meshgrid(
            #     np.linspace(-10.0, 10.0, 100),
            #     np.linspace(-10.0, 10.0, 100)
            # )
            # yy = ransac.predict(poly.transform(np.c_[xx.ravel(), zz.ravel()]))
            # xyz = np.c_[xx.ravel(), yy, zz.ravel()]
            # pcd_surface = o3d.geometry.PointCloud()
            # pcd_surface.points = o3d.utility.Vector3dVector(xyz)
            # pcd_surface.paint_uniform_color([1.0, 0.0, 0.0])
            # # [DEBUG] show inliers and outliers
            # pcd_inliers = pcd_guide.select_by_index(np.where(inlier_mask)[0])
            # pcd_outliers = pcd_guide.select_by_index(np.where(inlier_mask)[0], invert=True)
            # pcd_inliers.paint_uniform_color([0.0, 1.0, 0.0])
            # pcd = pcd_inliers + pcd_outliers + pcd_surface

            # transform reference model
            pcd_ref = o3d.io.read_point_cloud(fused_path)
            pcd_ref.scale(scale_g2e, [0.0, 0.0, 0.0])
            pcd_ref.rotate(rotate_g2e, center=[0.0, 0.0, 0.0])
            pcd_ref.translate(translate_g2e)

            # read mask
            mask = utils.alignment.undistort_image_dir(model_dir, mask_dir, rgb=False)
            mask_est = []
            for i in range(mask.shape[0]):
                m = cv2.resize(mask[i], (image_est.shape[2], image_est.shape[1]), interpolation=cv2.INTER_AREA)
                mask_est.append(m)
            mask_est = (np.array(mask_est) > 127).astype(np.uint8)
            # projection to surface
            points, colors = utils.alignment.depth_to_surface(
                intrinsics_est,
                extrinsics_est,
                image_est,
                mask_est,
                ransac,
                poly,
                max_depth=self.max_depth,
            )
            pcd_est = o3d.geometry.PointCloud()
            pcd_est.points = o3d.utility.Vector3dVector(points)
            pcd_est.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            pcd_est = pcd_est.voxel_down_sample(voxel_size=self.downsample_res)

            # write points as PLY format
            object_path = os.path.join(temp_dir, "object.ply")
            pcd = pcd_ref + pcd_est
            o3d.io.write_point_cloud(object_path, pcd, write_ascii=False, compressed=True)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(fig_path, output.open())
            shutil.move(object_path, output.open())
