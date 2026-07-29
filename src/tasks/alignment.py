import os
import shutil
import tempfile

import luigi
import numpy as np
import open3d as o3d
import pycolmap
import scipy.spatial.transform

import context
import tasks.depth
import tasks.multiview_stereo
import tasks.object_masking
import tasks.reconstruction
import tasks.video_sampling
import utils.alignment
import utils.object_masking
import utils.task


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

    def requires(self):
        video_sampling = tasks.video_sampling.VideoSamplingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )
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
        return [video_sampling, object_masking, reconstruction, depth]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[video_sampling], [object_masking], [reconstruction], [depth]] = self.input()
            image_dir = os.path.join(video_sampling.read(), "images")
            mask_dir = os.path.join(object_masking.read(), "masks")
            model_dir = os.path.join(reconstruction.read(), "model")
            depth_path = os.path.join(depth.read(), "results.npz")

            # extract reference model extrinsics
            extrinsics_base = []
            model = pycolmap.Reconstruction(model_dir)
            for image_id in sorted(model.images):
                image = model.image(image_id)
                extrinsics_base.append(np.concat([image.cam_from_world().matrix(), [[0, 0, 0, 1]]]))
            extrinsics_base = np.array(extrinsics_base)
            # extract estimated model extrinsics
            depth_result = np.load(depth_path)
            extrinsics_metric = depth_result["extrinsics"]  # world to camera matrix

            # align model (guided to estimated quantinity)
            path_base, path_metric, scale_b2m = utils.alignment.align_path_from_extrinsics(extrinsics_base, extrinsics_metric)
            ctx.logger.info(f"guide to est scale: {scale_b2m}")

            # read intrinsics and rgb, mask images
            images_rgb, intrinsics = utils.alignment.undistort_image_dir(model_dir, image_dir, rgb=True)
            masks_gray, intrinsics = utils.alignment.undistort_image_dir(model_dir, mask_dir, rgb=False)
            masks_bool = masks_gray > 127

            # align z-axis for camera forward, y-axis for camera up
            flip_mat = np.eye(4)
            flip_mat[0, 0] = -1.0
            flip_mat[1, 1] = -1.0
            scale_mat = np.eye(4)
            scale_mat[:3, :3] *= scale_b2m
            b2a_mat = scale_mat @ flip_mat @ extrinsics_base[0]
            # compute trajectory normal
            trajectory = np.linalg.inv(extrinsics_base @ np.linalg.inv(b2a_mat))[:, :3, 3]
            center = np.mean(trajectory, axis=0)
            _, _, v = np.linalg.svd(trajectory - center, full_matrices=False)
            normal = v[2, :]
            normal = normal if normal[1] > 0.0 else -normal
            rot, _ = scipy.spatial.transform.Rotation.align_vectors([[0.0, 1.0, 0.0]], [normal])
            normal_mat = np.eye(4)
            normal_mat[:3, :3] = rot.as_matrix()
            # apply normal alignment
            b2a_mat = normal_mat @ scale_mat @ flip_mat @ extrinsics_base[0]

            # write alignment data as NPZ format
            alignment_path = os.path.join(temp_dir, "alignment.npz")
            extrinsics = extrinsics_base @ np.linalg.inv(b2a_mat)
            np.savez_compressed(
                alignment_path,
                b2a_mat=b2a_mat,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                images_rgb=images_rgb,
                masks_bool=masks_bool
            )

            # [DEBUG] aligned trajectory
            fig_path = os.path.join(temp_dir, "trajectory.png")
            utils.alignment.plot_path(fig_path, path_base, path_metric)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(alignment_path, output.open())
            shutil.move(fig_path, output.open())  # [DEBUG]


class SurfaceTask(luigi.Task):
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

    ransac_threshold: luigi.FloatParameter = luigi.FloatParameter()  # [0.0, 1.0]
    max_depth: luigi.FloatParameter = luigi.FloatParameter()  # [m]
    voxel_downsample: luigi.FloatParameter = luigi.FloatParameter()  # [0.0, 1.0]

    def requires(self):
        tracking = tasks.object_masking.TrackingTask(
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
        )
        alignment = AlignmentTask(
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
        return [tracking, alignment, stereo_fusion_guide, stereo_fusion]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[tracking], [alignment], [stereo_fusion_guide], [stereo_fusion]] = self.input()
            tracking_path = os.path.join(tracking.read(), "tracking.npz")
            alignment_path = os.path.join(alignment.read(), "alignment.npz")
            fused_guide_path = os.path.join(stereo_fusion_guide.read(), "fused.ply")
            fused_path = os.path.join(stereo_fusion.read(), "fused.ply")

            # extract estimated model extrinsics
            alignment_result = np.load(alignment_path)
            b2a_mat = alignment_result["b2a_mat"]
            intrinsics = alignment_result["intrinsics"]
            extrinsics = alignment_result["extrinsics"]
            images_rgb = alignment_result["images_rgb"]
            masks_bool = alignment_result["masks_bool"]

            # process tracking
            width, height = images_rgb.shape[2], images_rgb.shape[1]
            tracking_pickle = np.load(tracking_path, allow_pickle=True)
            all_centers, all_labels = [], []
            for i, results in enumerate(tracking_pickle["all_results"]):
                boxes = results["boxes"]
                labels = results["labels"]
                available_centers, available_labels = [], []
                for j in range(len(boxes)):
                    box = boxes[j]
                    label = labels[j]
                    if label not in utils.object_masking.COCO_RU_CATEGORIES:  # extract road users
                        continue
                    bottom_center = np.array([(box[0] + box[2]) * 0.5, box[3]])
                    if bottom_center[1] > utils.object_masking.EGO_VEHICLE_HLINE:  # ignore ego-vehicle
                        continue
                    bottom_center *= np.array([width, height])  # normalize to undistored pixel coordinates
                    available_centers.append(bottom_center)
                    available_labels.append(label)
                all_centers.append(np.array(available_centers).reshape(-1, 2))  # (N, 2)
                all_labels.append(available_labels)
            ctx.logger.info(f"tracking points: {sum([len(c) for c in all_centers])}")

            # detect thin plate spline
            pcd_guide = o3d.io.read_point_cloud(fused_guide_path)
            pcd_guide.transform(b2a_mat)
            # thin plate spline fitting using RANSAC
            xyz = np.asarray(pcd_guide.points)
            model, inliers = utils.alignment.fit_to_tps(xyz, self.ransac_threshold)
            ctx.logger.info(f"ransac inlier ratio: {np.sum(inliers) / len(inliers)}")

            # transform reference model
            pcd_ref = o3d.io.read_point_cloud(fused_path)
            pcd_ref.transform(b2a_mat)

            # project points to thin-plate-spline
            images_rgb = images_rgb.astype(np.float64) / 255.0
            points, colors = utils.alignment.project_points_to_tps(
                intrinsics,
                extrinsics,
                images_rgb,
                masks_bool,
                model,
                max_depth=self.max_depth,
            )
            pcd_surface = o3d.geometry.PointCloud()
            pcd_surface.points = o3d.utility.Vector3dVector(points)
            pcd_surface.colors = o3d.utility.Vector3dVector(colors)
            bound = pcd_surface.get_max_bound() - pcd_surface.get_min_bound()
            voxel_size = max(bound[0], bound[1], bound[2]) * self.voxel_downsample
            pcd_surface = pcd_surface.voxel_down_sample(voxel_size)

            # project trackings to thin-plate-spline
            all_centers = utils.alignment.project_tracking_to_tps(
                intrinsics,
                extrinsics,
                all_centers,
                model,
                max_depth=self.max_depth,
            )
            all_results = []
            for centers, labels in zip(all_centers, all_labels):
                all_results.append({ "centers": centers, "labels": labels })

            # write points as PLY format
            object_path = os.path.join(temp_dir, "object.ply")
            pcd = pcd_surface + pcd_ref
            o3d.io.write_point_cloud(object_path, pcd, write_ascii=False, compressed=True)

            # write points as npz format
            tracking_path = os.path.join(temp_dir, "tracking.npz")
            np.savez_compressed(tracking_path, all_results=all_results)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(object_path, output.open())
            shutil.move(tracking_path, output.open())
