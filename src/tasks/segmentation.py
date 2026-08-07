import os
import shutil
import tempfile

import luigi
import numpy as np
import open3d as o3d
import cv2

import context
import tasks.alignment
import tasks.object_masking
import tasks.reconstruction
import tasks.video_sampling
import utils.alignment
import utils.object_masking
import utils.segmentation
import utils.task


class SegmentationTask(luigi.Task):
    resources = { "gpu_vol": 1 }

    input_path: luigi.StrParameter = luigi.StrParameter()
    fps: luigi.IntParameter = luigi.IntParameter()
    highres_width: luigi.IntParameter = luigi.IntParameter()
    highres_height: luigi.IntParameter = luigi.IntParameter()

    def requires(self):
        video_sampling = tasks.video_sampling.VideoSamplingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.highres_width,
            height=self.highres_height,
        )
        return [video_sampling]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[video_sampling]] = self.input()
            image_dir = os.path.join(video_sampling.read(), "images")

            semantic_seg_dir = os.path.join(temp_dir, "semantic_seg")
            concept_seg_dir = os.path.join(temp_dir, "concept_seg")
            segmentation_dir = os.path.join(temp_dir, "segmentation")
            os.makedirs(semantic_seg_dir, exist_ok=True)
            os.makedirs(concept_seg_dir, exist_ok=True)
            os.makedirs(segmentation_dir, exist_ok=True)

            ctx.logger.info("running semantic segmentation")
            num_semantic_seg = len(utils.segmentation.CITYSCAPE_CATEGORIES)
            utils.segmentation.semantic_segmentation(image_dir, semantic_seg_dir)

            ctx.logger.info("running concept segmentation")
            num_concept_seg = len(utils.segmentation.CONCEPT_CATEGORIES)
            utils.segmentation.concept_segmentation(image_dir, concept_seg_dir, utils.segmentation.CONCEPT_CATEGORIES)

            ctx.logger.info("merging results")
            utils.segmentation.merge_segmentation(
                image_dir,
                semantic_seg_dir,
                concept_seg_dir,
                segmentation_dir
            )
            num_seg = num_semantic_seg + num_concept_seg
            ctx.logger.info(f"segmentation completed: {num_seg} categories")

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(segmentation_dir, output.open())


class LiftingTask(luigi.Task):
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
    voxel_downsample: luigi.FloatParameter = luigi.FloatParameter()

    kernel_radius: luigi.FloatParameter = luigi.FloatParameter()  # [0, 1]

    def requires(self):
        segmentation = SegmentationTask(
            input_path=self.input_path,
            fps=self.fps,
            highres_width=self.highres_width,
            highres_height=self.highres_height,
        )
        object_masking = tasks.object_masking.ObjectMaskingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.highres_width,
            height=self.highres_height,
            mask_categories=utils.object_masking.STATIC_CATEGORIES,
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
        alignment = tasks.alignment.AlignmentTask(
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
        surface = tasks.alignment.SurfaceTask(
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
            ransac_threshold=self.ransac_threshold,
            max_depth=self.max_depth,
            voxel_downsample=self.voxel_downsample
        )
        return [segmentation, object_masking, reconstruction, alignment, surface]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[segmentation], [object_masking], [reconstruction], [alignment], [surface]] = self.input()
            segmentation_dir = os.path.join(segmentation.read(), "segmentation")
            mask_dir = os.path.join(object_masking.read(), "masks")
            model_dir = os.path.join(reconstruction.read(), "model")
            alignment_path = os.path.join(alignment.read(), "alignment.npz")
            surface_path = os.path.join(surface.read(), "object.ply")

            # undistort images using COLMAP
            lowres_mask_dir = os.path.join(temp_dir, "masks")
            os.makedirs(lowres_mask_dir, exist_ok=True)
            for filename in os.listdir(segmentation_dir):
                src_msk_path = os.path.join(mask_dir, filename)
                src_msk = cv2.imread(src_msk_path, cv2.IMREAD_GRAYSCALE)
                assert isinstance(src_msk, np.ndarray)
                dst_msk = cv2.resize(src_msk, (self.width, self.height), interpolation=cv2.INTER_AREA)
                dst_msk_path = os.path.join(lowres_mask_dir, filename)
                cv2.imwrite(dst_msk_path, dst_msk)
            masks_gray, intrinsics = utils.alignment.undistort_image_dir(model_dir, lowres_mask_dir, rgb=False)
            masks_bool = masks_gray > 127

            # undistort images using COLMAP
            seg_ids = np.zeros_like(masks_bool, dtype=np.uint8)
            for i, label in enumerate(utils.segmentation.CITYSCAPE_PLUS_CATEGORIES):
                lowres_segmentation_dir = os.path.join(temp_dir, f"segmentation_{label}")
                os.makedirs(lowres_segmentation_dir, exist_ok=True)
                for filename in os.listdir(segmentation_dir):
                    src_seg_path = os.path.join(segmentation_dir, filename)
                    src_seg = cv2.imread(src_seg_path, cv2.IMREAD_GRAYSCALE)
                    assert isinstance(src_seg, np.ndarray)
                    src_seg = (src_seg == i).astype(np.uint8) * 255
                    dst_seg = cv2.resize(src_seg, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    dst_seg_path = os.path.join(lowres_segmentation_dir, filename)
                    cv2.imwrite(dst_seg_path, dst_seg)
                seg_gray, _ = utils.alignment.undistort_image_dir(model_dir, lowres_segmentation_dir, rgb=False)
                seg_ids[seg_gray > 127] = i
            ctx.logger.info(f"seg ids {seg_ids.shape}")

            # extract estimated model extrinsics
            alignment_result = np.load(alignment_path)
            extrinsics = alignment_result["extrinsics"]

            # project 2D segmentation to 3D point space
            num_frames, height, width = seg_ids.shape
            num_feats = len(utils.segmentation.CITYSCAPE_PLUS_CATEGORIES)
            segs_feats = np.zeros((num_frames, height, width, num_feats), dtype=np.float64)
            for i in range(num_feats):
                segs_feats[seg_ids == i, i] = 1.0
            rays, ray_feats = utils.segmentation.project_ray(
                intrinsics,
                extrinsics,
                segs_feats,
                masks_bool,
            )  # rays (N, 2, 3), ray_feats: (N, F)
            pcd_input = o3d.io.read_point_cloud(surface_path)
            xyz = np.asarray(pcd_input.points)
            bound = pcd_input.get_max_bound() - pcd_input.get_min_bound()
            kernel_radius = max(bound[0], bound[1], bound[2]) * self.kernel_radius
            xyz_feats = utils.segmentation.intersection(
                rays.astype(np.float64),
                ray_feats.astype(np.float64),
                xyz.astype(np.float64),
                kernel_radius,
            )
            ctx.logger.info(f"ray feats {xyz_feats.shape}")

            # write points as npz format
            xyz_feats_path = os.path.join(temp_dir, "xyz_feats.npz")
            np.savez_compressed(xyz_feats_path, feats=xyz_feats)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(xyz_feats_path, output.open())
