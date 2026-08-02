import os

import cv2
import h5py
import luigi
import numpy as np
import open3d as o3d

import context
import tasks.alignment
import tasks.depth
import tasks.multiview_stereo
import tasks.object_masking
import tasks.reconstruction
import tasks.segmentation
import tasks.video_sampling
import utils.alignment
import utils.object_masking
import utils.segmentation
import utils.task


class MergeTask(luigi.Task):
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
    kernel_radius: luigi.FloatParameter = luigi.FloatParameter()

    def requires(self):
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
            voxel_downsample=self.voxel_downsample,
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
        lifting = tasks.segmentation.LiftingTask(
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
            voxel_downsample=self.voxel_downsample,
            kernel_radius=self.kernel_radius,
        )
        return [surface, alignment, lifting]

    def output(self):
        ctx = context.Context()
        return [utils.task.HDF5Target(ctx.export_dir, self)]

    def run(self):
        ctx = context.Context()

        [[surface], [alignment], [lifting]] = self.input()
        surface_path = os.path.join(surface.read(), "object.ply")
        tracking_path = os.path.join(surface.read(), "tracking.npz")
        alignment_path = os.path.join(alignment.read(), "alignment.npz")
        lifting_path = os.path.join(lifting.read(), "object.ply")

        ego_centers = []
        alignment_result = np.load(alignment_path)
        extrinsics = alignment_result["extrinsics"]
        for i in range(len(extrinsics)):
            position = np.linalg.inv(extrinsics[i])[:3, 3]
            ego_centers.append(position)
        ego_centers = np.array(ego_centers, dtype=np.float64)

        alt_frames, alt_labels, alt_centers = [], [], []
        tracking = np.load(tracking_path, allow_pickle=True)
        for i, results in enumerate(tracking["all_results"]):
            labels = results["labels"]
            position = results["centers"]
            alt_frames.append(i)
            alt_labels.append(labels)
            alt_centers.append(position)
        alt_frames = np.array(alt_frames, dtype=np.int64)
        alt_labels = np.concat(alt_labels, axis=0, dtype="T")
        alt_centers = np.concat(alt_centers, axis=0, dtype=np.float64)

        pcd_colored = o3d.io.read_point_cloud(surface_path)
        pcd_semantic = o3d.io.read_point_cloud(lifting_path)
        xyz = np.asarray(pcd_colored.points, dtype=np.float64)
        rgb = np.asarray(pcd_colored.colors, dtype=np.float64)
        semantic = np.asarray(pcd_semantic.colors, dtype=np.float64)

        semantic_labels = utils.segmentation.CITYSCAPE_CATEGORIES + utils.segmentation.CONCEPT_CATEGORIES
        hsv = cv2.cvtColor((semantic.reshape(1, -1, 3) * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)[0, :, :].astype(np.float64)
        semantic_ids = np.round(hsv[:, 0] / 180.0 * len(semantic_labels))
        semantic_ids[hsv[:, 1] < 127] = -1
        semantic_ids = semantic_ids.astype(np.int64)
        semantic_labels = np.array(semantic_labels, dtype="T")

        ctx.logger.info("writing output to database")
        [output] = self.output()
        with output.open() as f:
            f.attrs.update(self.param_kwargs)

            f.create_dataset("ego_centers", data=ego_centers, dtype=np.float32, compression="gzip")

            f.create_dataset("alt_frames", data=alt_frames, dtype=np.int32, compression="gzip")
            f.create_dataset("alt_labels", data=alt_labels, dtype=h5py.string_dtype(), compression="gzip")
            f.create_dataset("alt_centers", data=alt_centers, dtype=np.float32, compression="gzip")

            f.create_dataset("xyz", data=xyz, dtype=np.float32, compression="gzip")
            f.create_dataset("rgb", data=rgb, dtype=np.float32, compression="gzip")

            f.create_dataset("semantic_labels", data=semantic_labels, dtype=h5py.string_dtype(), compression="gzip")
            f.create_dataset("semantic_ids", data=semantic_ids, dtype=np.int32, compression="gzip")
