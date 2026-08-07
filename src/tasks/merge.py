import os

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
        lifting_path = os.path.join(lifting.read(), "xyz_feats.npz")

        ego_frames, ego_xyz = [], []
        alignment_result = np.load(alignment_path)
        extrinsics = alignment_result["extrinsics"]
        for i in range(len(extrinsics)):
            xyz = np.linalg.inv(extrinsics[i])[:3, 3]
            ego_xyz.append(xyz)
        ego_frames = np.arange(len(ego_xyz), dtype=np.int32)
        ego_xyz = np.array(ego_xyz, dtype=np.float32)
        ego_images = np.array(alignment_result["images_rgb"], dtype=np.uint8)

        alt_frames, alt_xyz, alt_labels = [], [], []
        tracking = np.load(tracking_path, allow_pickle=True)
        for i, results in enumerate(tracking["all_results"]):
            xyz, labels = results["centers"], results["labels"]
            alt_frames.append(i)
            alt_xyz.append(xyz)
            alt_labels.append(labels)
        alt_frames = np.array(alt_frames, dtype=np.int32)
        alt_xyz = np.concat(alt_xyz, axis=0).astype(np.float32)
        alt_labels = np.concat(alt_labels, axis=0, dtype="T")

        pcd = o3d.io.read_point_cloud(surface_path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        rgb = np.asarray(pcd.colors, dtype=np.float32)

        lifting = np.load(lifting_path)
        feats = lifting["feats"]
        feats_labels = np.array(utils.segmentation.CITYSCAPE_PLUS_CATEGORIES, dtype="T")

        ctx.logger.info("writing output to database")
        [output] = self.output()
        with output.open() as f:
            f.attrs.update(self.param_kwargs)

            f.create_dataset("ego_frames", data=ego_frames, dtype=np.int32, compression="gzip")
            f.create_dataset("ego_xyz", data=ego_xyz, dtype=np.float32, compression="gzip")
            f.create_dataset("ego_images", data=ego_images, dtype=np.uint8, compression="gzip")

            f.create_dataset("alt_frames", data=alt_frames, dtype=np.int32, compression="gzip")
            f.create_dataset("alt_xyz", data=alt_xyz, dtype=np.float32, compression="gzip")
            f.create_dataset("alt_labels", data=alt_labels, dtype=h5py.string_dtype(), compression="gzip")

            f.create_dataset("xyz", data=xyz, dtype=np.float32, compression="gzip")
            f.create_dataset("rgb", data=rgb, dtype=np.float32, compression="gzip")

            f.create_dataset("feats", data=feats, dtype=np.uint8, compression="gzip")
            f.create_dataset("feats_labels", data=feats_labels, dtype=h5py.string_dtype(), compression="gzip")
