import os

import cv2
import luigi
import msgpack
import numpy as np
import open3d as o3d
import pycolmap

import context
import tasks.alignment
import tasks.multiview_stereo
import tasks.object_masking
import tasks.reconstruction
import tasks.segmentation
import tasks.video_sampling
import utils.alignment
import utils.object_masking
import utils.segmentation
import utils.task


class ReconstructCommitTask(luigi.Task):
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

    def requires(self) -> list[luigi.Task]:
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

    def output(self) -> list[luigi.Target]:
        ctx = context.Context()
        return [utils.task.MsgpackTarget(ctx.database_dir, self)]

    def run(self) -> None:
        ctx = context.Context()

        [[video_sampling], [reconstruction]] = self.input()
        image_dir = os.path.join(video_sampling.read(), "images")
        model_dir = os.path.join(reconstruction.read(), "model")

        # extract reference model extrinsics
        extrinsics_base = []
        model = pycolmap.Reconstruction(model_dir)
        for image_id in sorted(model.images):
            image = model.image(image_id)
            extrinsics_base.append(np.concat([image.cam_from_world().matrix(), [[0, 0, 0, 1]]]))
        extrinsics_base = np.array(extrinsics_base)

        # read intrinsics and undistort images
        images_rgb, _ = utils.alignment.undistort_image_dir(model_dir, image_dir, rgb=True)

        # align
        b2a_mat = extrinsics_base[0]
        extrinsics = extrinsics_base @ np.linalg.inv(b2a_mat)

        ego_frames, ego_xyz = [], []
        for i in range(len(extrinsics)):
            xyz = np.linalg.inv(extrinsics[i])[:3, 3]
            ego_xyz.append(xyz)
        ego_frames = np.arange(len(ego_xyz), dtype=np.int32).tobytes()
        ego_xyz = np.concat(ego_xyz, dtype=np.float32).tobytes()
        bimages_rgb = list[bytes]()
        for image_rgb in images_rgb:
            _, bimage_rgb = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            bimage_rgb = bimage_rgb.tobytes()
            bimages_rgb.append(bimage_rgb)
        ego_images = bimages_rgb

        model = pycolmap.Reconstruction(model_dir)
        pcd_xyz, pcd_rgb = [], []
        for i in model.points3D:
            pcd_xyz.append(model.points3D[i].xyz)
            pcd_rgb.append(model.points3D[i].color)
        pcd_xyz = np.array(pcd_xyz, dtype=np.float32).tobytes()
        pcd_rgb = np.array(pcd_rgb, dtype=np.uint8).tobytes()

        # save as native byteorder.
        data = {
            "param": self.param_kwargs,
            "ego_frames": ego_frames,
            "ego_xyz": ego_xyz,
            "ego_images": ego_images,
            "pcd_xyz": pcd_xyz,
            "pcd_rgb": pcd_rgb,
        }
        ctx.logger.info("writing output to database")
        [output] = self.output()
        with output.open() as f:
            msgpack.pack(data, f, use_bin_type=True)


class SurfaceCommitTask(luigi.Task):
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

    def requires(self) -> list[luigi.Task]:
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
        return [surface, alignment]

    def output(self) -> list[luigi.Target]:
        ctx = context.Context()
        return [utils.task.MsgpackTarget(ctx.database_dir, self)]

    def run(self) -> None:
        ctx = context.Context()

        [[surface], [alignment]] = self.input()
        surface_path = os.path.join(surface.read(), "object.ply")
        tracking_path = os.path.join(surface.read(), "tracking.npz")
        alignment_path = os.path.join(alignment.read(), "alignment.npz")

        ego_frames, ego_xyz = [], []
        alignment_result = np.load(alignment_path)
        extrinsics = alignment_result["extrinsics"]
        for i in range(len(extrinsics)):
            xyz = np.linalg.inv(extrinsics[i])[:3, 3]
            ego_xyz.append(xyz)
        ego_frames = np.arange(len(ego_xyz), dtype=np.int32).tobytes()
        ego_xyz = np.array(ego_xyz, dtype=np.float32).tobytes()
        bimages_rgb = list[bytes]()
        for image_rgb in alignment_result["images_rgb"]:
            _, bimage_rgb = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            bimage_rgb = bimage_rgb.tobytes()
            bimages_rgb.append(bimage_rgb)
        ego_images = bimages_rgb

        alt_frames, alt_xyz, alt_labels = [], [], []
        tracking = np.load(tracking_path, allow_pickle=True)
        for i, results in enumerate(tracking["all_results"]):
            xyz, labels = results["centers"], results["labels"]
            alt_frames.append(i)
            alt_xyz.append(xyz)
            alt_labels.append(labels)
        alt_frames = np.array(alt_frames, dtype=np.int32).tobytes()
        alt_xyz = np.concat(alt_xyz, axis=0, dtype=np.float32).tobytes()
        alt_labels = np.concat(alt_labels, axis=0, dtype=np.dtypes.StringDType())
        alt_typename, alt_typemap = np.unique(alt_labels, return_inverse=True)
        alt_typemap = alt_typemap.astype(np.uint8).tobytes()
        alt_typename = alt_typename.tolist()

        pcd = o3d.io.read_point_cloud(surface_path)
        pcd_xyz = np.array(pcd.points, dtype=np.float32).tobytes()
        pcd_rgb = np.array(pcd.colors, dtype=np.float32)
        pcd_rgb = (pcd_rgb * 255.0).astype(np.uint8).tobytes()

        data = {
            "param": self.param_kwargs,
            "ego_frames": ego_frames,
            "ego_xyz": ego_xyz,
            "ego_images": ego_images,
            "alt_frames": alt_frames,
            "alt_xyz": alt_xyz,
            "alt_typemap": alt_typemap,
            "alt_typename": alt_typename,
            "pcd_xyz": pcd_xyz,
            "pcd_rgb": pcd_rgb,
        }
        ctx.logger.info("writing output to database")
        [output] = self.output()
        with output.open() as f:
            msgpack.pack(data, f, use_bin_type=True)


class LiftingCommitTask(luigi.Task):
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

    def requires(self) -> list[luigi.Task]:
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

    def output(self) -> list[luigi.Target]:
        ctx = context.Context()
        return [utils.task.MsgpackTarget(ctx.database_dir, self)]

    def run(self) -> None:
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
        ego_frames = np.arange(len(ego_xyz), dtype=np.int32).tobytes()
        ego_xyz = np.array(ego_xyz, dtype=np.float32).tobytes()
        bimages_rgb = list[bytes]()
        for image_rgb in alignment_result["images_rgb"]:
            _, bimage_rgb = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            bimage_rgb = bimage_rgb.tobytes()
            bimages_rgb.append(bimage_rgb)
        ego_images = bimages_rgb

        alt_frames, alt_xyz, alt_labels = [], [], []
        tracking = np.load(tracking_path, allow_pickle=True)
        for i, results in enumerate(tracking["all_results"]):
            xyz, labels = results["centers"], results["labels"]
            alt_frames.append(i)
            alt_xyz.append(xyz)
            alt_labels.append(labels)
        alt_frames = np.array(alt_frames, dtype=np.int32).tobytes()
        alt_xyz = np.concat(alt_xyz, axis=0, dtype=np.float32).tobytes()
        alt_labels = np.concat(alt_labels, axis=0, dtype=np.dtypes.StringDType())
        alt_typename, alt_typemap = np.unique(alt_labels, return_inverse=True)
        alt_typemap = alt_typemap.astype(np.uint8).tobytes()
        alt_typename = alt_typename.tolist()

        pcd = o3d.io.read_point_cloud(surface_path)
        pcd_xyz = np.array(pcd.points, dtype=np.float32).tobytes()
        pcd_rgb = np.array(pcd.colors, dtype=np.float32)
        pcd_rgb = (pcd_rgb * 255.0).astype(np.uint8).tobytes()

        lifting = np.load(lifting_path)
        feats = lifting["feats"]  # (N, M) where N is the number of points and M is the number of features
        feats = np.argmax(feats, axis=1)  # (N,) where each value is the index of the max feature
        feats[np.max(feats) == 0.0] = len(utils.segmentation.CITYSCAPE_PLUS_CATEGORIES) - 1  # set all zero features to `unknown`
        pcd_typemap = feats.astype(np.uint8).tobytes()
        pcd_typename = utils.segmentation.CITYSCAPE_PLUS_CATEGORIES

        data = {
            "param": self.param_kwargs,
            "ego_frames": ego_frames,
            "ego_xyz": ego_xyz,
            "ego_images": ego_images,
            "alt_frames": alt_frames,
            "alt_xyz": alt_xyz,
            "alt_typemap": alt_typemap,
            "alt_typename": alt_typename,
            "pcd_xyz": pcd_xyz,
            "pcd_rgb": pcd_rgb,
            "pcd_typemap": pcd_typemap,
            "pcd_typename": pcd_typename,
        }
        ctx.logger.info("writing output to database")
        [output] = self.output()
        with output.open() as f:
            msgpack.pack(data, f, use_bin_type=True)
