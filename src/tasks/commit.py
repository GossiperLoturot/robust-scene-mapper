import os
import tempfile

import cv2
import luigi
import msgpack
import numpy as np
import open3d as o3d
import pycolmap
import trimesh

import context
import tasks.alignment
import tasks.reconstruction
import tasks.segmentation
import tasks.video_sampling
import utils.alignment
import utils.commit
import utils.segmentation
import utils.task


class PrepareTaggingTask(luigi.Task):
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
        return [utils.task.FsFileTarget(ctx.database_dir, self, "msgpack")]

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

        # align to initial camera pose
        b2a_mat = extrinsics_base[0]
        extrinsics = extrinsics_base @ np.linalg.inv(b2a_mat)

        # read ego trajectory and images
        ego_frames, ego_xyz = [], []
        for i in range(len(extrinsics)):
            xyz = np.linalg.inv(extrinsics[i])[:3, 3]
            ego_xyz.append(xyz)
        ego_frames = np.arange(len(ego_xyz), dtype=np.int32)
        ego_xyz = np.array(ego_xyz, dtype=np.float32)
        bimages_rgb = list[bytes]()
        for image_rgb in images_rgb:
            _, bimage_rgb = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            bimage_rgb = bimage_rgb.tobytes()
            bimages_rgb.append(bimage_rgb)
        ego_images = bimages_rgb
        ctx.logger.info(f"ego_frames: {ego_frames.shape}, ego_xyz: {ego_xyz.shape}, ego_images: {len(ego_images)}")

        # read sparse point cloud
        model = pycolmap.Reconstruction(model_dir)
        pcd_xyz, pcd_rgb = [], []
        for i in model.points3D:
            pcd_xyz.append(model.points3D[i].xyz)
            pcd_rgb.append(model.points3D[i].color)
        pcd_xyz = np.array(pcd_xyz, dtype=np.float32)
        pcd_rgb = np.array(pcd_rgb, dtype=np.uint8)
        ctx.logger.info(f"pcd_xyz: {pcd_xyz.shape}, pcd_rgb: {pcd_rgb.shape}")

        # save as native byteorder.
        data = {
            "param": self.param_kwargs,
            "ego_frames": ego_frames.tobytes(),
            "ego_xyz": ego_xyz.tobytes(),
            "ego_images": ego_images,
            "pcd_xyz": pcd_xyz.tobytes(),
            "pcd_rgb": pcd_rgb.tobytes(),
        }
        ctx.logger.info("writing output to database")
        [output] = self.output()
        assert isinstance(output, utils.task.FsFileTarget)
        with output.open() as f:
            msgpack.pack(data, f, use_bin_type=True)


class PrepareCoordTask(luigi.Task):
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
    commit_tagging_path: luigi.StrParameter = luigi.StrParameter()

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
        return [utils.task.FsFileTarget(ctx.database_dir, self, "msgpack")]

    def run(self) -> None:
        ctx = context.Context()

        [[surface], [alignment]] = self.input()
        surface_path = os.path.join(surface.read(), "object.ply")
        alignment_path = os.path.join(alignment.read(), "alignment.npz")

        # read ego trajectory and images
        ego_frames, ego_xyz = [], []
        alignment_result = np.load(alignment_path)
        extrinsics = alignment_result["extrinsics"]
        for i in range(len(extrinsics)):
            xyz = np.linalg.inv(extrinsics[i])[:3, 3]
            ego_xyz.append(xyz)
        ego_frames = np.arange(len(ego_xyz), dtype=np.int32)
        ego_xyz = np.array(ego_xyz, dtype=np.float32)
        bimages_rgb = list[bytes]()
        for image_rgb in alignment_result["images_rgb"]:
            _, bimage_rgb = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            bimage_rgb = bimage_rgb.tobytes()
            bimages_rgb.append(bimage_rgb)
        ego_images = bimages_rgb
        ctx.logger.info(f"ego_frames: {ego_frames.shape}, ego_xyz: {ego_xyz.shape}, ego_images: {len(ego_images)}")

        # read dense point cloud
        pcd = o3d.io.read_point_cloud(surface_path)
        pcd_xyz = np.array(pcd.points, dtype=np.float32)
        pcd_rgb = np.array(pcd.colors, dtype=np.float32)
        pcd_rgb = (pcd_rgb * 255.0).astype(np.uint8)
        ctx.logger.info(f"pcd_xyz: {pcd_xyz.shape}, pcd_rgb: {pcd_rgb.shape}")

        # read commit tagging data
        with open(self.commit_tagging_path, "rb") as f:
            commit_tagging_data = msgpack.unpack(f)
        assert isinstance(commit_tagging_data, dict)
        tag = commit_tagging_data["tag"]

        data = {
            "param": self.param_kwargs,
            "tag": tag,
            "ego_frames": ego_frames.tobytes(),
            "ego_xyz": ego_xyz.tobytes(),
            "ego_images": ego_images,
            "pcd_xyz": pcd_xyz.tobytes(),
            "pcd_rgb": pcd_rgb.tobytes(),
        }
        ctx.logger.info("writing output to database")
        [output] = self.output()
        assert isinstance(output, utils.task.FsFileTarget)
        with output.open() as f:
            msgpack.pack(data, f, use_bin_type=True)


class PrepareAnnotationTask(luigi.Task):
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
    commit_tagging_path: luigi.StrParameter = luigi.StrParameter()

    kernel_radius: luigi.FloatParameter = luigi.FloatParameter()
    commit_coord_path: luigi.StrParameter = luigi.StrParameter()

    def requires(self) -> list[luigi.Task]:
        # read commit coord data
        with open(self.commit_coord_path, "rb") as f:
            commit_coord_data = msgpack.unpack(f)
        assert isinstance(commit_coord_data, dict)
        position = commit_coord_data["position"]
        scale = commit_coord_data["scale"]

        refine_surface = tasks.alignment.RefineSurfaceTask(
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
            position=position,
            scale=scale,
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
            position=position,
            scale=scale,
        )
        return [refine_surface, lifting]

    def output(self) -> list[luigi.Target]:
        ctx = context.Context()
        tracking = utils.task.FsFileTarget(ctx.database_dir, self, "msgpack")
        geometry = utils.task.FsFileTarget(ctx.database_dir, self, "ply")
        return [tracking, geometry]

    def run(self) -> None:
        ctx = context.Context()

        with tempfile.TemporaryDirectory() as temp_dir:
            [[refine_surface], [lifting]] = self.input()
            surface_path = os.path.join(refine_surface.read(), "object.ply")
            tracking_path = os.path.join(refine_surface.read(), "tracking.npz")
            alignment_path = os.path.join(refine_surface.read(), "alignment.npz")
            lifting_path = os.path.join(lifting.read(), "xyz_feats.npz")

            # read ego trajectory and images
            ego_frames, ego_xyz = [], []
            alignment_result = np.load(alignment_path)
            extrinsics = alignment_result["extrinsics"]
            for i in range(len(extrinsics)):
                xyz = np.linalg.inv(extrinsics[i])[:3, 3]
                ego_xyz.append(xyz)
            ego_frames = np.arange(len(ego_xyz), dtype=np.int32)
            ego_xyz = np.array(ego_xyz, dtype=np.float32)
            bimages_rgb = list[bytes]()
            for image_rgb in alignment_result["images_rgb"]:
                _, bimage_rgb = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                bimage_rgb = bimage_rgb.tobytes()
                bimages_rgb.append(bimage_rgb)
            ego_images = bimages_rgb
            ctx.logger.info(f"ego_frames: {ego_frames.shape}, ego_xyz: {ego_xyz.shape}, ego_images: {len(ego_images)}")

            # read tracking trajectory
            other_frames, other_xyz, other_labels = [], [], []
            tracking = np.load(tracking_path, allow_pickle=True)
            for i, results in enumerate(tracking["all_results"]):
                xyz, labels = results["centers"], results["labels"]
                other_frames.extend([i] * len(xyz))
                other_xyz.extend(xyz)
                other_labels.extend(labels)
            other_frames = np.array(other_frames, dtype=np.int32)
            other_xyz = np.array(other_xyz, dtype=np.float32)
            other_labels = np.array(other_labels, dtype=np.dtypes.StringDType())
            other_typename, other_typemap = np.unique(other_labels, return_inverse=True)
            other_typemap = other_typemap.astype(np.uint8)
            other_typename = other_typename.tolist()
            ctx.logger.info(f"other_frames: {other_frames.shape}, other_xyz: {other_xyz.shape}, other_typemap: {other_typemap.shape}, other_typename: {len(other_typename)}")

            # read dense point cloud
            pcd = o3d.io.read_point_cloud(surface_path)
            pcd_xyz = np.array(pcd.points, dtype=np.float32)
            pcd_rgb = np.array(pcd.colors, dtype=np.float32)
            pcd_rgb = (pcd_rgb * 255.0).astype(np.uint8)
            ctx.logger.info(f"pcd_xyz: {pcd_xyz.shape}, pcd_rgb: {pcd_rgb.shape}")

            # read dense point cloud semantic segmentation
            lifting = np.load(lifting_path)
            feats = lifting["feats"]  # (N, M) where N is the number of points and M is the number of features
            feats = np.argmax(feats, axis=1)  # (N,) where each value is the index of the max feature
            feats[np.max(feats) == 0.0] = len(utils.segmentation.CITYSCAPE_PLUS_CATEGORIES) - 1  # set all zero features to `unknown`
            pcd_typemap = feats.astype(np.uint8)
            pcd_typename = utils.segmentation.CITYSCAPE_PLUS_CATEGORIES
            ctx.logger.info(f"pcd_typemap: {pcd_typemap.shape}, pcd_typename: {len(pcd_typename)}")

            # semantic segmentation color as normals
            hues = np.linspace(0, 180, len(utils.segmentation.CITYSCAPE_PLUS_CATEGORIES), endpoint=False, dtype=np.uint8)
            palette = np.array([cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0, 0] for hue in hues])
            rgb = palette[feats]
            pcd.normals = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

            # read commit tagging data
            with open(self.commit_tagging_path, "rb") as f:
                commit_tagging_data = msgpack.unpack(f)
            assert isinstance(commit_tagging_data, dict)
            tag = commit_tagging_data["tag"]

            # output as msgpack file
            data = {
                "param": self.param_kwargs,
                "tag": tag,
                "ego_frames": ego_frames.tobytes(),
                "ego_xyz": ego_xyz.tobytes(),
                "ego_images": ego_images,
                "other_frames": other_frames.tobytes(),
                "other_xyz": other_xyz.tobytes(),
                "other_typemap": other_typemap.tobytes(),
                "other_typename": other_typename,
                "pcd_xyz": pcd_xyz.tobytes(),
                "pcd_rgb": pcd_rgb.tobytes(),
                "pcd_typemap": pcd_typemap.tobytes(),
                "pcd_typename": pcd_typename,
            }

            # output as PLY file
            ply_path = os.path.join(temp_dir, "object.ply")
            o3d.io.write_point_cloud(ply_path, pcd)

            ctx.logger.info("writing output to database")
            [tracking, geometry] = self.output()
            assert isinstance(tracking, utils.task.FsFileTarget)
            with tracking.open() as f:
                msgpack.pack(data, f, use_bin_type=True)
            assert isinstance(geometry, utils.task.FsFileTarget)
            with geometry.open() as f, open(ply_path, "rb") as g:
                f.write(g.read())


class PackTask(luigi.Task):
    prepare_path: luigi.StrParameter = luigi.StrParameter()
    commit_annotation_msgpack: luigi.StrParameter = luigi.StrParameter()
    commit_annotation_gltf: luigi.StrParameter = luigi.StrParameter()

    def output(self) -> list[luigi.Target]:
        ctx = context.Context()
        return [utils.task.FsFileTarget(ctx.database_dir, self, "msgpack")]

    def run(self) -> None:
        ctx = context.Context()

        with open(self.prepare_path, "rb") as f:
            prepare_data = msgpack.unpack(f)
        assert isinstance(prepare_data, dict)
        ctx.logger.info(f"prepare_data: {prepare_data.keys()}")

        with open(self.commit_annotation_msgpack, "rb") as f:
            commit_annotation_data = msgpack.unpack(f)
        assert isinstance(commit_annotation_data, dict)
        ctx.logger.info(f"commit_annotation_data: {commit_annotation_data.keys()}")

        # merge tracking data and commit data
        prepare_data["other_typemap"] = commit_annotation_data["other_typemap"]
        prepare_data["other_typename"] = commit_annotation_data["other_typename"]

        commit_annotation_geometry = trimesh.load_scene(self.commit_annotation_gltf)
        ctx.logger.info(f"commit_annotation_geometry: {commit_annotation_geometry.metadata}")

        pack_data = utils.commit.pack(prepare_data, commit_annotation_geometry)
        ctx.logger.info(f"pack_data: {pack_data.keys()}")

        ctx.logger.info("writing output to database")
        [output] = self.output()
        assert isinstance(output, utils.task.FsFileTarget)
        with output.open() as f:
            msgpack.pack(pack_data, f, use_bin_type=True)
