import os

import cv2
import luigi
import msgpack
import numpy as np
import open3d as o3d
import pycolmap
import scipy.spatial.transform

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
    random_downsample: luigi.FloatParameter = luigi.FloatParameter()
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
            random_downsample=self.random_downsample,
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
        quality_tag = commit_tagging_data["quality_tag"]

        data = {
            "param": self.param_kwargs,
            "quality_tag": quality_tag,
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
    extent: luigi.FloatParameter = luigi.FloatParameter()

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
        return [utils.task.FsFileTarget(ctx.database_dir, self, "msgpack")]

    def run(self) -> None:
        ctx = context.Context()

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
        pcd_h = np.array(pcd.normals, dtype=np.float32)[:, 0]
        ctx.logger.info(f"pcd_xyz: {pcd_xyz.shape}, pcd_rgb: {pcd_rgb.shape}, pcd_h: {pcd_h.shape}")

        # read dense point cloud semantic segmentation
        lifting = np.load(lifting_path)
        feats = lifting["feats"]  # (N, M) where N is the number of points and M is the number of features
        feats = np.argmax(feats, axis=1)  # (N,) where each value is the index of the max feature
        feats[np.max(feats) == 0.0] = len(utils.segmentation.CITYSCAPE_PLUS_CATEGORIES) - 1  # set all zero features to `unknown`
        pcd_typemap = feats.astype(np.uint8)
        pcd_typename = utils.segmentation.CITYSCAPE_PLUS_CATEGORIES
        ctx.logger.info(f"pcd_typemap: {pcd_typemap.shape}, pcd_typename: {len(pcd_typename)}")

        # crop point cloud
        pcd_idx = (pcd_xyz[:, 0] >= -self.extent) & (pcd_xyz[:, 0] <= self.extent) & (pcd_xyz[:, 2] >= -self.extent) & (pcd_xyz[:, 2] <= self.extent)
        pcd_xyz = pcd_xyz[pcd_idx]
        pcd_rgb = pcd_rgb[pcd_idx]
        pcd_h = pcd_h[pcd_idx]
        pcd_typemap = pcd_typemap[pcd_idx]

        # read commit tagging data
        with open(self.commit_tagging_path, "rb") as f:
            commit_tagging_data = msgpack.unpack(f)
        assert isinstance(commit_tagging_data, dict)
        quality_tag = commit_tagging_data["quality_tag"]

        # output as msgpack file
        data = {
            "param": self.param_kwargs,
            "quality_tag": quality_tag,
            "ego_frames": ego_frames.tobytes(),
            "ego_xyz": ego_xyz.tobytes(),
            "ego_images": ego_images,
            "other_frames": other_frames.tobytes(),
            "other_xyz": other_xyz.tobytes(),
            "other_typemap": other_typemap.tobytes(),
            "other_typename": other_typename,
            "pcd_xyz": pcd_xyz.tobytes(),
            "pcd_rgb": pcd_rgb.tobytes(),
            "pcd_h": pcd_h.tobytes(),
            "pcd_typemap": pcd_typemap.tobytes(),
            "pcd_typename": pcd_typename,
        }

        ctx.logger.info("writing output to database")
        [output] = self.output()
        assert isinstance(output, utils.task.FsFileTarget)
        with output.open() as f:
            msgpack.pack(data, f, use_bin_type=True)


class PackTask(luigi.Task):
    prepare_path: luigi.StrParameter = luigi.StrParameter()
    commit_annotation_path: luigi.StrParameter = luigi.StrParameter()
    extent: luigi.FloatParameter = luigi.FloatParameter()
    rasterize_res: luigi.IntParameter = luigi.IntParameter()
    rasterize_highres: luigi.IntParameter = luigi.IntParameter()

    def output(self) -> list[luigi.Target]:
        ctx = context.Context()
        return [utils.task.FsFileTarget(ctx.database_dir, self, "msgpack")]

    def run(self) -> None:
        ctx = context.Context()

        with open(self.prepare_path, "rb") as f:
            prepare_data = msgpack.unpack(f)
        assert isinstance(prepare_data, dict)
        ctx.logger.info(f"prepare_data: {prepare_data.keys()}")

        with open(self.commit_annotation_path, "rb") as f:
            commit_annotation_data = msgpack.unpack(f)
        assert isinstance(commit_annotation_data, dict)
        ctx.logger.info(f"commit_annotation_data: {commit_annotation_data.keys()}")

        # quality tag
        quality_tag = utils.commit.QualityTag(prepare_data["quality_tag"])
        quality_tag_str = [t.name for t in quality_tag]
        ctx.logger.info(f"quality_tag: {quality_tag_str}")

        # group tag
        group_tag = utils.commit.GroupTag(commit_annotation_data["group_tag"])
        group_tag_str = [t.name for t in group_tag]
        ctx.logger.info(f"group_tag: {group_tag_str}")

        # matrix for 2D projection [4, 4]
        mat = np.array([
            [0.5 * self.rasterize_highres / self.extent, 0.0, 0.0, 0.5 * self.rasterize_highres],
            [0.0, 0.0, 0.5 * self.rasterize_highres / self.extent, 0.5 * self.rasterize_highres],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        ctx.logger.info(f"create projection matrix: {mat}")

        # read point cloud data
        ctx.logger.info(f"reading point cloud")
        verts = np.frombuffer(prepare_data["pcd_xyz"], dtype=np.float32).reshape(-1, 3)
        verts_h = np.frombuffer(prepare_data["pcd_h"], dtype=np.float32).reshape(-1)
        class_ids = np.frombuffer(prepare_data["pcd_typemap"], dtype=np.uint8).reshape(-1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        _, idx = pcd.remove_statistical_outlier(nb_neighbors=32, std_ratio=2.0)
        verts, verts_h, class_ids = verts[idx], verts_h[idx], class_ids[idx]
        # extract points in different height levels
        ctx.logger.info(f"extracting points in different height levels")
        h0_mask = verts_h <= 0.25
        h1_mask = (verts_h <= 0.75) & (verts_h > 0.25)
        h2_mask = (verts_h <= 1.25) & (verts_h > 0.75)
        h3_mask = (verts_h <= 1.75) & (verts_h > 1.25)
        h4_mask = (verts_h <= 2.25) & (verts_h > 1.75)
        # rasterization
        ctx.logger.info(f"rasterizing point cloud to 2D maps")
        verts_hom = np.hstack([verts, np.ones((verts.shape[0], 1), dtype=np.float32)])
        verts_pix = (verts_hom @ mat.T)[:, [0, 1]]
        h0_map = utils.commit.rasterize_pcd(verts_pix[h0_mask], class_ids[h0_mask], self.rasterize_highres, self.rasterize_res)
        h1_map = utils.commit.rasterize_pcd(verts_pix[h1_mask], class_ids[h1_mask], self.rasterize_highres, self.rasterize_res)
        h2_map = utils.commit.rasterize_pcd(verts_pix[h2_mask], class_ids[h2_mask], self.rasterize_highres, self.rasterize_res)
        h3_map = utils.commit.rasterize_pcd(verts_pix[h3_mask], class_ids[h3_mask], self.rasterize_highres, self.rasterize_res)
        h4_map = utils.commit.rasterize_pcd(verts_pix[h4_mask], class_ids[h4_mask], self.rasterize_highres, self.rasterize_res)
        # encode
        ctx.logger.info(f"encoding rasterized maps to image")
        _, h0_map_enc = cv2.imencode(".png", h0_map)
        _, h1_map_enc = cv2.imencode(".png", h1_map)
        _, h2_map_enc = cv2.imencode(".png", h2_map)
        _, h3_map_enc = cv2.imencode(".png", h3_map)
        _, h4_map_enc = cv2.imencode(".png", h4_map)

        # create trajectory group
        ctx.logger.info("creating trajectory group")
        trajectory_group = dict[str, bytes]()
        # read ego trajectory
        ego_trajectory = np.frombuffer(prepare_data["ego_xyz"], dtype=np.float32).reshape(-1, 3)
        trajectory_group["ego"] = ego_trajectory.tobytes()
        # read other trajectory
        other_trajectory = np.frombuffer(prepare_data["other_xyz"], dtype=np.float32).reshape(-1, 3)
        other_trajectory_typemap = np.frombuffer(commit_annotation_data["other_typemap"], dtype=np.uint8)
        other_trajectory_typename = commit_annotation_data["other_typename"]
        for id in np.unique(other_trajectory_typemap):
            name = other_trajectory_typename[id]
            trajectory = other_trajectory[other_trajectory_typemap == id]
            trajectory_group[name] = trajectory.tobytes()

        # create lanes
        ctx.logger.info("creating lanes")
        lane_map = {}
        lane_typemap = commit_annotation_data["lane_typemap"]
        lane_xyz = np.array(commit_annotation_data["lane_xyz"], dtype=np.float32).reshape(-1, 3)
        for i in range(len(lane_typemap)):
            typ = lane_typemap[i]
            xyz = lane_xyz[i, :]

            lanes = lane_map.get(typ, [])
            lanes.append(xyz)
            lane_map[typ] = lanes
        # compute left and right lane center points for each lane type
        # (left, right from intersection center perspective)
        right_lane_xyz, left_lane_xyz, lane_ccw_map = {}, {}, {}
        for typ, lanes in lane_map.items():
            mean = np.mean(lanes, axis=0)
            shiftang = np.arctan2(mean[2], mean[0])
            shiftrot = scipy.spatial.transform.Rotation.from_rotvec(shiftang * np.array([0, 1, 0]))
            def key_func(shiftrot: scipy.spatial.transform.Rotation, x: np.ndarray) -> np.ndarray:
                shiftx = shiftrot.apply(x)
                return np.arctan2(shiftx[2], shiftx[0])
            lanes_ccw = sorted(lanes, key=lambda x: key_func(shiftrot, x), reverse=True)
            right_lane_xyz[typ] = (lanes_ccw[0] + lanes_ccw[1]) * 0.5
            left_lane_xyz[typ] = (lanes_ccw[-1] + lanes_ccw[-2]) * 0.5
            lane_ccw_map[typ] = lanes_ccw
        # connect lanes to form continuous lane lines
        ctx.logger.info("connecting lanes to form continuous lane lines")
        lane_lines = {}
        tangent_xyz = utils.commit.compute_tangent_xyz(ego_trajectory, window_length=9, polyorder=2)
        nearest_idx_in = utils.commit.find_nearest_idx(ego_trajectory, right_lane_xyz[utils.commit.LaneKind.IN.value])
        nearest_idx_out = utils.commit.find_nearest_idx(ego_trajectory, left_lane_xyz[utils.commit.LaneKind.OUT.value])
        # T shape intersection
        if group_tag & (utils.commit.GroupTag.TSHAPE_MINOR_IN | utils.commit.GroupTag.TSHAPE_MAJOR_IN):
            if utils.commit.LaneKind.IN_OPPOSITE in lane_ccw_map and utils.commit.LaneKind.OUT_OPPOSITE in lane_ccw_map:
                raise ValueError("Both IN_OPPOSITE and OUT_OPPOSITE lanes found for T shape intersection")
            # if exsits IN_OPPOSITE lane, connect IN and IN_OPPOSITE lanes, and connect OUT lane to intersection point.
            if utils.commit.LaneKind.IN_OPPOSITE in lane_ccw_map:
                ctx.logger.info("found IN_OPPOSITE lane, connecting IN and IN_OPPOSITE lanes")
                lane_line_in = utils.commit.connect_path_xyz(
                    right_lane_xyz[utils.commit.LaneKind.IN.value],
                    left_lane_xyz[utils.commit.LaneKind.IN_OPPOSITE.value],
                    tangent_xyz[nearest_idx_in],
                    interval=0.1,
                )
                lane_line_ino = utils.commit.connect_path_xyz(
                    left_lane_xyz[utils.commit.LaneKind.IN.value],
                    right_lane_xyz[utils.commit.LaneKind.IN_OPPOSITE.value],
                    tangent_xyz[nearest_idx_in],
                    interval=0.1,
                )[::-1, :]
                lane_line_out = utils.commit.connect_path_xyz(
                    left_lane_xyz[utils.commit.LaneKind.OUT.value],
                    utils.commit.compute_dual_intersection_xyz(
                        lane_line_in,
                        lane_line_ino,
                        left_lane_xyz[utils.commit.LaneKind.OUT.value],
                        -tangent_xyz[nearest_idx_out]
                    ),
                    -tangent_xyz[nearest_idx_out],
                    interval=0.1,
                )[::-1, :]
                lane_line_outo = utils.commit.connect_path_xyz(
                    right_lane_xyz[utils.commit.LaneKind.OUT.value],
                    utils.commit.compute_dual_intersection_xyz(
                        lane_line_in,
                        lane_line_ino,
                        right_lane_xyz[utils.commit.LaneKind.OUT.value],
                        -tangent_xyz[nearest_idx_out]
                    ),
                    -tangent_xyz[nearest_idx_out],
                    interval=0.1,
                )
                lane_lines[utils.commit.LaneKind.IN.name] = utils.commit.extend_path_xyz(lane_line_in, len0=5.0, len1=5.0, interval=0.1).tobytes()
                lane_lines[utils.commit.LaneKind.IN_OPPOSITE.name] = utils.commit.extend_path_xyz(lane_line_ino, len0=5.0, len1=5.0, interval=0.1).tobytes()
                lane_lines[utils.commit.LaneKind.OUT.name] = utils.commit.extend_path_xyz(lane_line_out, len0=0.0, len1=5.0, interval=0.1).tobytes()
                lane_lines[utils.commit.LaneKind.OUT_OPPOSITE.name] = utils.commit.extend_path_xyz(lane_line_outo, len0=5.0, len1=0.0, interval=0.1).tobytes()
            # if exsits OUT_OPPOSITE lane, connect OUT and OUT_OPPOSITE lanes, and connect IN lane to intersection point.
            elif utils.commit.LaneKind.OUT_OPPOSITE in lane_ccw_map:
                ctx.logger.info("found OUT_OPPOSITE lane, connecting OUT and OUT_OPPOSITE lanes")
                lane_line_out = utils.commit.connect_path_xyz(
                    left_lane_xyz[utils.commit.LaneKind.OUT.value],
                    right_lane_xyz[utils.commit.LaneKind.OUT_OPPOSITE.value],
                    -tangent_xyz[nearest_idx_out],
                    interval=0.1,
                )[::-1, :]
                lane_line_outo = utils.commit.connect_path_xyz(
                    right_lane_xyz[utils.commit.LaneKind.OUT.value],
                    left_lane_xyz[utils.commit.LaneKind.OUT_OPPOSITE.value],
                    -tangent_xyz[nearest_idx_out],
                    interval=0.1,
                )
                lane_line_in = utils.commit.connect_path_xyz(
                    right_lane_xyz[utils.commit.LaneKind.IN.value],
                    utils.commit.compute_dual_intersection_xyz(
                        lane_line_out,
                        lane_line_outo,
                        right_lane_xyz[utils.commit.LaneKind.IN.value],
                        tangent_xyz[nearest_idx_in]
                    ),
                    tangent_xyz[nearest_idx_in],
                    interval=0.1,
                )
                lane_line_ino = utils.commit.connect_path_xyz(
                    left_lane_xyz[utils.commit.LaneKind.IN.value],
                    utils.commit.compute_dual_intersection_xyz(
                        lane_line_out,
                        lane_line_outo,
                        left_lane_xyz[utils.commit.LaneKind.IN.value],
                        tangent_xyz[nearest_idx_in]
                    ),
                    tangent_xyz[nearest_idx_in],
                    interval=0.1,
                )[::-1, :]
                lane_lines[utils.commit.LaneKind.OUT.name] = utils.commit.extend_path_xyz(lane_line_out, len0=5.0, len1=5.0, interval=0.1).tobytes()
                lane_lines[utils.commit.LaneKind.OUT_OPPOSITE.name] = utils.commit.extend_path_xyz(lane_line_outo, len0=5.0, len1=5.0, interval=0.1).tobytes()
                lane_lines[utils.commit.LaneKind.IN.name] = utils.commit.extend_path_xyz(lane_line_in, len0=5.0, len1=0.0, interval=0.1).tobytes()
                lane_lines[utils.commit.LaneKind.IN_OPPOSITE.name] = utils.commit.extend_path_xyz(lane_line_ino, len0=0.0, len1=5.0, interval=0.1).tobytes()
            else:
                raise ValueError("No opposite lane found for T shape intersection")
        # Cross shape intersection
        else:
            if utils.commit.LaneKind.IN_OPPOSITE not in lane_ccw_map or utils.commit.LaneKind.OUT_OPPOSITE not in lane_ccw_map:
                raise ValueError("Both IN_OPPOSITE and OUT_OPPOSITE lanes must be found for Cross shape intersection")
            ctx.logger.info("found both IN_OPPOSITE and OUT_OPPOSITE lanes, connecting lanes to form continuous lane lines")
            lane_line_in = utils.commit.connect_path_xyz(
                right_lane_xyz[utils.commit.LaneKind.IN.value],
                left_lane_xyz[utils.commit.LaneKind.IN_OPPOSITE.value],
                tangent_xyz[nearest_idx_in],
                interval=0.1,
            )
            lane_line_ino = utils.commit.connect_path_xyz(
                left_lane_xyz[utils.commit.LaneKind.IN.value],
                right_lane_xyz[utils.commit.LaneKind.IN_OPPOSITE.value],
                tangent_xyz[nearest_idx_in],
                interval=0.1,
            )[::-1, :]
            lane_line_out = utils.commit.connect_path_xyz(
                left_lane_xyz[utils.commit.LaneKind.OUT.value],
                right_lane_xyz[utils.commit.LaneKind.OUT_OPPOSITE.value],
                -tangent_xyz[nearest_idx_out],
                interval=0.1,
            )[::-1, :]
            lane_line_outo = utils.commit.connect_path_xyz(
                right_lane_xyz[utils.commit.LaneKind.OUT.value],
                left_lane_xyz[utils.commit.LaneKind.OUT_OPPOSITE.value],
                -tangent_xyz[nearest_idx_out],
                interval=0.1,
            )
            lane_lines[utils.commit.LaneKind.IN.name] = utils.commit.extend_path_xyz(lane_line_in, len0=5.0, len1=5.0, interval=0.1).tobytes()
            lane_lines[utils.commit.LaneKind.IN_OPPOSITE.name] = utils.commit.extend_path_xyz(lane_line_ino, len0=5.0, len1=5.0, interval=0.1).tobytes()
            lane_lines[utils.commit.LaneKind.OUT.name] = utils.commit.extend_path_xyz(lane_line_out, len0=5.0, len1=5.0, interval=0.1).tobytes()
            lane_lines[utils.commit.LaneKind.OUT_OPPOSITE.name] = utils.commit.extend_path_xyz(lane_line_outo, len0=5.0, len1=5.0, interval=0.1).tobytes()

        # center offset ego trajectory (-: left, +: right)
        ctx.logger.info("computing lane offset for ego trajectory")
        lane_offset_in = utils.commit.compute_center_offset_xyz(
            right_lane_xyz[utils.commit.LaneKind.IN.value],
            tangent_xyz[nearest_idx_in],
            ego_trajectory
        )
        lane_offset_out = utils.commit.compute_center_offset_xyz(
            left_lane_xyz[utils.commit.LaneKind.OUT.value],
            tangent_xyz[nearest_idx_out],
            ego_trajectory
        )
        ctx.logger.info(f"lane_offset_in: {lane_offset_in}, lane_offset_out: {lane_offset_out}")

        # length of lane
        ctx.logger.info("computing lane width")
        lane_width_in = utils.commit.compute_path_width_xyz(
            lane_ccw_map[utils.commit.LaneKind.IN.value][1],
            lane_ccw_map[utils.commit.LaneKind.IN.value][0],
            tangent_xyz[nearest_idx_in]
        )
        lane_width_out = utils.commit.compute_path_width_xyz(
            lane_ccw_map[utils.commit.LaneKind.OUT.value][-2],
            lane_ccw_map[utils.commit.LaneKind.OUT.value][-1],
            tangent_xyz[nearest_idx_out]
        )
        ctx.logger.info(f"lane_width_in: {lane_width_in}, lane_width_out: {lane_width_out}")
        # num of lane
        lane_num_in = max(0, len(lane_ccw_map[utils.commit.LaneKind.IN.value]) - 1)
        lane_num_out = max(0, len(lane_ccw_map[utils.commit.LaneKind.OUT.value]) - 1)
        ctx.logger.info(f"lane_num_in: {lane_num_in}, lane_num_out: {lane_num_out}")

        # create lines
        ctx.logger.info("creating lines")
        lines = []
        line_typemap = commit_annotation_data["line_typemap"]
        line_xyz = np.array(commit_annotation_data["line_xyz"], dtype=np.float32).reshape(-1, 3, 3)
        for i in range(len(line_typemap)):
            lines.append({
                "type": utils.commit.LineKind(line_typemap[i]).name,
                "xyz": line_xyz[i, 0, :].tolist(),
                "forward": line_xyz[i, 1, :].tolist(),
                "right": line_xyz[i, 2, :].tolist(),
            })

        # pack data
        pack_data = {
            "param": prepare_data["param"],
            "quality_tag": quality_tag_str,
            "group_tag": group_tag_str,
            # rasterize
            "resolution": self.rasterize_highres,
            "xyz_to_pix": mat.tobytes(),
            "h0_map": h0_map_enc.tobytes(),
            "h1_map": h1_map_enc.tobytes(),
            "h2_map": h2_map_enc.tobytes(),
            "h3_map": h3_map_enc.tobytes(),
            "h4_map": h4_map_enc.tobytes(),
            # structure
            "trajectory_group": trajectory_group,
            "lane_lines": lane_lines,
            "lane_offset_in": lane_offset_in,
            "lane_offset_out": lane_offset_out,
            "lane_width_in": lane_width_in,
            "lane_width_out": lane_width_out,
            "lane_num_in": lane_num_in,
            "lane_num_out": lane_num_out,
            "lines": lines,
            # pcd
            "pcd_xyz": prepare_data["pcd_xyz"],
            "pcd_rgb": prepare_data["pcd_rgb"],
            "pcd_typemap": prepare_data["pcd_typemap"],
            "pcd_typename": prepare_data["pcd_typename"],
        }
        ctx.logger.info(f"pack_data: {pack_data.keys()}")

        ctx.logger.info("writing output to database")
        [output] = self.output()
        assert isinstance(output, utils.task.FsFileTarget)
        with output.open() as f:
            msgpack.pack(pack_data, f, use_bin_type=True)
