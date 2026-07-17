import os
import shutil
import tempfile

import luigi
import numpy as np
import open3d

import context
import tasks.depth
import tasks.multiview_stereo
import utils.task
import utils.object_masking


class MergeGeometryTask(luigi.Task):
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
    align_confidence: luigi.FloatParameter = luigi.FloatParameter()
    highres_width: luigi.IntParameter = luigi.IntParameter()
    highres_height: luigi.IntParameter = luigi.IntParameter()
    downsample_resolution: luigi.FloatParameter = luigi.FloatParameter()
    clipping_radius: luigi.FloatParameter = luigi.FloatParameter()

    def requires(self):
        depth_align = tasks.depth.DepthAlignTask(
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
            align_confidence=self.align_confidence,
            highres_width=self.highres_width,
            highres_height=self.highres_height,
            downsample_resolution=self.downsample_resolution,
            clipping_radius=self.clipping_radius,
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
        return [depth_align, stereo_fusion]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[depth_align], [stereo_fusion]] = self.input()
            bev_path = os.path.join(depth_align.read(), "object.ply")
            params_path = os.path.join(depth_align.read(), "params.npz")
            fused_path = os.path.join(stereo_fusion.read(), "fused.ply")

            bev = open3d.io.read_point_cloud(bev_path)
            fused = open3d.io.read_point_cloud(fused_path)
            with open(params_path, "rb") as f:
                params = np.load(f)
                world_to_plane, scale = params["world_to_plane"], params["scale"]
            ctx.logger.info("read bev and fused meshes")

            fused = fused.transform(world_to_plane)

            bev.scale(1.0 / scale)
            fused.scale(1.0 / scale)

            # write points as PLY format
            object_path = os.path.join(temp_dir, "object.ply")
            pcd = fused + bev
            open3d.io.write_point_cloud(object_path, pcd, write_ascii=False, compressed=True)

            [output] = self.output()
            shutil.move(object_path, output.open())
