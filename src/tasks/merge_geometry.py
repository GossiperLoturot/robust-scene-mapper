import os
import shutil
import tempfile

import luigi
import numpy as np
import trimesh

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
            mask_categories=utils.object_masking.STATIC_CATEGORIES,
        )
        return [depth_align, stereo_fusion]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[depth_align], [stereo_fusion]] = self.input()
            bev_path = os.path.join(depth_align.read(), "object.glb")
            params_path = os.path.join(depth_align.read(), "params.npz")
            fused_path = os.path.join(stereo_fusion.read(), "fused.ply")

            bev = trimesh.load(bev_path)
            fused = trimesh.load(fused_path)
            with open(params_path, "rb") as f:
                params = np.load(f)
                world_to_plane, scale = params["world_to_plane"], params["scale"]
            ctx.logger.info("read bev and fused meshes")

            fused = fused.apply_transform(world_to_plane)

            bev.apply_scale(1.0 / scale)
            fused.apply_scale(1.0 / scale)

            # write points as glTF 2.0 format
            object_path = os.path.join(temp_dir, "object.glb")
            scene = trimesh.Scene()
            scene.add_geometry(bev)
            scene.add_geometry(fused)
            scene.export(object_path)

            [output] = self.output()
            shutil.move(object_path, output.open())
