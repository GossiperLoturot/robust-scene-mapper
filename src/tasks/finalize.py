import os
import shutil
import sqlite3
import tempfile

import luigi

import context
import tasks.metric_depth
import tasks.multiview_stereo
import tasks.segmentation
import utils.finalize
import utils.task


class FinalizeTask(luigi.Task):
    input_path: luigi.StrParameter = luigi.StrParameter()
    fps: luigi.IntParameter = luigi.IntParameter()
    width: luigi.IntParameter = luigi.IntParameter()
    height: luigi.IntParameter = luigi.IntParameter()
    mask_classes: luigi.ListParameter = luigi.ListParameter()
    max_keypoints: luigi.IntParameter = luigi.IntParameter()
    depth_confidence: luigi.FloatParameter = luigi.FloatParameter()
    width_confidence: luigi.FloatParameter = luigi.FloatParameter()
    init_frame_width: luigi.IntParameter = luigi.IntParameter()
    init_frame_height: luigi.IntParameter = luigi.IntParameter()
    init_focal_length: luigi.FloatParameter = luigi.FloatParameter()
    seg_classes: luigi.ListParameter = luigi.ListParameter()

    def requires(self):
        metric_depth = tasks.metric_depth.MetricDepthTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            mask_classes=self.mask_classes,
            max_keypoints=self.max_keypoints,
            width_confidence=self.width_confidence,
            depth_confidence=self.depth_confidence,
            init_frame_width=self.init_frame_width,
            init_frame_height=self.init_frame_height,
            init_focal_length=self.init_focal_length,
        )
        segmentation = tasks.segmentation.SegmentationTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            mask_classes=self.mask_classes,
            max_keypoints=self.max_keypoints,
            width_confidence=self.width_confidence,
            depth_confidence=self.depth_confidence,
            init_frame_width=self.init_frame_width,
            init_frame_height=self.init_frame_height,
            init_focal_length=self.init_focal_length,
            seg_classes=self.seg_classes,
        )
        multiview_stereo = tasks.multiview_stereo.MultiviewStereoTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            mask_classes=self.mask_classes,
            max_keypoints=self.max_keypoints,
            width_confidence=self.width_confidence,
            depth_confidence=self.depth_confidence,
            init_frame_width=self.init_frame_width,
            init_frame_height=self.init_frame_height,
            init_focal_length=self.init_focal_length,
        )
        return [metric_depth, segmentation, multiview_stereo]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[metric_depth], [segmentation], [multiview_stereo]] = self.input()
            scale_path = os.path.join(metric_depth.read(), "scale.bin")
            segmentation_dir = os.path.join(segmentation.read(), "segmentation")
            dense_dir = os.path.join(multiview_stereo.read(), "dense")

            db_path = os.path.join(temp_dir, "finalize.db")
            with sqlite3.Connection(db_path) as conn:
                utils.finalize.finalize(conn, scale_path, segmentation_dir, dense_dir)
            ctx.logger.info("finalized database")

            ctx.logger.info("cubic-segmentation")
            utils.finalize.run_cubic_segmentation(db_path, self.seg_classes)

            [output] = self.output()
            shutil.move(db_path, output.open())
