import os
import pickle
import tempfile

import luigi

import context
import tasks.object_masking
import tasks.video_sampling
import utils.feature_matching
import utils.task


class FeatureMatchingTask(luigi.Task):
    resources = { "gpu_vol": 1 }

    input_path: luigi.StrParameter = luigi.StrParameter()
    fps: luigi.IntParameter = luigi.IntParameter()
    width: luigi.IntParameter = luigi.IntParameter()
    height: luigi.IntParameter = luigi.IntParameter()
    mask_classes: luigi.ListParameter = luigi.ListParameter()
    max_keypoints: luigi.IntParameter = luigi.IntParameter()
    depth_confidence: luigi.FloatParameter = luigi.FloatParameter()
    width_confidence: luigi.FloatParameter = luigi.FloatParameter()

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
            mask_classes=self.mask_classes,
        )
        return [video_sampling, object_masking]

    def output(self):
        ctx = context.Context()
        return [utils.task.DbTarget(ctx.database, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[video_sampling], [object_masking]] = self.input()
            with video_sampling.open_download() as f:
                f.extractall(temp_dir)
            image_dir = os.path.join(temp_dir, "images")
            with object_masking.open_download() as f:
                f.extractall(temp_dir)
            mask_dir = os.path.join(temp_dir, "masks")

            image_masks = utils.feature_matching.load_image_masks(image_dir, mask_dir)
            matching_pairs = utils.feature_matching.generate_matching_pairs(len(image_masks))

            # feature extract and matching
            matching_result = utils.feature_matching.extract_feature_and_match(
                image_masks=image_masks,
                matching_pairs=matching_pairs,
                max_keypoints=self.max_keypoints,
                depth_confidence=self.depth_confidence,
                width_confidence=self.width_confidence,
            )
            # save matching result
            matching_result_path = os.path.join(temp_dir, "matching_result.bin")
            with open(matching_result_path, "wb") as f:
                pickle.dump(matching_result, f)

            # create camera mapping
            camera_mapping = utils.feature_matching.create_camera_mapping(image_masks)
            # save camera mapping
            camera_mapping_path = os.path.join(temp_dir, "camera_mapping.bin")
            with open(camera_mapping_path, "wb") as f:
                pickle.dump(camera_mapping, f)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            with output.open_upload() as f:
                f.add(matching_result_path, "matching_result.bin")
                f.add(camera_mapping_path, "camera_mapping.bin")
