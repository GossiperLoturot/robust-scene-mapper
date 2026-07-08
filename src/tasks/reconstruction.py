import os
import pickle
import shutil
import tempfile

import luigi

import context
import tasks.feature_matching
import tasks.video_sampling
import utils.feature_matching
import utils.reconstruction
import utils.task


class ReconstructionTask(luigi.Task):
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

    def requires(self):
        video_sampling = tasks.video_sampling.VideoSamplingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )
        feature_matching = tasks.feature_matching.FeatureMatchingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            max_keypoints=self.max_keypoints,
            width_confidence=self.width_confidence,
            depth_confidence=self.depth_confidence,
        )
        return [video_sampling, feature_matching]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[video_sampling], [feature_matching]] = self.input()
            image_dir = os.path.join(video_sampling.read(), "images")
            matching_result_path = os.path.join(feature_matching.read(), "matching_result.bin")
            camera_mapping_path = os.path.join(feature_matching.read(), "camera_mapping.bin")

            # load correspondence
            with open(matching_result_path, "rb") as f:
                matching_result = pickle.load(f)
            assert type(matching_result) is utils.feature_matching.FeatureMatchingResult

            # load camera ids
            with open(camera_mapping_path, "rb") as f:
                camera_mapping = pickle.load(f)
            assert type(camera_mapping) is utils.feature_matching.CameraMapping

            # create temporary directory
            db_path = os.path.join(temp_dir, "colmap.db")
            pairs_path = os.path.join(temp_dir, "pairs.txt")

            utils.reconstruction.upload_database(
                db_path=db_path,
                pairs_path=pairs_path,
                frame_width=self.init_frame_width,
                frame_height=self.init_frame_height,
                init_focal_length=self.init_focal_length,
                matching_result=matching_result,
                camera_mapping=camera_mapping,
            )

            # create models directory
            models_dir = os.path.join(temp_dir, "models")
            os.makedirs(models_dir, exist_ok=True)

            # create single model directory
            single_model_dir = os.path.join(temp_dir, "model")
            os.makedirs(single_model_dir, exist_ok=True)

            # run reconstruct
            while True:
                try:
                    utils.reconstruction.incremental_reconstruction(
                        db_path=db_path,
                        image_dir=image_dir,
                        input_model_dir=models_dir,
                        output_model_dir=single_model_dir,
                    )
                    break
                except Exception as e:
                    ctx.logger.warning(f"reconstruction failed with error: {e}, retrying...")

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(single_model_dir, output.open())
