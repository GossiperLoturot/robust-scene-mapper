import os
import shutil
import tempfile

import luigi
import cv2

import context
import utils.task


class VideoSamplingTask(luigi.Task):
    input_path: luigi.StrParameter = luigi.StrParameter()
    fps: luigi.IntParameter = luigi.IntParameter()
    width: luigi.IntParameter = luigi.IntParameter()
    height: luigi.IntParameter = luigi.IntParameter()

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = os.path.join(temp_dir, "images")
            os.makedirs(image_dir, exist_ok=True)

            video_capture = cv2.VideoCapture(self.input_path)
            try:
                total_n, sampling_n = 0, 0
                interval = video_capture.get(cv2.CAP_PROP_FPS) / self.fps
                while True:
                    ok, frame = video_capture.read()
                    if not ok:
                        break

                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    frame_path = os.path.join(image_dir, f"{sampling_n:04d}.png")
                    cv2.imwrite(frame_path, frame)

                    total_n += 1
                    sampling_n += 1

                    # skip frames
                    next_n = int(round(sampling_n * interval))
                    while total_n < next_n:
                        ok = video_capture.grab()
                        if not ok:
                            break

                        total_n += 1
            finally:
                video_capture.release()

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(image_dir, output.open())
