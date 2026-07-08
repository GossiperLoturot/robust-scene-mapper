import logging
import os
import subprocess

os.environ["TORCH_HOME"] = ".cache/torch"
os.environ["HF_HOME"] = ".cache/huggingface"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

import kornia
import rich
import rich.logging
import transformers


def main():
    console = rich.console.Console()

    handler = rich.logging.RichHandler(console=console)
    handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


    logger.info("download DISK weights")
    kornia.feature.DISK.from_pretrained("depth")

    logger.info("download LightGlue weights")
    kornia.feature.LightGlueMatcher("disk")

    logger.info("download Mask2Former weights")
    transformers.Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
    transformers.Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")

    logger.info("download Grounding Dino weights")
    transformers.GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    transformers.GroundingDinoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")

    logger.info("download SAM2 weights")
    transformers.Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
    transformers.Sam2Model.from_pretrained("facebook/sam2.1-hiera-large")

    logger.info("download Depth Anything 3 weights")
    with subprocess.Popen(
        ["uv", "run", "hf", "download", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"], cwd="deps/depth-anything-3",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        if proc.stdout:
            for line in proc.stdout:
                console.print(line, end="")
        if proc.wait() != 0:
            raise RuntimeError("failed to download Depth Anything 3 weights")

    logger.info("successfully downloaded all weights")


if __name__ == "__main__":
    main()
