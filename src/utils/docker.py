import os
import pathlib
import subprocess
import tempfile

import context


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GAUSSIAN_SPLATTING_DIR = pathlib.Path("/opt/gaussian-splatting")
GROUNDED_SAM_MAIN_PATH = REPO_ROOT / "deps" / "grounded-sam-2" / "code" / "main.py"


def _run_process(command: list[str], cwd: str | None = None):
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        if proc.stdout:
            for line in proc.stdout:
                print(line, end="", flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f"failed to run command: {command}")


def _run_colmap_cuda(data_dir: str):
    _run_process(["colmap", "patch_match_stereo", "--workspace_path", data_dir])
    _run_process(["colmap", "stereo_fusion", "--workspace_path", data_dir, "--output_path", os.path.join(data_dir, "fused.ply")])
    _run_process(["colmap", "delaunay_mesher", "--input_path", data_dir, "--output_path", os.path.join(data_dir, "meshed-delaunay.ply")])


def _run_depth_anything(data_dir: str):
    _run_process([
        "da3",
        "auto",
        os.path.join(data_dir, "input"),
        "--export-dir",
        os.path.join(data_dir, "output"),
        "--export-format",
        "mini_npz",
        "--process-res",
        "256",
        "--no-align-to-input-ext-scale",
        "--auto-cleanup",
    ])


def _run_grounded_sam(data_dir: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = os.path.join(temp_dir, "data")
        os.symlink(data_dir, data_path)
        _run_process(["python", str(GROUNDED_SAM_MAIN_PATH)], cwd=temp_dir)


def _run_gaussian_splatting(data_dir: str):
    output_dir = os.path.join(data_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = GAUSSIAN_SPLATTING_DIR / "utils" / "Depth-Anything-V2" / "checkpoints" / "depth_anything_v2_vitl.pth"
    if not checkpoint_path.exists():
        _run_process([
            "wget",
            "-O",
            str(checkpoint_path),
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
        ])
    _run_process(["ln", "-sfn", data_dir, str(GAUSSIAN_SPLATTING_DIR / "data")])
    _run_process(["ln", "-sfn", output_dir, str(GAUSSIAN_SPLATTING_DIR / "outputs")])
    _run_process(["python", "utils/estimate_dataset_depths.py", "data/input"], cwd=str(GAUSSIAN_SPLATTING_DIR))
    _run_process([
        "python",
        "main.py",
        "fit",
        "--config",
        "configs/depth_regularization/estimated_inverse_depth-l1.yaml",
        "--data.path",
        "data/input",
        "--data.image_on_cpu",
        "false",
        "--data.image_uint8",
        "true",
        "--data.parser",
        "Colmap",
        "-n",
        "3dgs",
    ], cwd=str(GAUSSIAN_SPLATTING_DIR))
    _run_process(["python", "utils/ckpt2ply.py", "outputs/3dgs"], cwd=str(GAUSSIAN_SPLATTING_DIR))


def run_stage(stage_name: str):
    _ = context.Context()
    if stage_name == "colmap-cuda":
        _run_colmap_cuda("/tmp/colmap-cuda")
    elif stage_name == "depth-anything-3":
        _run_depth_anything("/tmp/depth-anything-3")
    elif stage_name == "grounded-sam-2":
        _run_grounded_sam("/tmp/grounded-sam-2")
    elif stage_name == "gaussian-splatting":
        _run_gaussian_splatting("/tmp/gaussian-splatting")
    else:
        raise RuntimeError(f"unsupported runtime stage: {stage_name}")


def run_docker_compose(container_conf_dir: str):
    run_stage(pathlib.Path(container_conf_dir).name)
