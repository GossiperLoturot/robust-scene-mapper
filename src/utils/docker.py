import subprocess

import context


def run_docker_compose(container_conf_dir: str):
    _ = context.Context()

    # begin docker compose process
    with subprocess.Popen(["docker", "compose", "up", "-d"], cwd=container_conf_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        if proc.stdout:
            for line in proc.stdout:
                print(line, end="", flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f"failed to up docker compose {container_conf_dir}.")

    with subprocess.Popen(["docker", "compose", "logs", "-f"], cwd=container_conf_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        if proc.stdout:
            for line in proc.stdout:
                print(line, end="", flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f"failed to follow docker compose logs {container_conf_dir}.")

    # cleanup docker compose process
    with subprocess.Popen(["docker", "compose", "down"], cwd=container_conf_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        if proc.stdout:
            for line in proc.stdout:
                print(line, end="", flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f"failed to down docker compose {container_conf_dir}.")
