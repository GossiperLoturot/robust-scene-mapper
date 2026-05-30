import argparse
import gzip
import json
import sqlite3
import tarfile
import tempfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWER_TEMPLATE_PATH = REPO_ROOT / "src" / "templates" / "viewer.html"
VERTICES_PER_FACE = 3


def _load_database_path(config_path: Path) -> str:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["global"]["database"]


def _connect_task_db(config_path: Path) -> sqlite3.Connection:
    database_path = _load_database_path(config_path)
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    return conn


def _extract_task_archive(conn: sqlite3.Connection, task_id: int, output_path: Path):
    row = conn.execute("SELECT artifact FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"task not found: {task_id}")
    artifact = row["artifact"]
    output_path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as temp_file:
        temp_file.write(artifact)
        temp_file.flush()
        with gzip.open(temp_file.name, "rb") as gz:
            with tarfile.open(fileobj=gz) as tar:
                tar.extractall(output_path)


def _load_mesh_data(finalize_db_path: Path) -> dict:
    with sqlite3.connect(finalize_db_path) as conn:
        vertex_rows = conn.execute("SELECT x, y, z FROM verts ORDER BY id").fetchall()
        index_rows = conn.execute("SELECT v1, v2, v3 FROM faces ORDER BY id").fetchall()
        class_rows = conn.execute("SELECT class_vec FROM class_vecs ORDER BY id").fetchall()
        point_rows = conn.execute("SELECT x, y, z, r, g, b FROM dense_points ORDER BY id").fetchall()

    vertices = []
    for x, y, z in vertex_rows:
        vertices.extend([float(x), float(y), float(z)])

    indices = []
    for v1, v2, v3 in index_rows:
        indices.extend([int(v1), int(v2), int(v3)])

    classes = [json.loads(class_vec) for (class_vec,) in class_rows]
    if classes:
        n_class = len(classes[0])
        vertex_classes = [[0.0 for _ in range(n_class)] for _ in range(len(vertex_rows))]
        for face_index, class_vec in enumerate(classes):
            face = indices[face_index * VERTICES_PER_FACE:face_index * VERTICES_PER_FACE + VERTICES_PER_FACE]
            for class_id in range(n_class):
                score = float(class_vec[class_id])
                vertex_classes[face[0]][class_id] += score
                vertex_classes[face[1]][class_id] += score
                vertex_classes[face[2]][class_id] += score
    else:
        vertex_classes = []

    point_vertices = []
    point_colors = []
    for x, y, z, r, g, b in point_rows:
        point_vertices.extend([float(x), float(y), float(z)])
        point_colors.extend([float(r) / 255.0, float(g) / 255.0, float(b) / 255.0])

    return {
        "vertices": vertices,
        "indices": indices,
        "classes": vertex_classes,
        "point_vertices": point_vertices,
        "point_colors": point_colors,
    }


def _render_view(task_conn: sqlite3.Connection, task_id: int, output_path: Path):
    with tempfile.TemporaryDirectory() as temp_dir:
        extract_dir = Path(temp_dir)
        _extract_task_archive(task_conn, task_id, extract_dir)
        mesh_data = _load_mesh_data(extract_dir / "finalize.db")
    template = VIEWER_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__MESH_DATA_JSON__", json.dumps(mesh_data, ensure_ascii=False))
    output_path.write_text(html, encoding="utf-8")


def _cmd_list(conn: sqlite3.Connection):
    rows = conn.execute("SELECT id, task_name, params_json FROM tasks ORDER BY id").fetchall()
    print("tasks:")
    for row in rows:
        print(f"\tid: {row['id']}, name: {row['task_name']}, params: {row['params_json']}")


def _cmd_unpack(conn: sqlite3.Connection, task_id: int, output_path: Path):
    _extract_task_archive(conn, task_id, output_path)
    print("extracted artifact.")


def _cmd_remove(conn: sqlite3.Connection, task_id: int):
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    print("removed task.")


def _cmd_view(conn: sqlite3.Connection, task_id: int, output_path: Path):
    _render_view(conn, task_id, output_path)
    print(f"generated html output: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")
    unpack_parser = subparsers.add_parser("unpack")
    unpack_parser.add_argument("--task-id", type=int, required=True)
    unpack_parser.add_argument("--output", default="output")

    remove_parser = subparsers.add_parser("remove")
    remove_parser.add_argument("--task-id", type=int, required=True)

    view_parser = subparsers.add_parser("view")
    view_parser.add_argument("--task-id", type=int, required=True)
    view_parser.add_argument("--output", default="output.html")

    args = parser.parse_args()
    config_path = Path(args.config)
    with _connect_task_db(config_path) as conn:
        if args.command == "list":
            _cmd_list(conn)
        elif args.command == "unpack":
            _cmd_unpack(conn, args.task_id, Path(args.output))
        elif args.command == "remove":
            _cmd_remove(conn, args.task_id)
        elif args.command == "view":
            _cmd_view(conn, args.task_id, Path(args.output))


if __name__ == "__main__":
    main()
