import json
import os
import pickle
import sqlite3
import subprocess

import cv2
import numpy as np
import pycolmap
import trimesh

import context
import utils.metric_depth
import utils.segmentation


CREATE_SCALE_TABLE = """CREATE TABLE IF NOT EXISTS scale (
    scale_coef REAL NOT NULL,
    scale_coef_iqr REAL NOT NULL)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY NOT NULL,
    extrinsics TEXT NOT NULL,
    intrinsics TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    data BLOB NOT NULL)"""

CREATE_SEGMENTATIONS_TABLE = """CREATE TABLE IF NOT EXISTS segmentations (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    image_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    mask BLOB NOT NULL)"""

CREATE_SPARSE_POINTS_TABLE = """CREATE TABLE IF NOT EXISTS sparse_points (
    id INTEGER PRIMARY KEY NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    z REAL NOT NULL,
    r INTEGER NOT NULL,
    g INTEGER NOT NULL,
    b INTEGER NOT NULL)"""

CREATE_DENSE_POINTS_TABLE = """CREATE TABLE IF NOT EXISTS dense_points (
    id INTEGER PRIMARY KEY NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    z REAL NOT NULL,
    r INTEGER NOT NULL,
    g INTEGER NOT NULL,
    b INTEGER NOT NULL)"""

CREATE_VERTS_TABLE = """CREATE TABLE IF NOT EXISTS verts (
    id INTEGER PRIMARY KEY NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    z REAL NOT NULL)"""

CREATE_FACES_TABLE = """CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY NOT NULL,
    v1 INTEGER NOT NULL,
    v2 INTEGER NOT NULL,
    v3 INTEGER NOT NULL)"""

CREATE_ALL = ";".join([
    CREATE_SCALE_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_SEGMENTATIONS_TABLE,
    CREATE_SPARSE_POINTS_TABLE,
    CREATE_DENSE_POINTS_TABLE,
    CREATE_VERTS_TABLE,
    CREATE_FACES_TABLE,
])


def finalize(conn: sqlite3.Connection, scale_path: str, segmentation_dir: str, dense_dir: str):
    # create tables
    conn.executescript(CREATE_ALL)

    # read metric depth scale
    with open(scale_path, "rb") as f:
        scale = pickle.load(f)
    assert isinstance(scale, utils.metric_depth.MetricDepthScale)
    conn.execute("INSERT INTO scale VALUES (?, ?)", (scale.scale_coef, scale.scale_coef_iqr))

    # read sparse reconstruction
    sparse_model_dir = os.path.join(dense_dir, "sparse")
    sparse_model = pycolmap.Reconstruction(sparse_model_dir)

    # read intrinsics matrix
    camera = sparse_model.camera(0)
    assert camera.model == pycolmap.CameraModelId.PINHOLE
    f1, f2, c1, c2 = camera.params
    intrinsics = np.array([[f1, 0, c1], [0, f2, c2], [0, 0, 1]])

    # read resolution
    width, height = camera.width, camera.height

    # foreach image
    for id in range(sparse_model.num_images()):

        # read extrinsics matrix
        image = sparse_model.image(id)
        extrinsics = np.concat([image.cam_from_world().matrix(), [[0, 0, 0, 1]]])

        # read image data
        image_path = os.path.join(dense_dir, "images", image.name)
        image_data = cv2.imread(image_path)
        assert isinstance(image_data, np.ndarray)
        _, blob = cv2.imencode(".png", image_data)

        # write images table
        extrinsics_json = json.dumps(extrinsics.flatten().tolist())
        intrinsics_json = json.dumps(intrinsics.flatten().tolist())
        conn.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?)",
            (id, extrinsics_json, intrinsics_json, width, height, blob.tobytes())
        )

        # read segmentations
        basename, _ = os.path.splitext(image.name)
        segmentation_path = os.path.join(segmentation_dir, basename + ".pkl")
        with open(segmentation_path, "rb") as f:
            segmentation_result = pickle.load(f)
        assert isinstance(segmentation_result, utils.segmentation.SegmentationResult)

        # write segmentations
        for annotation in segmentation_result.annotations:
            # read class name
            class_name = annotation.class_name
            # read class confidence
            confidence = annotation.confidence
            # read mask data
            mask_blob = annotation.mask_blob

            conn.execute(
                "INSERT INTO segmentations (image_id, class_name, confidence, mask) VALUES (?, ?, ?, ?)",
                (id, class_name, confidence, mask_blob)
            )

    # write sparse points
    rows = []
    for id in sparse_model.point3D_ids():
        point = sparse_model.point3D(id)
        x, y, z = point.xyz.astype(np.float32)
        r, g, b = point.color.astype(np.uint8)
        rows.append((id, float(x), float(y), float(z), int(r), int(g), int(b)))
    conn.executemany("INSERT INTO sparse_points VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    # write dense points
    dense_path = os.path.join(dense_dir, "fused.ply")
    dense_points = trimesh.load_scene(dense_path).to_geometry()
    assert isinstance(dense_points, trimesh.PointCloud)
    verts = dense_points.vertices.astype(np.float32)
    colors = dense_points.colors.astype(np.uint8)
    points = np.concatenate([verts, colors], axis=1)
    rows = [(i, float(x), float(y), float(z), int(r), int(g), int(b)) for i, (x, y, z, r, g, b, _) in enumerate(points)]
    conn.executemany("INSERT INTO dense_points VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    # write triangle mesh (verts and faces)
    mesh_path = os.path.join(dense_dir, "meshed-poisson.ply")
    mesh = trimesh.load_scene(mesh_path).to_geometry()
    assert isinstance(mesh, trimesh.Trimesh)
    verts = mesh.vertices.astype(np.float32)
    rows = [(i, float(x), float(y), float(z)) for i, (x, y, z) in enumerate(verts)]
    conn.executemany("INSERT INTO verts VALUES (?, ?, ?, ?)", rows)
    faces = mesh.faces.astype(np.int32)
    rows = [(i, int(v1), int(v2), int(v3)) for i, (v1, v2, v3) in enumerate(faces)]
    conn.executemany("INSERT INTO faces VALUES (?, ?, ?, ?)", rows)


def run_cubic_segmentation(db_path: str, seg_classes: list[str]):
    ctx = context.Context()

    prompt = ".".join(seg_classes)
    with subprocess.Popen(
        ["./cubic-segmentation", "raycast", db_path, prompt], cwd="deps/cubic-segmentation",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        if proc.stdout:
            for line in proc.stdout:
                ctx.console.print(line, end="")
        if proc.wait() != 0:
            raise RuntimeError(f"failed to run cubic-segmentation. database: {db_path}, prompt: {prompt}")
