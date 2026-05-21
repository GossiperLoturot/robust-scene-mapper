import json
import os
import pickle
import sqlite3
import subprocess

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import pycocotools.mask
import pycolmap

import utils.metric_depth


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

    # write metric depth scale
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
        segmentation_path = os.path.join(segmentation_dir, basename + ".json")
        with open(segmentation_path, "r") as f:
            segmentation_data = json.load(f)

        # write segmentations
        annotations = segmentation_data["annotations"]
        for annotation in annotations:
            # read class name
            class_name = annotation["class_name"]
            # read class confidence
            confidence = annotation["confidence"]
            # read mask data
            segmentation = pycocotools.mask.decode(annotation["segmentation"])
            _, blob = cv2.imencode(".png", segmentation * 255)

            conn.execute(
                "INSERT INTO segmentations (image_id, class_name, confidence, mask) VALUES (?, ?, ?, ?)",
                (id, class_name, confidence, blob.tobytes())
            )

    # write sparse points
    sparse_points = pd.DataFrame()
    for id in sparse_model.point3D_ids():
        point = sparse_model.point3D(id)
        x, y, z = point.xyz
        r, g, b = point.color
        sparse_points.loc[id, "x"] = x.astype(np.float32)
        sparse_points.loc[id, "y"] = y.astype(np.float32)
        sparse_points.loc[id, "z"] = z.astype(np.float32)
        sparse_points.loc[id, "r"] = r.astype(np.uint8)
        sparse_points.loc[id, "g"] = g.astype(np.uint8)
        sparse_points.loc[id, "b"] = b.astype(np.uint8)
    sparse_points.to_sql("sparse_points", conn, if_exists="append", index_label="id")

    # write dense points
    dense_path = os.path.join(dense_dir, "fused.ply")
    dense_points = o3d.io.read_point_cloud(dense_path)
    points = pd.DataFrame(np.asarray(dense_points.points).astype(np.float32), columns=["x", "y", "z"])
    colors = pd.DataFrame((np.asarray(dense_points.colors) * 255).astype(np.uint8), columns=["r", "g", "b"])
    dense_points = points.join(colors)
    dense_points.to_sql("dense_points", conn, if_exists="append", index_label="id")

    # write triangle mesh (verts and faces)
    mesh_path = os.path.join(dense_dir, "meshed-delaunay.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    verts = pd.DataFrame(np.asarray(mesh.vertices).astype(np.float32), columns=["x", "y", "z"])
    verts.to_sql("verts", conn, if_exists="append", index_label="id")
    faces = pd.DataFrame(np.asarray(mesh.triangles).astype(np.int32), columns=["v1", "v2", "v3"])
    faces.to_sql("faces", conn, if_exists="append", index_label="id")


def run_cubic_segmentation(bin_parent_dir: str, db_path: str, seg_classes: list[str]):
    prompt = ".".join(seg_classes)
    with subprocess.Popen(["./cubic-segmentation", "raycast", db_path, prompt], cwd=bin_parent_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        if proc.stdout:
            for line in proc.stdout:
                print(line, end="", flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f"failed to run cubic-segmentation. bin parent_dir: {bin_parent_dir}, database: {db_path}, prompt: {prompt}")
