import datetime

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import trimesh

import context


# pts2d: (N, 2) array, class_ids: (N,) array.
def pts2map(pts2d: np.ndarray, class_ids: np.ndarray, resolution: int) -> np.ndarray:
    x = np.clip(np.round(pts2d[:, 0]), 0, resolution - 1).astype(np.int32)
    y = np.clip(np.round(pts2d[:, 1]), 0, resolution - 1).astype(np.int32)

    pixel_idx = y * resolution + x
    max_cls = class_ids.max() + 1
    combined_keys = pixel_idx * max_cls + class_ids
    unique_keys, counts = np.unique(combined_keys, return_counts=True)
    unq_pixel_idx = unique_keys // max_cls
    unq_class_ids = unique_keys % max_cls

    class_image = np.zeros(resolution * resolution, dtype=np.uint8)
    max_counts = np.zeros(resolution * resolution, dtype=np.int32)
    for p_idx, c_id, count in zip(unq_pixel_idx, unq_class_ids, counts, strict=True):
        if count > max_counts[p_idx]:
            max_counts[p_idx] = count
            class_image[p_idx] = c_id
    return class_image.reshape((resolution, resolution))


def pack(data: dict, pcd: o3d.geometry.PointCloud, scene: trimesh.Scene, resolution: int = 512, extent: float = 10.0) -> None:
    ctx = context.Context()

    ctx.logger.info(f"in pack function: {data.keys()}")
    ctx.logger.info(f"pcd: {pcd}")
    ctx.logger.info(f"scene: {scene}")

    ctx.logger.info(f"create output map: {resolution}x{resolution}")
    road_same_map = np.zeros((resolution, resolution), dtype=np.uint8)
    road_opposite_map = np.zeros((resolution, resolution), dtype=np.uint8)
    crossing_map = np.zeros((resolution, resolution), dtype=np.uint8)

    # create rerun session
    timecode = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    session_name = f"rerun_robustscenemapper_{timecode}"
    rr.init(session_name)
    rr.connect_grpc()
    ctx.logger.info(f"rerun session: {session_name}")

    # matrix for 2D projection
    mat = np.array([[0.5 * resolution / extent, 0.0], [0.0, 0.0], [0.0, 0.5 * resolution / extent], [0.5 * resolution, 0.5 * resolution]], dtype=np.float32)

    # show pcd
    positions = np.frombuffer(data["pcd_xyz"], dtype=np.float32).reshape(-1, 3)
    class_ids = np.frombuffer(data["pcd_typemap"], dtype=np.uint8).reshape(-1)
    semrgb = np.array(pcd.normals, dtype=np.float32).reshape(-1, 3)
    mask_l0 = positions[:, 1] <= 0.25
    mask_l1 = (positions[:, 1] <= 0.75) & (positions[:, 1] > 0.25)
    mask_l2 = (positions[:, 1] <= 1.25) & (positions[:, 1] > 0.75)
    mask_l3 = (positions[:, 1] <= 1.75) & (positions[:, 1] > 1.25)
    mask_l4 = (positions[:, 1] <= 2.25) & (positions[:, 1] > 1.75)

    positions_hom = np.hstack([positions, np.ones((positions.shape[0], 1), dtype=np.float32)])
    map_l0 = pts2map((positions_hom @ mat)[mask_l0], class_ids[mask_l0], resolution=resolution)
    map_l1 = pts2map((positions_hom @ mat)[mask_l1], class_ids[mask_l1], resolution=resolution)
    map_l2 = pts2map((positions_hom @ mat)[mask_l2], class_ids[mask_l2], resolution=resolution)
    map_l3 = pts2map((positions_hom @ mat)[mask_l3], class_ids[mask_l3], resolution=resolution)
    map_l4 = pts2map((positions_hom @ mat)[mask_l4], class_ids[mask_l4], resolution=resolution)
    rr.log("pcd.0", rr.Points3D(positions=positions[mask_l0], colors=semrgb[mask_l0], radii=0.01))
    rr.log("pcd.1", rr.Points3D(positions=positions[mask_l1], colors=[0.0, 0.0, 1.0], radii=0.01))
    rr.log("pcd.2", rr.Points3D(positions=positions[mask_l2], colors=[0.3, 0.0, 1.0], radii=0.01))
    rr.log("pcd.3", rr.Points3D(positions=positions[mask_l3], colors=[0.6, 0.0, 1.0], radii=0.01))
    rr.log("pcd.4", rr.Points3D(positions=positions[mask_l4], colors=[0.9, 0.0, 1.0], radii=0.01))

    # show sign
    for node_name in scene.graph.nodes:
        if node_name.startswith("S"):
            transform, _ = scene.graph.get(node_name)
            position = transform[:3, 3]
            rr.log(f"sign.{node_name}", rr.Boxes3D(half_sizes=[0.5, 0.5, 0.5], centers=position, colors=[0.0, 1.0, 0.0]))

    # show road geometry
    for geometry_name, node_names in scene.graph.geometry_nodes.items():
        mesh = scene.geometry.get(geometry_name)
        for node_name in node_names:
            transform, _ = scene.graph.get(node_name)
            verts = trimesh.transformations.transform_points(mesh.vertices, transform).reshape(-1, 3)
            indices = mesh.faces.reshape(-1)

            verts_hom = np.hstack([verts, np.ones((verts.shape[0], 1), dtype=np.float32)])
            poly2d = np.round((verts_hom @ mat)[indices]).astype(np.int32).reshape(-1, 3, 1, 2)
            if node_name.startswith("road.same"):
                cv2.fillPoly(road_same_map, poly2d, color=1, lineType=cv2.LINE_AA)
                rr.log(f"geometry.{node_name}", rr.Mesh3D(vertex_positions=verts + np.array([0.0, -0.1, 0.0]), triangle_indices=indices, albedo_factor=[0.5, 0.5, 0.8, 0.5]))
            if node_name.startswith("road.opposite"):
                cv2.fillPoly(road_opposite_map, poly2d, color=1, lineType=cv2.LINE_AA)
                rr.log(f"geometry.{node_name}", rr.Mesh3D(vertex_positions=verts + np.array([0.0, -0.1, 0.0]), triangle_indices=indices, albedo_factor=[0.5, 0.8, 0.5, 0.5]))
            if node_name.startswith("crossing"):
                cv2.fillPoly(crossing_map, poly2d, color=1, lineType=cv2.LINE_AA)
                rr.log(f"geometry.{node_name}", rr.Mesh3D(vertex_positions=verts + np.array([0.0, -0.1, 0.0]), triangle_indices=indices, albedo_factor=[0.8, 0.5, 0.5, 0.5]))

    # show ego trajectory
    trajectory = np.frombuffer(data["ego_xyz"], dtype=np.float32).reshape(-1, 3)
    rr.log("ego_traj.0", rr.LineStrips3D(strips=trajectory, colors=[0.0, 1.0, 0.0], radii=0.02))

    # show tracking trajectory
    trajectory = np.frombuffer(data["alt_xyz"], dtype=np.float32).reshape(-1, 3)
    trajectory_typemap = np.frombuffer(data["alt_typemap"], dtype=np.uint8).reshape(-1)
    trajectory_typename = data["alt_typename"]
    trajectory_group = {id: trajectory[trajectory_typemap == id] for id in np.unique(trajectory_typemap)}
    for id, trajectory in trajectory_group.items():
        rr.log(f"tracking_traj.{trajectory_typename[id]}", rr.LineStrips3D(strips=trajectory + np.array([0.0, 0.1, 0.0]), colors=[0.0, 0.0, 1.0], radii=0.02))

    cv2.imwrite("map_l0.png", map_l0 * 10)
    cv2.imwrite("map_l1.png", map_l1 * 10)
    cv2.imwrite("map_l2.png", map_l2 * 10)
    cv2.imwrite("map_l3.png", map_l3 * 10)
    cv2.imwrite("map_l4.png", map_l4 * 10)
    cv2.imwrite("road_same_map.png", road_same_map * 255)
    cv2.imwrite("road_opposite_map.png", road_opposite_map * 255)
    cv2.imwrite("crossing_map.png", ((road_same_map | road_opposite_map) & crossing_map) * 255)
