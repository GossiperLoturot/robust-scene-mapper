import cv2
import numpy as np
import trimesh

import context


# verts: image space coordinates
def rasterize_pcd(verts: np.ndarray, class_ids: np.ndarray, res: int, downsample_res: int) -> np.ndarray:
    verts = verts / res * downsample_res
    verts = np.round(verts).astype(np.int32)
    x = np.clip(verts[:, 0], 0, downsample_res - 1)
    y = np.clip(verts[:, 1], 0, downsample_res - 1)

    max_cls = class_ids.max() + 1
    pixel_idx = y * downsample_res + x
    combined_keys = pixel_idx * max_cls + class_ids
    unique_keys, counts = np.unique(combined_keys, return_counts=True)
    unq_pixel_idx = unique_keys // max_cls
    unq_class_ids = unique_keys % max_cls

    image = np.zeros(downsample_res * downsample_res, dtype=np.uint8)
    max_counts = np.zeros(downsample_res * downsample_res, dtype=np.int32)
    for p_idx, c_id, count in zip(unq_pixel_idx, unq_class_ids, counts, strict=True):
        if count > max_counts[p_idx]:
            max_counts[p_idx] = count
            image[p_idx] = c_id
    image = image.reshape((downsample_res, downsample_res))

    image = cv2.resize(image, (res, res), interpolation=cv2.INTER_AREA)
    return np.array(image, dtype=np.uint8)


# verts: image space coordinates
def rasterize_mesh(verts: np.ndarray, res: int) -> np.ndarray:
    poly2d = verts.reshape(-1, 3, 1, 2)
    poly2d = np.round(poly2d).astype(np.int32)

    image = np.zeros((res, res), dtype=np.uint8)
    cv2.fillPoly(image, list(poly2d), 255, lineType=cv2.LINE_AA)
    return image


# extnt: the bounding box size of the scene in meters
# res: the resolution of the output image
# downsample_res: the resolution for point cloud rasterization
def packing(data: dict, geometry_commit: trimesh.Scene, extent: float = 10.0, res: int = 512, downsample_res: int = 128) -> dict:
    ctx = context.Context()

    # matrix for 2D projection
    mat = np.array([[0.5 * res / extent, 0.0], [0.0, 0.0], [0.0, 0.5 * res / extent], [0.5 * res, 0.5 * res]], dtype=np.float32)
    ctx.logger.info(f"create projection matrix: {mat}")

    # read point cloud data
    verts = np.frombuffer(data["pcd_xyz"], dtype=np.float32).reshape(-1, 3)
    class_ids = np.frombuffer(data["pcd_typemap"], dtype=np.uint8).reshape(-1)
    # extract points in different height levels
    h0_mask = verts[:, 1] <= 0.25
    h1_mask = (verts[:, 1] <= 0.75) & (verts[:, 1] > 0.25)
    h2_mask = (verts[:, 1] <= 1.25) & (verts[:, 1] > 0.75)
    h3_mask = (verts[:, 1] <= 1.75) & (verts[:, 1] > 1.25)
    h4_mask = (verts[:, 1] <= 2.25) & (verts[:, 1] > 1.75)
    # rasterization
    verts_hom = np.hstack([verts, np.ones((verts.shape[0], 1), dtype=np.float32)])
    h0_map = rasterize_pcd((verts_hom @ mat)[h0_mask], class_ids[h0_mask], res=res, downsample_res=downsample_res)
    h1_map = rasterize_pcd((verts_hom @ mat)[h1_mask], class_ids[h1_mask], res=res, downsample_res=downsample_res)
    h2_map = rasterize_pcd((verts_hom @ mat)[h2_mask], class_ids[h2_mask], res=res, downsample_res=downsample_res)
    h3_map = rasterize_pcd((verts_hom @ mat)[h3_mask], class_ids[h3_mask], res=res, downsample_res=downsample_res)
    h4_map = rasterize_pcd((verts_hom @ mat)[h4_mask], class_ids[h4_mask], res=res, downsample_res=downsample_res)

    # read road geometry
    road_same_verts, road_opposite_verts, crossing_verts = [], [], []
    for geometry_name, node_names in geometry_commit.graph.geometry_nodes.items():
        mesh = geometry_commit.geometry.get(geometry_name)
        assert isinstance(mesh, trimesh.Trimesh)
        for node_name in node_names:
            transform, _ = geometry_commit.graph.get(node_name)
            verts = trimesh.transformations.transform_points(mesh.vertices, transform).reshape(-1, 3)
            indices = mesh.faces.reshape(-1)
            assert isinstance(verts, np.ndarray) and isinstance(indices, np.ndarray)
            # group by
            if node_name.startswith("road.same"):
                road_same_verts.append(verts[indices])
            if node_name.startswith("road.opposite"):
                road_opposite_verts.append(verts[indices])
            if node_name.startswith("crossing"):
                crossing_verts.append(verts[indices])
    # road.same geometry
    verts = np.array(road_same_verts, dtype=np.float32).reshape(-1, 3)
    verts_hom = np.hstack([verts, np.ones((verts.shape[0], 1), dtype=np.float32)])
    road_same_map = rasterize_mesh(verts_hom @ mat, res=res)
    # road.opposite geometry
    verts = np.array(road_opposite_verts, dtype=np.float32).reshape(-1, 3)
    verts_hom = np.hstack([verts, np.ones((verts.shape[0], 1), dtype=np.float32)])
    road_opposite_map = rasterize_mesh(verts_hom @ mat, res=res)
    # crossing geometry
    verts = np.array(crossing_verts, dtype=np.float32).reshape(-1, 3)
    verts_hom = np.hstack([verts, np.ones((verts.shape[0], 1), dtype=np.float32)])
    crossing_map = rasterize_mesh(verts_hom @ mat, res=res)
    # compose lane map
    lane_map = np.zeros((res, res), dtype=np.uint8)
    lane_map[road_same_map > 127] = 1
    lane_map[road_opposite_map > 127] = 2
    lane_map[((road_same_map > 127) | (road_opposite_map > 127)) & (crossing_map > 127)] = 3

    # read sign
    sign_group = dict[str, list[float]]()
    for node_name in geometry_commit.graph.nodes:
        if node_name.startswith("S"):
            transform, _ = geometry_commit.graph.get(node_name)
            position = transform[:3, 3]
            sign_group[node_name] = position.tolist()

    # create trajectory group
    trajectory_group = dict[str, bytes]()
    # read ego trajectory
    ego_trajectory = np.frombuffer(data["ego_xyz"], dtype=np.float32).reshape(-1, 3)
    trajectory_group["ego"] = ego_trajectory.tobytes()
    # read other trajectory
    other_trajectory = np.frombuffer(data["alt_xyz"], dtype=np.float32).reshape(-1, 3)
    other_trajectory_typemap = np.frombuffer(data["alt_typemap"], dtype=np.uint8).reshape(-1)
    other_trajectory_typename = data["alt_typename"]
    for id in np.unique(other_trajectory_typemap):
        name = other_trajectory_typename[id]
        trajectory = other_trajectory[other_trajectory_typemap == id]
        trajectory_group[name] = trajectory.tobytes()

    # encode
    _, h0_map_enc = cv2.imencode(".png", h0_map)
    _, h1_map_enc = cv2.imencode(".png", h1_map)
    _, h2_map_enc = cv2.imencode(".png", h2_map)
    _, h3_map_enc = cv2.imencode(".png", h3_map)
    _, h4_map_enc = cv2.imencode(".png", h4_map)
    _, lane_map_enc = cv2.imencode(".png", lane_map)
    return {
        "param": data["param"],
        "resolution": res,
        "h0_map": h0_map_enc.tobytes(),
        "h1_map": h1_map_enc.tobytes(),
        "h2_map": h2_map_enc.tobytes(),
        "h3_map": h3_map_enc.tobytes(),
        "h4_map": h4_map_enc.tobytes(),
        "lane_map": lane_map_enc.tobytes(),
        "sign_group": sign_group,
        "trajectory_group": trajectory_group,
        "pcd_xyz": data["pcd_xyz"],
        "pcd_rgb": data["pcd_rgb"],
        "pcd_typemap": data["pcd_typemap"],
        "pcd_typename": data["pcd_typename"],
    }
