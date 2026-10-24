import enum

import cv2
import numpy as np
import scipy.signal


class QualityTag(enum.IntFlag):
    NONE = 0
    QUALITY_A = 1 << 0
    QUALITY_B = 1 << 1
    QUALITY_C = 1 << 2


class GroupTag(enum.IntFlag):
    NONE = 0

    DAY = 1 << 0
    NIGHT = 1 << 1
    SUNNY = 1 << 2
    RAINNY = 1 << 3

    TLS = 1 << 4
    STOP_RULE = 1 << 5
    POLE_SIGN = 1 << 6
    ROAD_SIGN = 1 << 7
    COLORED_ROAD = 1 << 8

    SINGLE_AGENT = 1 << 9
    MULTI_AGENT = 1 << 10
    LEFT_TURN = 1 << 11
    RIGHT_TURN = 1 << 12

    WLINE_IN = 1 << 13
    WLINE_OUT = 1 << 14

    QUALITY_A = 1 << 15
    QUALITY_B = 1 << 16
    QUALITY_C = 1 << 17

    TSHAPE_MINOR_IN = 1 << 18
    TSHAPE_MAJOR_IN = 1 << 19


class LaneKind(enum.IntEnum):
    IN = 0
    OUT = 1
    IN_OPPOSITE = 2
    OUT_OPPOSITE = 3


class LineKind(enum.IntEnum):
    STOP_LINE = 0
    CROSSING = 1


# verts: image space coordinates
def rasterize_pcd(verts: np.ndarray, class_ids: np.ndarray, output_res: int, rasterize_res: int) -> np.ndarray:
    verts = np.array(verts, dtype=np.float64)
    class_ids = np.array(class_ids, dtype=np.int64)

    verts = verts / output_res * rasterize_res
    verts = np.round(verts).astype(np.int32)
    idx = (verts[:, 0] >= 0) & (verts[:, 0] < rasterize_res) & (verts[:, 1] >= 0) & (verts[:, 1] < rasterize_res)
    x, y, class_ids = verts[idx, 0], verts[idx, 1], class_ids[idx]

    max_cls = class_ids.max() + 1
    pixel_idx = y * rasterize_res + x
    combined_keys = pixel_idx * max_cls + class_ids
    unique_keys, counts = np.unique(combined_keys, return_counts=True)
    unq_pixel_idx = unique_keys // max_cls
    unq_class_ids = unique_keys % max_cls

    image = np.full(rasterize_res * rasterize_res, 255, dtype=np.uint8)
    max_counts = np.zeros(rasterize_res * rasterize_res, dtype=np.int32)
    for p_idx, c_id, count in zip(unq_pixel_idx, unq_class_ids, counts, strict=True):
        if count > max_counts[p_idx]:
            max_counts[p_idx] = count
            image[p_idx] = c_id
    image = image.reshape((rasterize_res, rasterize_res))

    image = cv2.resize(image, (output_res, output_res), interpolation=cv2.INTER_NEAREST)
    return np.array(image, dtype=np.uint8)


def compute_tangent_xyz(path: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    path = np.array(path, dtype=np.float64)

    if len(path) < window_length:
        raise ValueError(f"Path length ({len(path)}) must be greater than window_length ({window_length}).")

    # compute derivatives using Savitzky-Golay filter
    dx = scipy.signal.savgol_filter(path[:, 0], window_length=window_length, polyorder=polyorder, deriv=1, axis=0)
    dy = scipy.signal.savgol_filter(path[:, 1], window_length=window_length, polyorder=polyorder, deriv=1, axis=0)
    dz = scipy.signal.savgol_filter(path[:, 2], window_length=window_length, polyorder=polyorder, deriv=1, axis=0)
    assert isinstance(dx, np.ndarray) and isinstance(dy, np.ndarray) and isinstance(dz, np.ndarray)

    # combine derivatives into a single array
    tangents = np.column_stack([dx, dy, dz])

    # normalize the tangent vectors to unit length
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
    return np.array(tangents, dtype=np.float32)


def find_nearest_idx(path: np.ndarray, p: np.ndarray) -> int:
    path = np.array(path, dtype=np.float64)
    p = np.array(p, dtype=np.float64)

    dists = np.linalg.norm(path - p, axis=1)
    return int(np.argmin(dists))


def extend_path_xyz(path: np.ndarray, len0: float, len1: float, interval: float) -> np.ndarray:
    path = np.array(path, dtype=np.float64)

    tangent_xyz = compute_tangent_xyz(path, window_length=9, polyorder=2)
    p0, t0 = path[0], tangent_xyz[0]
    p1, t1 = path[-1], tangent_xyz[-1]
    d0_steps = np.arange(interval, len0, interval, dtype=np.float64)
    d1_steps = np.arange(interval, len1, interval, dtype=np.float64)
    tip0_path = np.array([p0 - d * t0 for d in reversed(d0_steps)], dtype=np.float64).reshape(-1, 3)
    tip1_path = np.array([p1 + d * t1 for d in d1_steps], dtype=np.float64).reshape(-1, 3)

    final_path = np.concat([tip0_path, path, tip1_path], axis=0)
    return np.array(final_path, dtype=np.float32)


def compute_intersection_xz(path: np.ndarray, p0: np.ndarray, t0: np.ndarray) -> np.ndarray:
    path = np.array(path, dtype=np.float64)
    p0 = np.array(p0, dtype=np.float64)
    t0 = np.array(t0, dtype=np.float64)

    pred_path = path[:-1]
    succ_path = path[1:]
    dpath = succ_path - pred_path

    n_segments = len(pred_path)

    # 2. 直線とセグメントの交点を求めるための連立方程式を準備
    # 直線: p0 + t * t0 (-inf < t < inf)
    # セグメント: pred_path + s * dpath  (0 <= s <= 1)
    # 移項して行列形式にする: [t0, -dpath] * [t, s]^T = pred_path - p0

    # A行列の構築 (N - 1, 2, 2)
    a_mat = np.empty((n_segments, 2, 2), dtype=np.float64)
    a_mat[:, :, 0] = t0
    a_mat[:, :, 1] = -dpath
    b = pred_path - p0

    # 3. 各セグメントに対して連立方程式を解く
    for i in range(n_segments):
        # 行列式で平行判定 (0に近い場合は交差しない)
        det = np.linalg.det(a_mat[i])
        if np.abs(det) < 1e-8:
            continue

        try:
            # np.linalg.solve で t と s を一括計算
            _, s = np.linalg.solve(a_mat[i], b[i])

            # 条件判定:
            # - s がセグメント上にあるか (0 <= s <= 1)
            if 0.0 <= s <= 1.0:
                # 交点座標の計算
                xyz = pred_path[i] + s * dpath[i]
                return np.array(xyz, dtype=np.float32)

        except np.linalg.LinAlgError:
            continue

    raise ValueError("No intersection found between the line and the path segments.")


def compute_intersection_xyz(path: np.ndarray, p0: np.ndarray, t0: np.ndarray) -> np.ndarray:
    xz = compute_intersection_xz(path[:, [0, 2]], p0[[0, 2]], t0[[0, 2]])
    return np.array([xz[0], p0[1], xz[1]], dtype=np.float32)


def compute_dual_intersection_xyz(path0: np.ndarray, path1: np.ndarray, p_: np.ndarray, t_: np.ndarray) -> np.ndarray:
    xyz0 = compute_intersection_xyz(path0, p_, t_)
    xyz1 = compute_intersection_xyz(path1, p_, t_)
    return (xyz0 + xyz1) * 0.5


def connect_path_xy(p0: np.ndarray, p1: np.ndarray, t0: np.ndarray, interval: float) -> np.ndarray:
    p0 = np.array(p0, dtype=np.float64)
    p1 = np.array(p1, dtype=np.float64)
    t0 = np.array(t0, dtype=np.float64)

    # 1. 始点接線の単位ベクトルと法線ベクトルを計算
    t_norm = np.linalg.norm(t0)
    if t_norm == 0:
        raise ValueError("The tangent vector t0 must not be zero.")
    t_unit = t0 / t_norm

    # 法線ベクトル (反時計回りに90度回転)
    n_unit = np.array([-t_unit[1], t_unit[0]], dtype=np.float64)

    # 2. 始点から終点への変位ベクトル
    d = p1 - p0

    if np.linalg.norm(d) < 1e-4:
        return np.array([p0])

    # 終点が始点の接線方向と完全に平行 (直線上のケース) か判定
    cross_check = np.dot(d, n_unit)

    if np.abs(cross_check) < 1e-4:
        # 曲率 0 (直線のケース)
        total_length = np.linalg.norm(d)
        num_points = max(2, int(np.ceil(total_length / interval)) + 1)
        t_vals = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
        curve = np.outer(1 - t_vals, p0) + np.outer(t_vals, p1)
        return np.array(curve, dtype=np.float32)

    # 3. 円弧の中心と半径を解析的に計算
    # 半径 R = |d|^2 / (2 * (d * n_unit))
    radius_signed = np.dot(d, d) / (2.0 * cross_check)
    radius = np.abs(radius_signed)
    center = p0 + radius_signed * n_unit

    # 4. 始点と終点の中心からの相対ベクトルと角度を計算
    v0 = p0 - center
    v1 = p1 - center

    alpha0 = np.arctan2(v0[1], v0[0])

    # 符号付きの回転角 (デルタ角) を正確に算出
    cos_theta = np.dot(v0, v1) / (radius**2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin_theta = (v0[0] * v1[1] - v0[1] * v1[0]) / (radius ** 2)
    delta_theta = np.arctan2(sin_theta, cos_theta)

    # 5. 円弧の全長と、指定間隔に基づいたサンプル点の生成
    arc_length = radius * np.abs(delta_theta)
    num_points = max(2, int(np.ceil(arc_length / interval)) + 1)
    s_vals = np.linspace(0, arc_length, num_points, dtype=np.float64)

    # 弧長に比例した角度の配列を生成
    angles = alpha0 + (s_vals / arc_length) * delta_theta

    # 6. 座標の展開
    curve_x = center[0] + radius * np.cos(angles)
    curve_y = center[1] + radius * np.sin(angles)

    curve = np.column_stack([curve_x, curve_y])
    return np.array(curve, dtype=np.float32)


def connect_path_xyz(p0: np.ndarray, p1: np.ndarray, t0: np.ndarray, interval: float) -> np.ndarray:
    xz = connect_path_xy(p0[[0, 2]], p1[[0, 2]], t0[[0, 2]], interval)
    y = np.linspace(p0[1], p1[1], xz.shape[0], dtype=np.float32)
    xyz = np.column_stack((xz[:, 0], y, xz[:, 1]))
    return np.array(xyz, dtype=np.float32)


def compute_center_offset_xz(p0: np.ndarray, t0: np.ndarray, path: np.ndarray) -> float:
    p0 = np.array(p0, dtype=np.float64)
    t0 = np.array(t0, dtype=np.float64)
    path = np.array(path, dtype=np.float64)

    # d_plane > 0 indicates the direction of the normal vector, < 0 indicates the opposite direction.
    d_plane = np.dot(path - p0, t0)

    # segments that cross the line
    sign_change = (d_plane[:-1] * d_plane[1:]) <= 0
    cross_idx = np.where(sign_change)[0]
    if len(cross_idx) == 0:
        return float("nan")

    sdists = []
    for idx in cross_idx:
        p1 = path[idx]
        p2 = path[idx + 1]
        d1 = d_plane[idx]
        d2 = d_plane[idx + 1]

        # the intersection point between the trajectory and the line
        denom = d1 - d2
        t = 0.0 if np.abs(denom) < 1e-8 else d1 / denom
        intersection_point = (1 - t) * p1 + t * p2

        # Project the intersection point onto the direction of the line
        # (right side relative to the normal vector is positive, left side is negative)
        line_dir = np.array([-t0[1], t0[0]], dtype=np.float64)
        sdist = np.dot(intersection_point - p0, line_dir)
        sdists.append(sdist)

    offset = min(sdists, key=np.abs)
    return float(offset)


def compute_center_offset_xyz(p0: np.ndarray, t0: np.ndarray, path: np.ndarray) -> float:
    return compute_center_offset_xz(p0[[0, 2]], t0[[0, 2]], path[:, [0, 2]])


def compute_path_width_xz(p0: np.ndarray, p1: np.ndarray, t_: np.ndarray) -> float:
    dir = np.array([-t_[1], t_[0]], dtype=np.float64)
    width = np.abs(np.dot(p1 - p0, dir))
    return float(width)


def compute_path_width_xyz(p0: np.ndarray, p1: np.ndarray, t0: np.ndarray) -> float:
    return compute_path_width_xz(p0[[0, 2]], p1[[0, 2]], t0[[0, 2]])
