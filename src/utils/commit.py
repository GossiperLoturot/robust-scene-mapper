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


def compute_tangent_xyz(trajectory: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    if len(trajectory) < window_length:
        raise ValueError(f"Trajectory length ({len(trajectory)}) must be greater than window_length ({window_length}).")

    # compute derivatives using Savitzky-Golay filter
    dx = scipy.signal.savgol_filter(trajectory[:, 0], window_length=window_length, polyorder=polyorder, deriv=1, axis=0)
    dy = scipy.signal.savgol_filter(trajectory[:, 1], window_length=window_length, polyorder=polyorder, deriv=1, axis=0)
    dz = scipy.signal.savgol_filter(trajectory[:, 2], window_length=window_length, polyorder=polyorder, deriv=1, axis=0)
    assert isinstance(dx, np.ndarray) and isinstance(dy, np.ndarray) and isinstance(dz, np.ndarray)

    # combine derivatives into a single array
    tangents = np.column_stack([dx, dy, dz])

    # normalize the tangent vectors to unit length
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return tangents / norms


def connect_lane_xy(p0: np.ndarray, p1: np.ndarray, t0: np.ndarray, interval: float) -> np.ndarray:
    # compute normal vector
    n0 = np.array([-t0[1], t0[0]], dtype=np.float32)

    dp = p1 - p0
    denom = np.dot(dp, n0)
    dist = np.linalg.norm(dp)

    # 1. in case of a straight line or very short distance, return linear interpolation
    if np.abs(denom) < 1e-8 or dist < 1e-8:
        if dist < 1e-8:
            return p0.reshape(1, 2)
        num_points = max(2, int(np.ceil(dist / interval)) + 1)
        t = np.linspace(0, 1, num_points)[:, np.newaxis]
        return (1 - t) * p0 + t * p1

    # 2. in casr of a circular arc, compute the center and radius
    s = np.dot(dp, dp) / (2.0 * denom)
    c = p0 + s * n0
    rot = np.abs(s)

    # compute angles for p0 and p1 relative to the center
    v0 = p0 - c
    v1 = p1 - c
    theta0 = np.arctan2(v0[1], v0[0])
    theta1 = np.arctan2(v1[1], v1[0])

    # compute the angle difference and ensure it is in the range [-pi, pi]
    dtheta = theta1 - theta0
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    tangent_check = np.array([-v0[1], v0[0]], dtype=np.float32) if s > 0 else np.array([v0[1], -v0[0]], dtype=np.float32)
    if np.dot(tangent_check, t0) < 0:
        if dtheta > 0:
            dtheta -= 2 * np.pi
        else:
            dtheta += 2 * np.pi

    # compute the number of points based on the arc length and interval
    arc_length = rot * np.abs(dtheta)
    num_points = max(2, int(np.ceil(arc_length / interval)) + 1)

    # generate points along the arc
    thetas = np.linspace(theta0, theta0 + dtheta, num_points)
    return np.column_stack([c[0] + rot * np.cos(thetas), c[1] + rot * np.sin(thetas)])


def connect_lane_xyz(p0: np.ndarray, p1: np.ndarray, t0: np.ndarray, interval: float) -> np.ndarray:
    xz = connect_lane_xy(p0[[0, 2]], p1[[0, 2]], t0[[0, 2]], interval)
    y = np.linspace(p0[1], p1[1], xz.shape[0])
    return np.array([xz[:, 0], y, xz[:, 1]], dtype=np.float32).T


def extend_lane_xyz(xyz: np.ndarray, extend_length: float, interval: float) -> np.ndarray:
    tangent_xyz = compute_tangent_xyz(xyz, window_length=9, polyorder=2)
    p0, t0 = xyz[0], tangent_xyz[0]
    p1, t1 = xyz[-1], tangent_xyz[-1]
    d_steps = np.arange(interval, extend_length, interval)
    edge0_xyz = np.array([p0 - d * t0 for d in reversed(d_steps)], dtype=np.float32)
    edge1_xyz = np.array([p1 + d * t1 for d in d_steps], dtype=np.float32)
    return np.concat([edge0_xyz, xyz, edge1_xyz], axis=0)


def compute_center_offset_xyz(c: np.ndarray, n: np.ndarray, traj: np.ndarray) -> float:
    return compute_center_offset_xz(c[[0, 2]], n[[0, 2]], traj[:, [0, 2]])


def compute_center_offset_xz(c: np.ndarray, n: np.ndarray, traj: np.ndarray) -> float:
    # d_plane > 0 indicates the direction of the normal vector, < 0 indicates the opposite direction.
    d_plane = np.dot(traj - c, n)

    # segments that cross the line
    sign_change = (d_plane[:-1] * d_plane[1:]) <= 0
    cross_idx = np.where(sign_change)[0]
    if len(cross_idx) == 0:
        return float('nan')

    sdists = []
    for idx in cross_idx:
        p1 = traj[idx]
        p2 = traj[idx + 1]
        d1 = d_plane[idx]
        d2 = d_plane[idx + 1]

        # the intersection point between the trajectory and the line
        denom = d1 - d2
        t = 0.0 if np.abs(denom) < 1e-12 else d1 / denom
        intersection_point = (1 - t) * p1 + t * p2

        # Project the intersection point onto the direction of the line
        # (right side relative to the normal vector is positive, left side is negative)
        line_dir = np.array([-n[1], n[0]], dtype=np.float32)
        sdist = np.dot(intersection_point - c, line_dir)
        sdists.append(sdist)

    return float(min(sdists, key=np.abs))


def compute_lane_width_xyz(p0: np.ndarray, p1: np.ndarray, tc: np.ndarray) -> float:
    return compute_lane_width_xz(p0[[0, 2]], p1[[0, 2]], tc[[0, 2]])


def compute_lane_width_xz(p0: np.ndarray, p1: np.ndarray, tc: np.ndarray) -> float:
    line_dir = np.array([-tc[1], tc[0]], dtype=np.float32)
    return float(np.abs(np.dot(p1 - p0, line_dir)))


def find_nearest_idx(traj: np.ndarray, p: np.ndarray) -> int:
    dists = np.linalg.norm(traj - p, axis=1)
    return int(np.argmin(dists))
