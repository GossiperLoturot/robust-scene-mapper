import dataclasses

import numpy as np
import pycolmap
import scipy


@dataclasses.dataclass
class MetricDepthScale:
    scale_coef: float
    scale_coef_iqr: float


def calculate_scale(metric_depth, model: pycolmap.Reconstruction) -> MetricDepthScale:
    metric_extrinsics = metric_depth["extrinsics"]

    relative_extrinsics = np.zeros_like(metric_extrinsics)
    for image_id in range(model.num_images()):
        image = model.image(image_id)
        relative_extrinsics[image_id] = np.concat([image.cam_from_world().matrix(), [[0, 0, 0, 1]]])

    metric_translate = np.linalg.inv(metric_extrinsics) @ np.array([0, 0, 0, 1])
    relative_translate = np.linalg.inv(relative_extrinsics) @ np.array([0, 0, 0, 1])

    metric_delta = np.linalg.norm(metric_translate[1:] - metric_translate[:-1], axis=1)
    relative_delta = np.linalg.norm(relative_translate[1:] - relative_translate[:-1], axis=1)

    scale_coef = float(np.median(metric_delta / relative_delta))
    scale_coef_iqr = float(scipy.stats.iqr(metric_delta / relative_delta))
    return MetricDepthScale(scale_coef=scale_coef, scale_coef_iqr=scale_coef_iqr)
