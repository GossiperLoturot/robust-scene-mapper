import os
import shutil

import numpy as np
import pycolmap

import context
import utils.database
import utils.feature_matching


def upload_database(
    db_path: str,
    pairs_path: str,
    frame_width: int,
    frame_height: int,
    init_focal_length: float,
    matching_result: utils.feature_matching.FeatureMatchingResult,
    camera_mapping: utils.feature_matching.CameraMapping,
):
    # create COLMAP database
    db_file = utils.database.COLMAPDatabase.connect(db_path)
    # create pairs.txt
    pairs_file = open(pairs_path, "w")

    with db_file, pairs_file:
        # initialize
        db_file.create_tables()

        # Add camera
        for i in range(camera_mapping.num_cameras):
            db_file.add_camera(
                model=pycolmap.CameraModelId.OPENCV_FISHEYE.value,
                width=frame_width,
                height=frame_height,
                params=np.array([init_focal_length, init_focal_length, frame_width / 2, frame_height / 2, 0, 0, 0, 0]),
                prior_focal_length=True,
                camera_id=i,
            )

        # for each single
        for i in range(len(matching_result.image_paths)):
            # Add image
            name = os.path.basename(matching_result.image_paths[i])
            db_file.add_image(
                name=name,
                camera_id=camera_mapping.image_to_camera[i],
                prior_q=np.full(4, np.nan),
                prior_t=np.full(3, np.nan),
                image_id=i,
            )

            # Add keypoints
            db_file.add_keypoints(image_id=i, keypoints=matching_result.keypoints_dict[i])

        # for each pairs
        for i, j in matching_result.matches_dict:
            # Add matches
            matches = matching_result.matches_dict[(i, j)]
            db_file.add_matches(image_id1=i, image_id2=j, matches=matches)

            # Add pairs for geometry verification
            name_i = os.path.basename(matching_result.image_paths[i])
            name_j = os.path.basename(matching_result.image_paths[j])
            pairs_file.write("{} {}\n".format(name_i, name_j))

        # finalize
        db_file.commit()

    # geometry verification (RANSAC / Epipolar geometry constraint)
    pycolmap.verify_matches(db_path, pairs_path)


def incremental_reconstruction(
    db_path: str,
    image_dir: str,
    input_model_dir: str,
    output_model_dir: str,
):
    ctx = context.Context()

    # reconstruct
    opt = pycolmap.IncrementalPipelineOptions()
    opt.ignore_watermarks = True
    reconstructions = pycolmap.incremental_mapping(db_path, image_dir, input_model_dir, opt)

    # when no reconstruction
    if len(reconstructions) == 0:
        shutil.rmtree(input_model_dir)
        raise ValueError("No reconstruction found.")

    # select one fragment
    max_num_images = 0
    largest_model_id = None
    for id, model in reconstructions.items():
        if model.num_images() > max_num_images:
            max_num_images = model.num_images()
            largest_model_id = id
    assert largest_model_id is not None

    if max_num_images < 20:
        raise Exception("reconstruction succeeded, but register image is too small!")

    # write and clean up
    reconstructions[largest_model_id].write(output_model_dir)
    shutil.rmtree(input_model_dir)
    ctx.logger.info(reconstructions[largest_model_id].summary())
