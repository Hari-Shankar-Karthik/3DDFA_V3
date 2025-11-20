# coding: utf-8

__author__ = "cleardusk"

import argparse
import imageio
import numpy as np
from tqdm import tqdm
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA

from temporal_smoother import TemporalSmoother
from hyperparams import smoother_hyperparams


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os

        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        os.environ["OMP_NUM_THREADS"] = "4"

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == "gpu"
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a video file
    # before run this line, make sure you have installed `imageio-ffmpeg`
    reader = imageio.get_reader(args.video_input)

    # NEW: Initialize the smoother
    smoother = TemporalSmoother(**smoother_hyperparams)

    # Initialize landmarks list
    landmarks_list = []

    # run
    dense_flag = args.opt in ("2d_dense", "3d")
    pre_ver = None
    for i, frame in tqdm(enumerate(reader), total=reader.count_frames()):
        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i == 0:
            # the first frame, detect face, here we only use the first face, you can change depending on your need
            boxes = face_boxes(frame_bgr)
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy="landmark")
        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy="landmark")

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

        # NEW: Smooth before reconstructing vertices
        raw_param = param_lst[0]
        ver_raw = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        pre_ver = ver_raw
        if args.no_smooth:
            smoothed_param = raw_param
        else:
            smoothed_param = smoother.smooth(raw_param)
        ver_smooth = tddfa.recon_vers(
            [smoothed_param], roi_box_lst, dense_flag=dense_flag
        )[0]

        # Landmark extraction
        # ver_smooth is (3, N), we take x, y rows ([:2, :]) -> (2, N)
        # then transpose (.T) -> (N, 2)
        sparse_landmarks_2d = ver_smooth[:2, :].T
        landmarks_list.append(sparse_landmarks_2d)

    # --- Save landmarks and release reader
    reader.close()
    landmarks_array = np.array(landmarks_list)
    np.save(args.o_landmarks, landmarks_array)

    print(f"\nFinished processing. Landmarks saved to: {args.o_landmarks}")
    print(f"Output shape: {landmarks_array.shape} (Frames, N_Landmarks, 2)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The smooth demo of 3DDFA_V2 for video files, saving landmarks"
    )

    parser.add_argument(
        "-i",
        "--video_input",
        type=str,
        required=True,
        help="Path to the input video file (.avi)",
    )
    parser.add_argument(
        "-o_lmk",
        "--o_landmarks",
        type=str,
        required=True,
        help="Path to save the output landmarks .npy file",
    )

    parser.add_argument("-c", "--config", type=str, default="configs/mb1_120x120.yml")
    parser.add_argument("-m", "--mode", default="gpu", type=str, help="gpu or cpu mode")
    parser.add_argument(
        "-o",
        "--opt",
        type=str,
        default="2d_sparse",
        choices=["2d_sparse", "2d_dense", "3d"],
    )
    parser.add_argument("--onnx", action="store_true", default=True)
    parser.add_argument("-n", "--no_smooth", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
