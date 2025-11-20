# coding: utf-8

__author__ = "cleardusk"

import argparse
import imageio
import cv2
from tqdm import tqdm
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render

from utils.functions import cv_draw_landmark
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

    # Given a camera
    # before run this line, make sure you have installed `imageio-ffmpeg`
    reader = imageio.get_reader("<video0>")

    # NEW: Initialize the smoother
    smoother = TemporalSmoother(**smoother_hyperparams)

    # run
    dense_flag = args.opt in ("2d_dense", "3d")
    pre_ver = None
    for i, frame in tqdm(enumerate(reader)):
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
        smoothed_param = smoother.smooth(raw_param)
        ver_smooth = tddfa.recon_vers(
            [smoothed_param], roi_box_lst, dense_flag=dense_flag
        )[0]

        if args.opt == "2d_sparse":
            # since we use padding
            img_draw = cv_draw_landmark(frame_bgr, ver_smooth)
        elif args.opt == "2d_dense":
            img_draw = cv_draw_landmark(frame_bgr, ver_smooth, size=1)
        elif args.opt == "3d":
            img_draw = render(frame_bgr, [ver_smooth], tddfa.tri, alpha=0.7)
        else:
            raise ValueError(f"Unknown opt {args.opt}")

        cv2.imshow("image", img_draw)
        k = cv2.waitKey(20)
        if k & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The smooth demo of webcam of 3DDFA_V2"
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

    args = parser.parse_args()
    main(args)
