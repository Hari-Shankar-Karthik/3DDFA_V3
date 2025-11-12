import numpy as np
import argparse
import os
import glob


def parse_pts_file(filepath):
    """
    Parses a Menpo-3D .pts file.
    Returns a numpy array of shape (68, 2).
    """
    landmarks = []
    with open(filepath) as f:
        lines = f.readlines()
        # Skip header 'version: 1', 'n_points: 68', and '{' '}' lines
        for line in lines[3:-1]:
            x, y = line.strip().split()
            landmarks.append([float(x), float(y)])
    return np.array(landmarks)


def load_ground_truth(gt_dir):
    """
    Loads all ground-truth landmarks from a directory
    containing .pts files.
    """
    gt_landmarks = []
    gt_bboxes = []

    # Find all .pts files and sort them numerically by frame
    pts_files = sorted(glob.glob(os.path.join(gt_dir, "*.pts")))
    if not pts_files:
        raise FileNotFoundError(f"No .pts files found in directory: {gt_dir}")

    for filepath in pts_files:
        landmarks_68 = parse_pts_file(filepath)
        gt_landmarks.append(landmarks_68)

        # Calculate bounding box from landmarks
        # [x_min, y_min, x_max, y_max]
        x_min, y_min = landmarks_68.min(axis=0)
        x_max, y_max = landmarks_68.max(axis=0)
        gt_bboxes.append([x_min, y_min, x_max, y_max])

    return np.array(gt_landmarks), np.array(gt_bboxes)


def get_norm_factor(bbox):
    """
    Calculates the normalization factor (bounding box size).
    As per the paper, this is sqrt(width * height).
    Bbox is [x_min, y_min, x_max, y_max]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.sqrt(w * h)


def calculate_paper_metrics(pred_landmarks, gt_landmarks, gt_bboxes):
    """
    Calculates the NME (Accuracy) and Stability (Jitter) metrics
    as defined in the 3DDFA-V2 paper (Section 3.1).
    """

    # Align frame counts: predictions might be shorter/longer
    num_frames_pred = len(pred_landmarks)
    num_frames_gt = len(gt_landmarks)

    if num_frames_pred > num_frames_gt:
        print(
            f"Warning: Predictions ({num_frames_pred}) longer than "
            f"Ground Truth ({num_frames_gt}). Trimming predictions."
        )
        pred_landmarks = pred_landmarks[:num_frames_gt]
    elif num_frames_gt > num_frames_pred:
        print(
            f"Warning: Ground Truth ({num_frames_gt}) longer than "
            f"Predictions ({num_frames_pred}). Trimming Ground Truth."
        )
        gt_landmarks = gt_landmarks[:num_frames_pred]
        gt_bboxes = gt_bboxes[:num_frames_pred]

    num_frames = len(pred_landmarks)
    if num_frames < 2:
        print("Error: Need at least 2 frames to calculate stability.")
        return None, None

    nme_errors = []
    stability_errors = []

    for t in range(num_frames):
        pred_t = pred_landmarks[t]
        gt_t = gt_landmarks[t]
        bbox_t = gt_bboxes[t]

        norm_factor = get_norm_factor(bbox_t)
        if norm_factor == 0:
            continue

        # --- 1. Calculate NME (Accuracy) for frame t ---
        diff_t = pred_t - gt_t
        l2_distances_t = np.linalg.norm(diff_t, axis=1)
        nme_t = np.mean(l2_distances_t) / norm_factor
        nme_errors.append(nme_t)

        # --- 2. Calculate Stability (Jitter) for frame t (vs t-1) ---
        if t > 0:
            pred_t_minus_1 = pred_landmarks[t - 1]
            gt_t_minus_1 = gt_landmarks[t - 1]

            # Ground-truth offset (delta p)
            delta_p = gt_t - gt_t_minus_1
            # Predicted offset (delta q)
            delta_q = pred_t - pred_t_minus_1

            # Error of the offsets
            diff_of_deltas = delta_q - delta_p
            l2_distances_stability = np.linalg.norm(diff_of_deltas, axis=1)

            # Normalize by bounding box size at frame t
            stability_t = np.mean(l2_distances_stability) / norm_factor
            stability_errors.append(stability_t)

    # Average the errors and convert to percentage
    final_nme = np.mean(nme_errors) * 100
    final_stability = np.mean(stability_errors) * 100

    return final_nme, final_stability


def main(args):
    # 1. Load your predicted landmarks
    try:
        pred_landmarks = np.load(args.preds)
        print(
            f"Loaded predicted landmarks from {args.preds} (Shape: {pred_landmarks.shape})"
        )
    except Exception as e:
        print(f"Error: Could not load predictions file: {args.preds}")
        print(e)
        return

    # 2. Load the ground truth data
    try:
        gt_landmarks, gt_bboxes = load_ground_truth(args.gt_dir)
        print(f"Loaded {len(gt_landmarks)} ground-truth frames from {args.gt_dir}")
    except Exception as e:
        print(f"Error: Could not load ground-truth data from directory: {args.gt_dir}")
        print(e)
        return

    # 3. Calculate metrics
    nme, stability = calculate_paper_metrics(pred_landmarks, gt_landmarks, gt_bboxes)

    if nme is not None:
        print("\n--- Final Metrics (Lower is Better) ---")
        print(f"  NME (Accuracy):     {nme:.4f}%")
        print(f"  Stability (Jitter): {stability:.4f}%")
        print("-----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate 3DDFA-V2 NME and Stability metrics."
    )
    parser.add_argument(
        "--preds",
        type=str,
        required=True,
        help="Path to the .npy file containing predicted landmarks.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Path to the *directory* containing ground-truth .pts files for the video.",
    )

    args = parser.parse_args()
    main(args)
