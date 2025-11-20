import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
import pandas as pd


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

    if not nme_errors:
        return None, None

    # Average the errors and convert to percentage
    final_nme = np.mean(nme_errors) * 100
    final_stability = np.mean(stability_errors) * 100

    return final_nme, final_stability


def main(args):
    N = args.N

    results = {}

    # Ensure the output directory exists
    output_dir = "preds"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate from 1 up to and including N
    for i in tqdm(range(1, N + 1)):
        # Format the index i as a 3-digit string (e.g., 1 -> '001', 59 -> '059')
        video_id = f"{i:03d}"

        # Define paths based on your requested structure
        gt_dir = os.path.join("300VW_Dataset_2015_12_14", video_id, "annot")
        preds_base_dir = os.path.join("preds", video_id)

        baseline_file = os.path.join(preds_base_dir, "baseline.npy")
        v3_file = os.path.join(preds_base_dir, "v3.npy")

        video_results = {}
        gt_landmarks, gt_bboxes = None, None

        # 1. Load Ground Truth Data (once per video)
        try:
            gt_landmarks, gt_bboxes = load_ground_truth(gt_dir)
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error loading ground truth for {video_id} from {gt_dir}: {e}")
            continue

        # 2. Calculate Baseline Metrics
        try:
            pred_landmarks_baseline = np.load(baseline_file)
            nme_b, stab_b = calculate_paper_metrics(
                pred_landmarks_baseline, gt_landmarks.copy(), gt_bboxes.copy()
            )
            video_results["NME_baseline"] = nme_b
            video_results["Stability_baseline"] = stab_b
        except FileNotFoundError:
            video_results["NME_baseline"] = np.nan
            video_results["Stability_baseline"] = np.nan
        except Exception as e:
            print(f"Error calculating baseline metrics: {e}")
            video_results["NME_baseline"] = np.nan
            video_results["Stability_baseline"] = np.nan

        # 3. Calculate V3 Metrics
        try:
            pred_landmarks_v3 = np.load(v3_file)
            nme_v3, stab_v3 = calculate_paper_metrics(
                pred_landmarks_v3, gt_landmarks.copy(), gt_bboxes.copy()
            )
            video_results["NME_v3"] = nme_v3
            video_results["Stability_v3"] = stab_v3
        except FileNotFoundError:
            video_results["NME_v3"] = np.nan
            video_results["Stability_v3"] = np.nan
        except Exception as e:
            print(f"Error calculating v3 metrics: {e}")
            video_results["NME_v3"] = np.nan
            video_results["Stability_v3"] = np.nan

        # Store results for this video, keyed by the index 'i'
        results[i] = video_results

    # 4. Create and Save Pandas DataFrame
    if results:
        # Create DataFrame from the dictionary, using index 'i' as the row index
        df = pd.DataFrame.from_dict(results, orient="index")
        df.index.name = "Video_ID"

        output_file = os.path.join(output_dir, "metrics_summary.csv")
        df.to_csv(output_file)

        print(f"Evaluation complete. Results saved to: {output_file}")
    else:
        print("\nNo results to save. Check file paths and input argument N.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate 3DDFA-V2 NME and Stability metrics for a range of videos."
    )
    parser.add_argument(
        "N",
        type=int,
        help="The upper limit for the video index (i.e., N in i=1..N).",
    )
    args = parser.parse_args()
    main(args)
