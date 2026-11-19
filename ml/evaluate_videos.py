"""
evaluate_videos.py
Automated evaluation harness for FormCoach on 30+ labelled workout clips.
Computes:
  - Rep-count Mean Absolute Error (MAE) overall & per exercise
  - Form-error Precision, Recall, and F1 per class (Rule-Based Baseline vs ONNX Classifier)
  - Confusion matrices & failure gallery (baggy clothing, side view, low light)
  - Optional feature extraction CSV (--extract-features) for fine-tuning train_classifier.py

Usage:
  pip install -r ml/requirements.txt
  python ml/evaluate_videos.py --labels data/labels.csv --clips-dir data/clips --out-dir results/
"""

import argparse
import csv
import json
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None

CLASSES = ["good_form", "depth_issue", "alignment_issue"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

THRESHOLDS = {
    "squat": {"primary": "knee", "up": 158, "descend": 142, "bottom": 108, "shallow": 112},
    "pushup": {"primary": "elbow", "up": 155, "descend": 140, "bottom": 105, "shallow": 108},
    "lunge": {"primary": "knee", "up": 156, "descend": 140, "bottom": 108, "shallow": 112},
}


def angle_between(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 180.0
    cos_t = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_t)))


def angle_from_vertical(hip, shoulder):
    v = np.array([shoulder.x - hip.x, shoulder.y - hip.y])
    mag = np.linalg.norm(v)
    if mag < 1e-6:
        return 0.0
    cos_t = np.clip(-v[1] / mag, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_t)))


def extract_features_from_landmarks(lm, exercise):
    left_ids = [11, 13, 15, 23, 25, 27]
    right_ids = [12, 14, 16, 24, 26, 28]
    l_vis = np.mean([lm[i].visibility for i in left_ids])
    r_vis = np.mean([lm[i].visibility for i in right_ids])
    is_left = l_vis >= r_vis

    s = lm[11 if is_left else 12]
    e = lm[13 if is_left else 14]
    w = lm[15 if is_left else 16]
    h = lm[23 if is_left else 24]
    k = lm[25 if is_left else 26]
    a = lm[27 if is_left else 28]
    f = lm[31 if is_left else 32]

    oh = lm[24 if is_left else 23]
    ok = lm[26 if is_left else 25]
    oa = lm[28 if is_left else 27]

    knee = angle_between(h, k, a)
    opp_knee = angle_between(oh, ok, oa)
    hip = angle_between(s, h, k)
    elbow = angle_between(s, e, w)
    torso = angle_from_vertical(h, s)
    body_line = angle_between(s, h, a)

    facing = 1.0 if (f.x - a.x) >= 0 else -1.0
    shin = max(math.hypot(k.x - a.x, k.y - a.y), 0.05)
    kot = ((k.x - f.x) * facing) / shin

    active_knee = min(knee, opp_knee) if exercise == "lunge" else knee
    back_knee = max(knee, opp_knee) if exercise == "lunge" else knee

    feats = np.array([
        1.0 if exercise == "squat" else 0.0,
        1.0 if exercise == "pushup" else 0.0,
        1.0 if exercise == "lunge" else 0.0,
        active_knee, hip, elbow, torso, body_line, kot, back_knee
    ], dtype=np.float32)

    return feats, float(max(l_vis, r_vis))


def rule_classify_bottom(exercise, feats):
    knee, hip, elbow, torso, body_line, kot, _ = feats[3:]
    if exercise == "squat":
        if torso > 42 or (kot > 0.35 and knee < 135):
            return 2
        if knee > 112:
            return 1
        return 0
    if exercise == "pushup":
        if body_line < 156:
            return 2
        if elbow > 108:
            return 1
        return 0
    # lunge
    if torso > 26 or (kot > 0.32 and knee < 135):
        return 2
    if knee > 112:
        return 1
    return 0


def evaluate_clip(video_path: Path, exercise: str, ort_sess, meta, task_model_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(task_model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    cfg = THRESHOLDS[exercise]
    phase = "READY"
    reps = 0
    min_angle = 180.0
    bottom_feats = None
    rule_rep_preds = []
    onnx_rep_preds = []
    extracted_bottom_features = []
    mean = np.array(meta["feature_mean"], dtype=np.float32)
    std = np.array(meta["feature_std"], dtype=np.float32)

    alpha = 0.35  # EMA smoothing
    smoothed_primary = None
    low_vis_frames = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((total_frames / fps) * 1000)
        res = landmarker.detect_for_video(mp_img, ts_ms)
        if not res.pose_landmarks:
            low_vis_frames += 1
            continue

        feats, vis = extract_features_from_landmarks(res.pose_landmarks[0], exercise)
        if vis < 0.60:
            low_vis_frames += 1
            continue

        raw_primary = feats[5] if cfg["primary"] == "elbow" else feats[3]
        smoothed_primary = raw_primary if smoothed_primary is None else (alpha * raw_primary + (1 - alpha) * smoothed_primary)

        if phase == "READY":
            if smoothed_primary < cfg["descend"]:
                phase = "DESCENDING"
                min_angle = smoothed_primary
                bottom_feats = feats.copy()
        elif phase in ("DESCENDING", "BOTTOM"):
            if smoothed_primary < min_angle:
                min_angle = smoothed_primary
                bottom_feats = feats.copy()
            if smoothed_primary <= cfg["bottom"]:
                phase = "BOTTOM"
            elif smoothed_primary > min_angle + 15:
                phase = "ASCENDING"
        elif phase == "ASCENDING":
            if smoothed_primary >= cfg["up"]:
                reps += 1
                if bottom_feats is not None:
                    extracted_bottom_features.append(bottom_feats)
                    rule_rep_preds.append(rule_classify_bottom(exercise, bottom_feats))
                    norm = ((bottom_feats - mean) / np.where(std > 1e-6, std, 1.0)).reshape(1, -1)
                    logits = ort_sess.run(None, {"input": norm})[0][0]
                    onnx_rep_preds.append(int(np.argmax(logits)))
                phase = "READY"
                min_angle = 180.0

    cap.release()
    landmarker.close()

    # Majority vote across reps in the clip
    def majority(arr):
        if not arr:
            return 0
        vals, counts = np.unique(arr, return_counts=True)
        return int(vals[np.argmax(counts)])

    return {
        "pred_reps": reps,
        "rule_class": majority(rule_rep_preds),
        "onnx_class": majority(onnx_rep_preds),
        "low_vis_ratio": round(low_vis_frames / max(total_frames, 1), 3),
        "bottom_features": extracted_bottom_features,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate FormCoach on labelled workout clips")
    parser.add_argument("--labels", type=str, default="data/labels.csv")
    parser.add_argument("--clips-dir", type=str, default="data/clips")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--extract-features", action="store_true", help="Save extracted bottom-of-rep features CSV")
    args = parser.parse_args()

    if mp is None:
        raise SystemExit("Please install mediapipe first: pip install mediapipe")

    root = Path(__file__).resolve().parent.parent
    labels_path = root / args.labels
    clips_dir = root / args.clips_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((root / "models/model_metadata.json").read_text(encoding="utf-8"))
    ort_sess = ort.InferenceSession(str(root / "models/form_classifier.onnx"), providers=["CPUExecutionProvider"])

    task_model_path = root / "models/pose_landmarker_lite.task"
    if not task_model_path.exists():
        print("Downloading MediaPipe Pose Landmarker Lite model (~5.5 MB)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            str(task_model_path),
        )

    rows = []
    with labels_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    y_true, y_rule, y_onnx = [], [], []
    rep_errors = []
    per_ex_errors = {"squat": [], "pushup": [], "lunge": []}
    failures = []
    extracted_rows = []

    for r in rows:
        clip_path = clips_dir / r["filename"]
        if not clip_path.exists():
            print(f"[SKIP] Missing clip: {clip_path}")
            continue

        ex = r["exercise"].strip().lower()
        true_reps = int(r["true_reps"])
        true_cls = CLASS_TO_IDX[r["true_form_class"].strip()]

        res = evaluate_clip(clip_path, ex, ort_sess, meta, task_model_path)
        err = abs(res["pred_reps"] - true_reps)
        rep_errors.append(err)
        per_ex_errors[ex].append(err)
        y_true.append(true_cls)
        y_rule.append(res["rule_class"])
        y_onnx.append(res["onnx_class"])

        if err > 0 or res["onnx_class"] != true_cls:
            failures.append({
                "filename": r["filename"],
                "exercise": ex,
                "condition": r.get("condition", ""),
                "true_reps": true_reps,
                "pred_reps": res["pred_reps"],
                "true_form": CLASSES[true_cls],
                "rule_form": CLASSES[res["rule_class"]],
                "onnx_form": CLASSES[res["onnx_class"]],
                "low_vis_ratio": res["low_vis_ratio"],
            })

        if args.extract_features:
            for feat in res["bottom_features"]:
                extracted_rows.append(list(feat) + [true_cls])

    if not y_true:
        print("No matching clips found in data/clips/. Run ml/fetch_open_source_clips.py for dataset instructions.")
        return

    summary = {
        "clips_evaluated": len(y_true),
        "overall_rep_mae": round(float(np.mean(rep_errors)), 3),
        "rep_mae_by_exercise": {k: (round(float(np.mean(v)), 3) if v else None) for k, v in per_ex_errors.items()},
        "rule_baseline_report": classification_report(y_true, y_rule, target_names=CLASSES, output_dict=True, zero_division=0),
        "onnx_classifier_report": classification_report(y_true, y_onnx, target_names=CLASSES, output_dict=True, zero_division=0),
        "onnx_confusion_matrix": confusion_matrix(y_true, y_onnx, labels=[0, 1, 2]).tolist(),
        "failure_cases": failures,
    }

    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nEvaluated {len(y_true)} clips | Overall Rep MAE: {summary['overall_rep_mae']}")
    print("Rule-Based Baseline Report:")
    print(classification_report(y_true, y_rule, target_names=CLASSES, zero_division=0))
    print("ONNX Classifier Report:")
    print(classification_report(y_true, y_onnx, target_names=CLASSES, zero_division=0))
    print(f"Saved full results & failure gallery to: {out_dir / 'evaluation_summary.json'}")

    if args.extract_features and extracted_rows:
        out_csv = root / "data/extracted_poses.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(meta["feature_names"] + ["label"])
            writer.writerows(extracted_rows)
        print(f"Saved {len(extracted_rows)} extracted pose rows to {out_csv}")


if __name__ == "__main__":
    main()
