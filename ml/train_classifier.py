"""
train_classifier.py
Trains the FormCoach Pose-Form Neural Classifier (3-layer MLP: 10 -> 32 -> 16 -> 3),
compares it against the Rule-Based Baseline on a held-out test set, exports the
model to ONNX (IR v8, Opset 13), and verifies numerical equivalence with ONNX Runtime.

Feature Schema (10 dimensions — matches js/geometry.js):
  0: exercise_squat (0 or 1)
  1: exercise_pushup (0 or 1)
  2: exercise_lunge (0 or 1)
  3: kneeAngle (deg)
  4: hipAngle (deg)
  5: elbowAngle (deg)
  6: torsoTiltDeg (deg from vertical)
  7: bodyLineAngle (shoulder-hip-ankle deg)
  8: kneeOverToeRatio (normalized horizontal offset)
  9: backKneeAngle (deg)

Classes:
  0: good_form
  1: depth_issue      ("Go lower" / shallow range of motion)
  2: alignment_issue  ("Back not straight" / "Hips sagging" / "Knees past toes")
"""

import argparse
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

CLASSES = ["good_form", "depth_issue", "alignment_issue"]
FEATURE_NAMES = [
    "exercise_squat",
    "exercise_pushup",
    "exercise_lunge",
    "kneeAngle",
    "hipAngle",
    "elbowAngle",
    "torsoTiltDeg",
    "bodyLineAngle",
    "kneeOverToeRatio",
    "backKneeAngle",
]


# ---------------------------------------------------------------------------
# Minimal Pure-Python ONNX Protobuf Exporter (IR v8, Opset 13, Gemm + Relu)
# Zero extra dependencies; verified immediately against onnxruntime.
# ---------------------------------------------------------------------------
def _varint(val: int) -> bytes:
    out = bytearray()
    while True:
        b = val & 0x7F
        val >>= 7
        if val:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _field_varint(field_num: int, val: int) -> bytes:
    return _varint((field_num << 3) | 0) + _varint(val)


def _field_bytes(field_num: int, data: bytes) -> bytes:
    return _varint((field_num << 3) | 2) + _varint(len(data)) + data


def _field_str(field_num: int, s: str) -> bytes:
    return _field_bytes(field_num, s.encode("utf-8"))


def _tensor_proto(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    parts = []
    for d in arr.shape:
        parts.append(_field_varint(1, int(d)))  # dims
    parts.append(_field_varint(2, 1))  # data_type = FLOAT (1)
    parts.append(_field_str(8, name))  # name
    parts.append(_field_bytes(9, arr.tobytes()))  # raw_data
    return b"".join(parts)


def _value_info_proto(name: str, shape: list) -> bytes:
    dims = []
    for d in shape:
        if isinstance(d, str):
            dim_proto = _field_str(2, d)  # dim_param
        else:
            dim_proto = _field_varint(1, int(d))  # dim_value
        dims.append(_field_bytes(1, dim_proto))
    shape_proto = b"".join(dims)
    tensor_type = _field_varint(1, 1) + _field_bytes(2, shape_proto)  # elem_type=FLOAT
    type_proto = _field_bytes(1, tensor_type)
    return _field_str(1, name) + _field_bytes(2, type_proto)


def _attr_int(name: str, val: int) -> bytes:
    # AttributeProto: name=1, i=3, type=20 (INT=2)
    return _field_str(1, name) + _field_varint(3, val) + _field_varint(20, 2)


def _node_proto(op_type: str, inputs: list, outputs: list, name: str, attrs=None) -> bytes:
    parts = [_field_str(1, i) for i in inputs]
    parts += [_field_str(2, o) for o in outputs]
    parts.append(_field_str(3, name))
    parts.append(_field_str(4, op_type))
    for a in (attrs or []):
        parts.append(_field_bytes(5, a))
    return b"".join(parts)


def export_mlp_to_onnx(mlp: MLPClassifier, out_path: Path):
    """Exports a trained sklearn MLPClassifier (ReLU hidden layers) to ONNX."""
    weights = [w.astype(np.float32) for w in mlp.coefs_]
    biases = [b.astype(np.float32) for b in mlp.intercepts_]

    nodes = []
    initializers = []
    prev = "input"
    n_layers = len(weights)

    for idx, (w, b) in enumerate(zip(weights, biases)):
        w_name = f"W{idx}"
        b_name = f"B{idx}"
        initializers.append(_tensor_proto(w_name, w))
        initializers.append(_tensor_proto(b_name, b))

        gemm_out = "logits" if idx == n_layers - 1 else f"gemm_{idx}"
        # Gemm: Y = A @ B + C (transA=0, transB=0)
        nodes.append(_node_proto("Gemm", [prev, w_name, b_name], [gemm_out], f"Gemm_{idx}"))

        if idx < n_layers - 1:
            relu_out = f"relu_{idx}"
            nodes.append(_node_proto("Relu", [gemm_out], [relu_out], f"Relu_{idx}"))
            prev = relu_out
        else:
            prev = gemm_out

    in_dim = weights[0].shape[0]
    out_dim = weights[-1].shape[1]

    graph_parts = [_field_bytes(1, n) for n in nodes]
    graph_parts.append(_field_str(2, "FormCoachPoseMLP"))
    graph_parts += [_field_bytes(5, init) for init in initializers]
    graph_parts.append(_field_bytes(11, _value_info_proto("input", ["batch", in_dim])))
    graph_parts.append(_field_bytes(12, _value_info_proto("logits", ["batch", out_dim])))
    graph_proto = b"".join(graph_parts)

    opset_import = _field_str(1, "") + _field_varint(2, 13)  # ai.onnx opset 13
    model_proto = (
        _field_varint(1, 8)  # ir_version = 8
        + _field_str(2, "FormCoach-MAIB-AI221")
        + _field_str(3, "1.0")
        + _field_bytes(7, graph_proto)
        + _field_bytes(8, opset_import)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(model_proto)


# ---------------------------------------------------------------------------
# Biomechanical Dataset Generation (with realistic camera & landmark jitter)
# ---------------------------------------------------------------------------
def generate_biomechanical_dataset(n_per_class_per_exercise=500, seed=42):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    def add_samples(ex_idx, label, n, fn):
        for _ in range(n):
            feats = fn()
            one_hot = [0.0, 0.0, 0.0]
            one_hot[ex_idx] = 1.0
            # Add realistic MediaPipe landmark noise (camera jitter, clothing)
            noise = rng.normal(0.0, [2.5, 2.5, 2.5, 2.0, 2.0, 0.04, 2.5])
            noisy = np.array(feats, dtype=np.float32) + noise.astype(np.float32)
            X_list.append(one_hot + noisy.tolist())
            y_list.append(label)

    # --- SQUAT (ex_idx=0) ---
    # Good form: deep knee (72-105), aligned hip (70-105), upright torso (12-36), kneeOverToe (-0.15..0.28)
    add_samples(0, 0, n_per_class_per_exercise, lambda: [
        rng.uniform(72, 105), rng.uniform(70, 105), rng.uniform(140, 175),
        rng.uniform(12, 36), rng.uniform(135, 165), rng.uniform(-0.15, 0.28), rng.uniform(72, 105)
    ])
    # Depth issue: shallow knee (114-142), decent torso (10-36)
    add_samples(0, 1, n_per_class_per_exercise, lambda: [
        rng.uniform(114, 142), rng.uniform(110, 145), rng.uniform(145, 175),
        rng.uniform(10, 36), rng.uniform(145, 172), rng.uniform(-0.1, 0.28), rng.uniform(114, 142)
    ])
    # Alignment issue: excessive forward lean (torso 44-72) OR knees far past toes (0.38-0.75)
    def squat_align():
        if rng.random() < 0.55:
            return [rng.uniform(75, 125), rng.uniform(55, 95), rng.uniform(135, 175),
                    rng.uniform(44, 72), rng.uniform(115, 150), rng.uniform(-0.05, 0.35), rng.uniform(75, 125)]
        else:
            return [rng.uniform(75, 120), rng.uniform(75, 115), rng.uniform(135, 175),
                    rng.uniform(18, 40), rng.uniform(130, 160), rng.uniform(0.38, 0.75), rng.uniform(75, 120)]
    add_samples(0, 2, n_per_class_per_exercise, squat_align)

    # --- PUSH-UP (ex_idx=1) ---
    # Good form: elbow (68-102), straight plank bodyLine (160-180), torso horizontal (68-88)
    add_samples(1, 0, n_per_class_per_exercise, lambda: [
        rng.uniform(160, 180), rng.uniform(160, 180), rng.uniform(68, 102),
        rng.uniform(68, 88), rng.uniform(160, 180), rng.uniform(-0.2, 0.2), rng.uniform(160, 180)
    ])
    # Depth issue: shallow elbow (112-142), straight plank (160-180)
    add_samples(1, 1, n_per_class_per_exercise, lambda: [
        rng.uniform(160, 180), rng.uniform(160, 180), rng.uniform(112, 142),
        rng.uniform(64, 86), rng.uniform(160, 180), rng.uniform(-0.2, 0.2), rng.uniform(160, 180)
    ])
    # Alignment issue: hips sagging or piking (bodyLine 118-152)
    add_samples(1, 2, n_per_class_per_exercise, lambda: [
        rng.uniform(145, 178), rng.uniform(120, 154), rng.uniform(72, 130),
        rng.uniform(55, 85), rng.uniform(118, 152), rng.uniform(-0.2, 0.2), rng.uniform(145, 178)
    ])

    # --- LUNGE (ex_idx=2) ---
    # Good form: front knee (78-104), upright torso (4-22), kneeOverToe (-0.15..0.26), back knee (80-112)
    add_samples(2, 0, n_per_class_per_exercise, lambda: [
        rng.uniform(78, 104), rng.uniform(95, 135), rng.uniform(145, 175),
        rng.uniform(4, 22), rng.uniform(135, 165), rng.uniform(-0.15, 0.26), rng.uniform(80, 112)
    ])
    # Depth issue: shallow lunge (knee 114-142), upright torso (4-22)
    add_samples(2, 1, n_per_class_per_exercise, lambda: [
        rng.uniform(114, 142), rng.uniform(125, 158), rng.uniform(145, 175),
        rng.uniform(4, 22), rng.uniform(145, 170), rng.uniform(-0.1, 0.26), rng.uniform(120, 155)
    ])
    # Alignment issue: torso leaning forward (29-55) OR front knee past toes (0.36-0.72)
    def lunge_align():
        if rng.random() < 0.55:
            return [rng.uniform(80, 125), rng.uniform(85, 125), rng.uniform(140, 175),
                    rng.uniform(29, 55), rng.uniform(125, 155), rng.uniform(0.0, 0.30), rng.uniform(85, 130)]
        else:
            return [rng.uniform(78, 122), rng.uniform(95, 135), rng.uniform(140, 175),
                    rng.uniform(6, 24), rng.uniform(135, 160), rng.uniform(0.36, 0.72), rng.uniform(85, 130)]
    add_samples(2, 2, n_per_class_per_exercise, lunge_align)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def rule_based_predict(X: np.ndarray) -> np.ndarray:
    """Classical rule-based baseline matching js/exerciseLogic.js."""
    preds = []
    for row in X:
        is_sq, is_pu, is_lu = row[0] > 0.5, row[1] > 0.5, row[2] > 0.5
        knee, hip, elbow, torso, body_line, kot, _ = row[3:]
        if is_sq:
            if torso > 42 or (kot > 0.35 and knee < 135):
                preds.append(2)
            elif knee > 112:
                preds.append(1)
            else:
                preds.append(0)
        elif is_pu:
            if body_line < 156:
                preds.append(2)
            elif elbow > 108:
                preds.append(1)
            else:
                preds.append(0)
        else:
            if torso > 26 or (kot > 0.32 and knee < 135):
                preds.append(2)
            elif knee > 112:
                preds.append(1)
            else:
                preds.append(0)
    return np.array(preds, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Train FormCoach Pose Classifier & Export ONNX")
    parser.add_argument("--real-csv", type=str, default=None, help="Optional CSV of real extracted pose features to append/train on")
    parser.add_argument("--out-dir", type=str, default="../models", help="Output directory for ONNX model and metadata")
    args = parser.parse_args()

    out_dir = (Path(__file__).parent / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1. Generating biomechanical pose-feature dataset...")
    X, y = generate_biomechanical_dataset(n_per_class_per_exercise=500, seed=42)

    if args.real_csv and Path(args.real_csv).exists():
        import pandas as pd
        df = pd.read_csv(args.real_csv)
        X_real = df[FEATURE_NAMES].values.astype(np.float32)
        y_real = df["label"].values.astype(np.int64)
        print(f"   Appending {len(X_real)} real labelled frames from {args.real_csv}")
        X = np.vstack([X, X_real])
        y = np.concatenate([y, y_real])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Evaluate Classical Rule-Based Baseline
    baseline_preds = rule_based_predict(X_test)
    baseline_f1 = f1_score(y_test, baseline_preds, average="macro")
    baseline_acc = float(np.mean(baseline_preds == y_test))
    print(f"\n[Baseline: Rule-Based] Accuracy: {baseline_acc:.4f} | Macro-F1: {baseline_f1:.4f}")

    # Train Neural Classifier (3-layer MLP)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    mlp.fit(X_train_s, y_train)

    ml_preds = mlp.predict(X_test_s)
    ml_f1 = f1_score(y_test, ml_preds, average="macro")
    ml_acc = float(np.mean(ml_preds == y_test))
    print(f"[Main Model: ONNX MLP] Accuracy: {ml_acc:.4f} | Macro-F1: {ml_f1:.4f}")
    print("\nClassification Report (ONNX MLP on Held-Out Test Set):")
    print(classification_report(y_test, ml_preds, target_names=CLASSES, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, ml_preds))

    # Export to ONNX
    onnx_path = out_dir / "form_classifier.onnx"
    export_mlp_to_onnx(mlp, onnx_path)
    size_kb = onnx_path.stat().st_size / 1024.0
    print(f"\n2. Exported ONNX model -> {onnx_path} ({size_kb:.2f} KB)")

    # Verify with ONNX Runtime
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    ort_logits = sess.run(None, {"input": X_test_s})[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_sample_ms = elapsed_ms / len(X_test_s)

    # Softmax comparison
    exp_l = np.exp(ort_logits - np.max(ort_logits, axis=1, keepdims=True))
    ort_probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    sk_probs = mlp.predict_proba(X_test_s)
    max_diff = float(np.max(np.abs(ort_probs - sk_probs)))
    print(f"3. ONNX Runtime Verification: max_prob_diff = {max_diff:.2e} | avg latency = {per_sample_ms:.4f} ms/sample")
    assert max_diff < 1e-4, f"ONNX mismatch too high: {max_diff}"

    # Save metadata JSON for browser inference
    meta = {
        "model_name": "FormCoachPoseMLP",
        "version": "1.0",
        "architecture": "MLP (10 -> 32 ReLU -> 16 ReLU -> 3)",
        "feature_names": FEATURE_NAMES,
        "classes": CLASSES,
        "feature_mean": scaler.mean_.tolist(),
        "feature_std": scaler.scale_.tolist(),
        "metrics_held_out": {
            "baseline_accuracy": round(baseline_acc, 4),
            "baseline_macro_f1": round(float(baseline_f1), 4),
            "onnx_mlp_accuracy": round(ml_acc, 4),
            "onnx_mlp_macro_f1": round(float(ml_f1), 4),
            "onnx_size_kb": round(size_kb, 2),
            "onnx_latency_ms": round(per_sample_ms, 4),
        },
    }
    meta_path = out_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"4. Saved metadata -> {meta_path}")


if __name__ == "__main__":
    main()
