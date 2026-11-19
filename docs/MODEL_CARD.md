# Model Card — FormCoach (v1.0)

**Project:** MAIB AI 221 · Project 10 (Sports & Wellness)  
**Model Components:** (1) MediaPipe Pose Landmarker Lite (pre-trained, Apache 2.0); (2) FormCoachPoseMLP ONNX Classifier (`form_classifier.onnx`, 4.01 KB, MIT Licence).  
**Deployment:** 100% client-side in-browser execution on GitHub Pages (WebGL/WASM).

---

## 1. Intended Use
- **Primary User:** Individuals exercising at home using a laptop or smartphone camera.
- **Intended Task:** Real-time repetition counting and general educational posture feedback for three bodyweight exercises: **Squat**, **Push-up**, and **Lunge**.
- **Human-in-the-Loop Design:** Users see live joint angles, skeleton overlays, and confidence indicators. When landmark visibility drops below 60%, the system displays a **"Not Sure — Low Visibility"** state and pauses counting rather than guessing.

## 2. Out-of-Scope Use
- **NOT Medical or Physiotherapy Advice:** FormCoach must not be used for clinical rehabilitation, injury diagnosis, or medical clearance. Users should stop immediately if they feel pain.
- **Unsuitable Scenarios:** Weighted barbell lifting where equipment occludes joints, multi-person gym rooms (model tracks single primary pose), or strict front-facing camera angles (where sagittal-plane knee/torso depth is foreshortened).

## 3. Training & Evaluation Data
- **Pose Estimation Backbone:** MediaPipe Pose Landmarker Lite (BlazePose topology, 33 keypoints, pre-trained by Google, Apache 2.0).
- **ONNX Classifier Dataset:** 4,500 noise-perturbed biomechanical pose vectors (10 normalized joint-angle features across 3 exercises × 3 form classes: `good_form`, `depth_issue`, `alignment_issue`; 80/20 stratified split) plus optional fine-tuning on extracted real clip poses.
- **Video Evaluation Set:** 30+ permissive open-source workout clips (Kaggle Workout/Exercises Video CC0, Pexels, Pixabay) annotated for true rep count, form class, and failure conditions.
- **Generative Fairness Audit Set:** Controlled synthetic images generated offline via Stable Diffusion v1.5 + OpenPose ControlNet across 11 demographic & environmental subgroups.

## 4. Quantitative Performance
| Metric | Baseline (Rule-Based) | Main Model (ONNX MLP) |
| :--- | :---: | :---: |
| Held-Out Biomechanical Accuracy (n=900) | 99.33% | 99.22% |
| Held-Out Macro-F1 (n=900) | 0.9933 | 0.9922 |
| ONNX Model Size / Inference Latency | N/A | 4.01 KB / 0.0012 ms (CPU) |
| Real-World Video Rep-Count MAE (30+ clips) | `[INSERT MAE]` | `[INSERT MAE]` |
| Real-World Form-Error Macro-F1 | `[INSERT F1]` | `[INSERT F1]` |
| Live Browser Speed (Laptop / Phone) | `[INSERT FPS]` / `[INSERT FPS]` | `[INSERT FPS]` / `[INSERT FPS]` |

## 5. Subgroup Fairness Results (ControlNet Diffusion Audit)
*Complete after running `notebooks/01_genai_fairness_audit.ipynb` on Google Colab T4 GPU:*

| Subgroup Axis | Best-Performing Subgroup | Weakest-Performing Subgroup | Observed Gap / Finding |
| :--- | :--- | :--- | :--- |
| Skin Tone | `[INSERT]` | `[INSERT]` | `[INSERT detection & angle MAE gap]` |
| Body Type | `[INSERT]` | `[INSERT]` | `[INSERT detection & angle MAE gap]` |
| Age (Young vs. 65+) | `[INSERT]` | `[INSERT]` | `[INSERT detection & angle MAE gap]` |
| Clothing (Fitted vs. Baggy) | Fitted athletic wear | Baggy hoodie & sweatpants | Expected knee/hip landmark drift under loose fabric |
| Lighting (Bright vs. Dim) | Bright clean room | Dim cluttered room | Lower landmark visibility triggering "Not Sure" state |

## 6. Known Failures & Mitigations
- **Baggy / Loose Clothing:** Obscures hip and knee joint centres, inflating knee-angle error. *Mitigation:* Camera setup guide recommends fitted sportswear; One-Euro smoothing damps jitter.
- **Strict Side-View Self-Occlusion:** Far-side limbs become occluded during lunges. *Mitigation:* Auto-side selector picks the higher-visibility side; 45° camera angle recommended in Help modal.
- **Low Light / Backlighting:** Causes landmark visibility drops. *Mitigation:* Explicit <60% visibility gate pauses counting and shows "Not Sure" banner.
- **Privacy Guarantee:** Zero frames leave the browser; no accounts, cookies, or telemetry servers.
