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

## 5. Subgroup Fairness Results (ControlNet Diffusion Audit, $n = 88$)
Evaluated on 88 pose-conditioned synthetic images (4 canonical poses × 11 subgroups × 2 seeds; see `results/fairness_audit_summary.csv` & `results/fairness_audit_chart.png`):

| Subgroup Axis | Best-Performing Subgroup | Weakest-Performing Subgroup | Observed Gap / Finding |
| :--- | :--- | :--- | :--- |
| Skin Tone | Light Skin (100% det, 75.0% form acc, 21.63° MAE) | Medium & Dark Skin (100% det, 62.5% form acc, 20.22°–21.04° MAE) | 100% detection across all tones; 12.5 pp form-acc gap (1/8 image diff) warrants larger real-world test |
| Body Type | Athletic & Plus-Size tied (100% det, 87.5% form acc) | None (parity observed) | Equitable performance: 12.74° MAE (athletic) vs. 12.18° MAE (plus-size) |
| Age (Young vs. 65+) | Older Adult 65+ (100% det, 87.5% form acc, 17.14° MAE) | Young Adult (100% det, 75.0% form acc, 18.06° MAE) | Strong parity across age groups |
| Clothing (Fitted vs. Baggy) | Fitted Athletic (100% det, 0.961 vis, 62.5% form acc) | **Baggy Clothing (75.0% det, 0.715 vis, 50.0% form acc)** | **Largest drop (-25 pp detection, -0.246 visibility)** — motivates `<60%` "Not Sure" gate & fitted-wear tip |
| Lighting (Bright vs. Dim) | Dim Cluttered (100% det, 75.0% form acc, 14.29° MAE) | Bright Clean (100% det, 62.5% form acc, 25.07° MAE) | Occasional SD 1.5 pose drift on bright-room prompts highlights synthetic-to-real audit nuance |

## 6. Known Failures & Mitigations
- **Baggy / Loose Clothing:** Obscures hip and knee joint centres, inflating knee-angle error. *Mitigation:* Camera setup guide recommends fitted sportswear; One-Euro smoothing damps jitter.
- **Strict Side-View Self-Occlusion:** Far-side limbs become occluded during lunges. *Mitigation:* Auto-side selector picks the higher-visibility side; 45° camera angle recommended in Help modal.
- **Low Light / Backlighting:** Causes landmark visibility drops. *Mitigation:* Explicit <60% visibility gate pauses counting and shows "Not Sure" banner.
- **Privacy Guarantee:** Zero frames leave the browser; no accounts, cookies, or telemetry servers.
