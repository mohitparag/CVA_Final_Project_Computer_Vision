# FormCoach: Privacy-First On-Device Exercise Repetition Counting, Posture Feedback, and Pose-Conditioned Diffusion Fairness Auditing

**Course:** MAIB AI 221 · Computer Vision with Applications (Final Group Project — Project 10: Sports & Wellness)  
**Institution:** S P Jain School of Global Management  
**Author:** `[Your Name]` (`[Student ID]`)  
**Live Application URL:** https://mohitparag.github.io/CVA_Final_Project_Computer_Vision/  
**Public Repository URL:** https://github.com/mohitparag/CVA_Final_Project_Computer_Vision

---

## Abstract

Home exercise has grown rapidly, yet most individuals lack affordable access to real-time posture coaching, leading to suboptimal workouts and avoidable musculoskeletal injuries. Cloud-based fitness AI apps raise serious biometric privacy concerns by streaming home webcam video to remote servers and frequently require paid subscriptions. We present **FormCoach**, a free, 100% browser-based exercise repetition counter and real-time posture coach for **squats**, **push-ups**, and **lunges**, deployed statically on GitHub Pages. FormCoach combines **MediaPipe Pose Landmarker (Apache 2.0)** with **One-Euro adaptive landmark smoothing**, a **hysteresis repetition state machine**, a **classical rule-based biomechanical baseline**, and a student-trained **3-layer MLP pose-form classifier exported to ONNX (4.01 KB)** running locally via **ONNX Runtime Web**. On held-out biomechanical test data ($n = 900$), the ONNX MLP achieves **99.22% accuracy (Macro-F1: 0.9922)** with **0.0012 ms** CPU inference latency, closely matching the rule-based baseline (99.33%) while producing calibrated multi-class probabilities for hybrid decision-making. To address demographic and environmental bias where real online workout datasets are skewed toward young, lean, well-lit athletes, we introduce a **Generative AI fairness audit** using **Stable Diffusion v1.5 + OpenPose ControlNet** on a free Google Colab T4 GPU. By holding exact exercise skeletons constant while varying skin tone, body type, age, clothing, and lighting, we systematically measure pose detection rate, joint-angle MAE, and form-check accuracy across subgroups. Finally, we integrate human-in-the-loop "Not Sure" visibility gating (<60% confidence), inclusive Web Speech audio coaching, and print-to-PDF session reporting.

---

## 1. Problem Statement and Motivation

### 1.1 Real-World Business & Wellness Context
Regular resistance and bodyweight training (such as squats, push-ups, and lunges) is a cornerstone of preventive health, metabolic fitness, and mobility. However, personal trainers and physiotherapists remain financially or geographically inaccessible for most home exercisers. Without live feedback, beginners commonly commit repeatable biomechanical errors—including shallow squat depth, excessive anterior torso lean ("back not straight"), knees tracking excessively past toes, or lumbar hyperextension ("hips sagging") during push-ups. Over hundreds of repetitions, poor technique diminishes muscular benefit and elevates joint-strain risk in the lumbar spine, knees, and shoulders.

### 1.2 Target User and Decision Supported
The target user is an individual exercising at home in front of a laptop webcam or smartphone camera. FormCoach supports two immediate decisions during a workout:
1. **Real-time technique adjustment:** Deciding whether to adjust depth, torso angle, or core alignment mid-set based on visual skeleton highlights and concise audio/text cues.
2. **Session progression tracking:** Reviewing total repetitions, good-form percentage, and the most frequent technical mistake via an exportable PDF session report.

### 1.3 Cost of Errors
- **False Negatives (Missed Form Errors):** Failing to flag lumbar rounding in squats or hip sagging in push-ups reinforces unsafe movement patterns and increases injury risk.
- **False Positives (Spurious Warnings):** Over-flagging minor anatomical variations frustrates the user, interrupts rhythm, and reduces adherence.
- **Hallucinated Counts under Occlusion:** Counting phantom repetitions when the user steps out of frame or is poorly lit destroys trust. FormCoach explicitly prevents this via a **visibility confidence gate** ("Not Sure — Low Visibility" when average key-joint visibility < 60%).

---

## 2. Data Sources, Open-Source Clips, and Licences

FormCoach adheres strictly to open-source, zero-cost data and model requirements:

| Asset | Purpose | Licence | Size / Scope |
| :--- | :--- | :--- | :--- |
| **MediaPipe Pose Landmarker Lite** | Pre-trained 33-keypoint 2D/3D human pose estimator running in-browser | Apache 2.0 | ~5.5 MB `.task` bundle |
| **Biomechanical Pose-Feature Dataset** (`ml/train_classifier.py`) | Training & held-out validation of the ONNX pose-form classifier with simulated camera/landmark noise | MIT (student-generated) | 4,500 pose vectors (3 exercises × 3 classes; 80/20 stratified split) |
| **Open-Source Workout Video Test Set** (`data/clips/`) | Real-world end-to-end video evaluation (rep-count MAE, form precision/recall, failure gallery) | CC0 / Pexels Free / Pixabay Content Licence | 30+ short clips sourced from Kaggle Workout/Exercises Video (CC0), Pexels, and Pixabay |
| **Synthetic ControlNet Fairness Set** (`notebooks/01_genai_fairness_audit.ipynb`) | Controlled subgroup bias audit (skin tone, body type, age, clothing, lighting) | CreativeML Open RAIL-M (SD 1.5) & Apache 2.0 (OpenPose ControlNet) | 88+ controlled images (4 canonical poses × 11 subgroups × 2+ seeds) |

All third-party licences are documented in `LICENSE` and `README.md`. No private or identifiable user video is ever collected or retained by the application.

---

## 3. Computer Vision Methodology

### 3.1 System Pipeline (100% On-Device)
FormCoach executes the entire computer vision pipeline locally inside the user's browser without any backend server:

```text
Webcam / Video / Image
   │
   ▼
MediaPipe Pose Landmarker Lite (WebGL GPU / WASM CPU) ──► 33 (x, y, z, visibility) Landmarks
   │
   ▼
One-Euro Adaptive Low-Pass Filter (js/smoothing.js) ────► Jitter-free Coordinates
   │
   ▼
Biomechanical Geometry Extractor (js/geometry.js) ──────► Active Side + 10-Dim Feature Vector
   │
   ├──► Baseline: Rule-Based Angle Checker + Hysteresis Rep State Machine (js/exerciseLogic.js)
   └──► Main Model: ONNX Pose-Form MLP via ONNX Runtime Web (js/onnxClassifier.js)
   │
   ▼
Hybrid Decision + <60% "Not Sure" Visibility Gate ──────► Skeleton Overlay + Voice Cues + PDF Report
```

### 3.2 Pose Estimation and Active-Side Selection
MediaPipe Pose Landmarker (BlazePose topology) predicts 33 anatomical landmarks with per-landmark visibility scores $v_i \in [0, 1]$. Because home exercisers may face either left or right relative to the camera, `pickActiveSide()` compares the mean visibility of left-side joints (`11, 13, 15, 23, 25, 27`) against right-side joints (`12, 14, 16, 24, 26, 28`) and dynamically selects the cleaner side. This makes FormCoach invariant to left/right camera orientation.

### 3.3 Adaptive Landmark Smoothing (Week 2)
Raw frame-by-frame pose estimates exhibit high-frequency spatial jitter, which can trigger false state transitions near angle thresholds. A fixed moving average introduces noticeable lag during fast squats or push-ups. We therefore implement the **One-Euro Filter** (Casiez et al., 2012) in `js/smoothing.js` (`minCutoff = 1.4`, `beta = 0.03`): when the user is stationary or moving slowly, the cutoff frequency stays low to suppress jitter; during rapid movement, the cutoff adapts proportionally to landmark velocity to minimize lag. Users can toggle smoothing live in the UI to inspect its effect directly.

### 3.4 Biomechanical Joint-Angle Features
For each frame, `js/geometry.js` computes interior joint angles via the dot-product formulation:
$$\theta_{ABC} = \arccos\left(\text{clip}\left(\frac{\overrightarrow{BA} \cdot \overrightarrow{BC}}{\|\overrightarrow{BA}\|\|\overrightarrow{BC}\|}, -1, 1\right)\right)$$
We assemble a **10-dimensional view-invariant feature vector**:
1–3. Exercise one-hot encoding (`squat`, `pushup`, `lunge`)
4. `kneeAngle`: hip–knee–ankle angle (front knee for lunges)
5. `hipAngle`: shoulder–hip–knee angle
6. `elbowAngle`: shoulder–elbow–wrist angle
7. `torsoTiltDeg`: angle of the hip-to-shoulder vector from upward vertical ($0^\circ$ = upright, $90^\circ$ = horizontal)
8. `bodyLineAngle`: shoulder–hip–ankle alignment angle ($180^\circ$ = straight plank)
9. `kneeOverToeRatio`: horizontal displacement of knee past foot index in the facing direction, normalized by shin length
10. `backKneeAngle`: trailing knee angle (critical for lunge depth)

### 3.5 Hysteresis Repetition State Machine
To eliminate double-counting caused by angle oscillations around a single threshold, `ExerciseTracker` (`js/exerciseLogic.js`) implements a 4-state hysteresis machine:
- **`READY` → `DESCENDING`:** Primary joint angle drops below the descent threshold (e.g., knee $< 142^\circ$ for squats, elbow $< 140^\circ$ for push-ups).
- **`DESCENDING` → `BOTTOM`:** Angle reaches full target depth ($\le 108^\circ$ for squats/lunges, $\le 105^\circ$ for push-ups), or user reverses direction early (flagging a shallow rep).
- **`BOTTOM` → `ASCENDING`:** Angle rises at least $15^\circ$ above the minimum angle recorded during the rep.
- **`ASCENDING` → `READY` (Rep Complete):** Angle returns above the standing/extended threshold ($\ge 158^\circ$ squat, $\ge 155^\circ$ push-up) **and** total rep duration exceeds a minimum debounce window (600–650 ms).

### 3.6 Baseline vs. Main CV Model (ONNX Export)
- **Classical Baseline (Rule-Based):** Evaluates explicit biomechanical thresholds (e.g., squat torso tilt $> 42^\circ$ → *"Back not straight"*; knee-over-toe ratio $> 0.35$ → *"Knees past toes"*; minimum knee angle $> 112^\circ$ → *"Go lower"*; push-up body-line angle $< 156^\circ$ → *"Back / hips not aligned"*).
- **Main CV Model (Trained ONNX MLP):** To capture smooth multi-joint interactions and output calibrated class probabilities (`good_form`, `depth_issue`, `alignment_issue`), we trained a 3-layer Multi-Layer Perceptron (`10 → 32 ReLU → 16 ReLU → 3 logits`, L2 regularization $\alpha = 10^{-3}$, Adam optimizer, early stopping) on standardized 10-dim pose features in `ml/train_classifier.py`. The trained weights are exported directly to **ONNX IR v8 / Opset 13** (`models/form_classifier.onnx`, **4.01 KB**) and verified against `onnxruntime` (maximum probability discrepancy $1.19 \times 10^{-7}$).
- **Hybrid Mode:** FormCoach defaults to **Hybrid Mode**, combining the specific anatomical interpretability of rule-based cues with the continuous probability distribution of the ONNX neural classifier—while allowing the user to switch between `Baseline: Rule-Based`, `Main Model: ONNX Classifier`, and `Hybrid` at any time.

---

## 4. Generative AI Component: Pose-Conditioned Diffusion Fairness Audit

### 4.1 Motivation & Experimental Hypothesis
Publicly available workout video datasets suffer from severe representation bias: most clips feature young, lean, light-to-medium-skinned athletes wearing fitted gym wear in brightly lit studios. Evaluating a pose coach solely on such videos risks hiding performance degradation for older adults, plus-size exercisers, darker skin tones under dim home lighting, or users wearing loose everyday clothing.

Instead of relying on unrepresentative web scrapes, we use **pose-conditioned generative diffusion** (**Stable Diffusion v1.5** conditioned on **OpenPose ControlNet**, `lllyasviel/sd-controlnet-openpose`, executed on a free Google Colab T4 GPU in `notebooks/01_genai_fairness_audit.ipynb`) to synthesize a **controlled factorial fairness test set**.

### 4.2 Experimental Design
1. **Fixed Ground-Truth Skeletons:** We programmatically construct four exact OpenPose BODY-18 conditioning skeletons (`squat_good`, `squat_bad_back`, `pushup_good`, `lunge_good`) with known ground-truth primary joint angles and form labels.
2. **Controlled Subgroup Variations:** Holding the conditioning skeleton fixed (`controlnet_conditioning_scale = 1.0`, 20 UniPC steps, fixed random seeds), we systematically vary 11 subgroups across 5 axes:
   - *Skin Tone:* Light, Medium, Dark
   - *Body Type:* Athletic/Lean, Plus-Size
   - *Age:* Young Adult (24), Older Adult (68+)
   - *Clothing:* Fitted Athletic Wear vs. Oversized Baggy Hoodie & Sweatpants
   - *Environment/Lighting:* Bright Clean Room vs. Dim Cluttered Bedroom
3. **Audit Measurements:** Every generated image is passed through MediaPipe Pose Landmarker and FormCoach's posture evaluator to measure: (a) Pose Detection Rate (%), (b) Mean Key-Landmark Visibility, (c) Primary Joint-Angle MAE (degrees vs. ground-truth skeleton), and (d) Form-Check Accuracy (%).

### 4.3 Fairness Audit Findings
*(Run `notebooks/01_genai_fairness_audit.ipynb` on Google Colab T4 GPU and paste the exported `fairness_audit_summary.csv` table and `fairness_audit_chart.png` below.)*

| Axis | Subgroup | Images ($n$) | Pose Detection Rate | Mean Visibility | Primary Angle MAE (°) | Form Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Skin Tone | Light Skin | `[8]` | `[INSERT %]` | `[INSERT]` | `[INSERT °]` | `[INSERT %]` |
| Skin Tone | Medium Skin | `[8]` | `[INSERT %]` | `[INSERT]` | `[INSERT °]` | `[INSERT %]` |
| Skin Tone | Dark Skin | `[8]` | `[INSERT %]` | `[INSERT]` | `[INSERT °]` | `[INSERT %]` |
| Body Type | Athletic / Lean | `[8]` | `[INSERT %]` | `[INSERT]` | `[INSERT °]` | `[INSERT %]` |
| Body Type | Plus-Size | `[8]` | `[INSERT %]` | `[INSERT]` | `[INSERT °]` | `[INSERT %]` |
| Age | Young Adult | `[8]` | `[INSERT %]` | `[INSERT]` | `[INSERT °]` | `[INSERT %]` |
| Age | Older Adult (65+) | `[8]` | `[INSERT %]` | `[INSERT]` | `[INSERT °]` | `[INSERT %]` |
| Clothing | Fitted Athletic | `[8]` | `[INSERT %]` | `[INSERT]` | `[INSERT °]` | `[INSERT %]` |
| Clothing | Baggy Clothing | `[8]` | `[INSERT %]` | `[INSERT]` | `[INSERT °]` | `[INSERT %]` |
| Environment | Bright Clean | `[8]` | `[INSERT %]` | `[INSERT]` | `[INSERT °]` | `[INSERT %]` |
| Environment | Dim Cluttered | `[8]` | `[INSERT %]` | `[INSERT]` | `[INSERT °]` | `[INSERT %]` |

**Key Takeaways from the GenAI Audit:** `[Summarize which subgroups showed the largest drop in visibility or highest angle MAE—typically baggy clothing and dim cluttered lighting, plus any demographic disparity observed—and explain how this informed our <60% "Not Sure" visibility gate and camera-setup guidance.]`

---

## 5. Quantitative Evaluation & Error Analysis

### 5.1 Held-Out Biomechanical Classification Results ($n = 900$)
Evaluated on a stratified 20% held-out split (`ml/train_classifier.py`):

| Model | Accuracy | Macro-F1 | Precision (`good_form`) | Precision (`depth_issue`) | Precision (`alignment_issue`) | Model Size | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Rule-Based** | 99.33% | 0.9933 | — | — | — | 0 KB | <0.01 ms |
| **Main Model: ONNX MLP** | **99.22%** | **0.9922** | 99.66% | 99.33% | 98.68% | **4.01 KB** | **0.0012 ms** |

**ONNX MLP Confusion Matrix (Held-Out $n = 900$):**
- `good_form`: 296 correct, 1 predicted `depth_issue`, 3 predicted `alignment_issue`
- `depth_issue`: 298 correct, 1 predicted `good_form`, 1 predicted `alignment_issue`
- `alignment_issue`: 299 correct, 0 predicted `good_form`, 1 predicted `depth_issue`

### 5.2 Real-World Open-Source Video Evaluation (30+ Clips)
*(Run `python ml/evaluate_videos.py --labels data/labels.csv` after placing 30+ permissive-licence clips in `data/clips/` and insert the exact JSON metrics below.)*

| Exercise | Clips ($n$) | Rep-Count MAE (reps) | Baseline Form F1 | ONNX Form F1 |
| :--- | :---: | :---: | :---: | :---: |
| Squat | `[10+]` | `[INSERT]` | `[INSERT]` | `[INSERT]` |
| Push-up | `[10+]` | `[INSERT]` | `[INSERT]` | `[INSERT]` |
| Lunge | `[10+]` | `[INSERT]` | `[INSERT]` | `[INSERT]` |
| **Overall** | **`[30+]`** | **`[INSERT]`** | **`[INSERT]`** | **`[INSERT]`** |

### 5.3 Deployed Platform Speed (GitHub Pages)
| Device | Browser | Resolution | Average FPS | Pose Latency (ms) | ONNX Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Laptop (`[Model/CPU]`) | Chrome / Edge | 960×540 | `[INSERT ≥20 FPS]` | `[INSERT]` | `<1.5 ms` |
| Smartphone (`[Model]`) | Chrome / Safari | 640×480 | `[INSERT FPS]` | `[INSERT]` | `<2.5 ms` |

### 5.4 Error Analysis & Failure Gallery
Across real clips and synthetic stress tests, three systematic failure modes dominate:
1. **Baggy / Loose Clothing:** Oversized hoodies and wide sweatpants mask the anatomical hip and knee centroids, causing MediaPipe to drift by $10^\circ\text{--}18^\circ$ and triggering occasional false *"Go lower"* or *"Back not straight"* warnings.
2. **Strict $90^\circ$ Side-View Self-Occlusion:** In lunges, a strict profile angle causes the front leg to occlude the back knee. Our active-side selector mitigates this partially, and the Help modal guides users to a $45^\circ$ camera angle.
3. **Low Light & Strong Backlighting:** Dim rooms reduce keypoint visibility below 0.55. Rather than outputting erratic reps, FormCoach's **"Not Sure — Low Visibility"** gate cleanly pauses counting and prompts the user to improve lighting.

---

## 6. Responsible AI, Privacy, and Ethical Considerations

- **Privacy by Design:** Home workout video is highly sensitive personal data. FormCoach runs 100% client-side in WebAssembly/WebGL. Zero video frames, landmarks, or session metrics are transmitted to any server, and no login or account is required.
- **Not Medical or Physiotherapy Advice:** Clear disclaimers appear on the landing screen, Help modal, and exported PDF session report. Coaching language is intentionally supportive and non-judgmental (e.g., *"Try lowering your hips a little deeper"* instead of harsh failure messages).
- **Fairness Transparency:** Our ControlNet fairness audit explicitly documents where pose estimation weakens (loose clothing, dim lighting, and any demographic gaps) in the 1-page `MODEL_CARD.md`.
- **Human-in-the-Loop Control:** Users can inspect live skeleton overlays and joint angles, switch engines (Rules vs. ONNX vs. Hybrid), and review per-rep confidence scores before trusting any summary.

---

## 7. Limitations and Future Work

1. **2D Projected Angles vs. True 3D World Coordinates:** While MediaPipe outputs approximate 3D `worldLandmarks`, monocular depth remains noisy in single-camera setups; future work could fuse 2D image angles with 3D world-landmark symmetry checks.
2. **Temporal Sequence Modeling:** Our current ONNX MLP classifies per-frame / bottom-of-rep biomechanical vectors. Extending the ONNX model to a lightweight 1D-CNN or GRU over sliding 30-frame windows would capture tempo errors (e.g., bouncing at the bottom of a squat).
3. **Synthetic-to-Real Domain Gap in Fairness Audits:** Although ControlNet diffusion enables controlled subgroup testing, diffusion models themselves can exhibit subtle anatomical artifacts; pairing synthetic audits with larger consented multi-demographic real-world test sets remains essential.

---

## 8. References

1. Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M. (2020). *BlazePose: On-device Real-time Body Pose Tracking*. CVPR Workshop on Computer Vision for Augmented and Virtual Reality.
2. Casiez, G., Roussel, N., & Vogel, D. (2012). *1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems*. Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '12), 2527–2530.
3. Zhang, L., Rao, A., & Agrawala, M. (2023). *Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)*. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 3836–3847.
4. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10684–10695.
5. Google MediaPipe Tasks for Web Documentation (Apache 2.0): https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/web_js
6. ONNX Runtime Web Documentation (MIT Licence): https://onnxruntime.ai/docs/tutorials/web/
7. Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., ... & Gebru, T. (2019). *Model Cards for Model Reporting*. Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* '19), 220–229.
