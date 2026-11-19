# FormCoach — On-Device AI Exercise Rep Counter & Posture Coach

**MAIB AI 221 · Computer Vision with Applications · Final Project (Project 10: Sports & Wellness)**  
S P Jain School of Global Management

FormCoach is a **100% browser-based, privacy-first** exercise repetition counter and real-time posture coach for **Squats**, **Push-ups**, and **Lunges**. All computer vision inference runs locally on the user's laptop or smartphone using **MediaPipe Pose Landmarker (Apache 2.0)** and a student-trained **ONNX Pose-Form Classifier** via **ONNX Runtime Web**—no video is ever recorded, stored, or uploaded.

- **Live App URL (GitHub Pages):** https://mohitparag.github.io/CVA_Final_Project_Computer_Vision/
- **Public Repository:** https://github.com/mohitparag/CVA_Final_Project_Computer_Vision
- **Zero Paid Services:** 100% free static hosting (GitHub Pages), open-weight models, and free Google Colab T4 GPU for the Generative AI fairness audit.

---

## Features

- **3 Supported Exercises:** Squat, Push-up, and Lunge.
- **Dual Input Modes:** Live webcam (desktop & mobile front/rear camera) + video/image upload for testing clips and inspecting failure cases.
- **Real-Time CV Pipeline:**
  - **Pose Estimation:** MediaPipe Pose Landmarker Lite (WebGL GPU / WASM fallback, ≥20 FPS on laptop).
  - **Adaptive Smoothing:** One-Euro Filter (`js/smoothing.js`) toggleable live to compare smoothed vs. raw landmark jitter.
  - **Repetition State Machine:** Hysteresis-based (`READY → DESCENDING → BOTTOM → ASCENDING`) with duration debounce.
  - **Baseline vs. Main Model:** Switch live between **Rule-Based Biomechanical Checks (Baseline)**, **ONNX Pose-Form MLP Classifier (4.01 KB, Main Model)**, and **Hybrid Mode**.
- **Clear "Not Sure" Visibility Gating:** Pauses counting and warns the user whenever key joint visibility drops below 60%.
- **Stretch Goals Included:**
  - **Voice Cues:** Browser-native `window.speechSynthesis` real-time coaching.
  - **Session Summary & PDF Export:** Rep-by-rep breakdown, good-form percentage, most common mistake, and one-click print/export to PDF.

---

## Repository Structure

```text
formcoach/
├── index.html                        # Static web app entry point (GitHub Pages root)
├── css/styles.css                    # Responsive UI + print-to-PDF stylesheet
├── js/
│   ├── app.js                        # Main controller (webcam, upload, canvas, telemetry, PDF)
│   ├── poseEngine.js                 # MediaPipe Pose Landmarker wrapper (Apache 2.0)
│   ├── smoothing.js                  # One-Euro Filter landmark smoother
│   ├── geometry.js                   # Biomechanical joint-angle & 10-dim feature extractor
│   ├── exerciseLogic.js              # Hysteresis rep state machine + rule-based baseline
│   ├── onnxClassifier.js             # ONNX Runtime Web inference wrapper
│   └── voiceCoach.js                 # Web Speech API real-time audio cues
├── models/
│   ├── form_classifier.onnx          # Trained 3-layer MLP exported to ONNX (4.01 KB)
│   └── model_metadata.json           # Scaler stats, class names, held-out metrics
├── ml/
│   ├── requirements.txt              # Python ML & evaluation dependencies
│   ├── train_classifier.py           # Trains MLP vs. rule-based baseline & exports ONNX
│   ├── fetch_open_source_clips.py    # Prepares data/labels.csv & open-source clip guide
│   └── evaluate_videos.py            # Automated rep MAE, precision/recall & failure gallery
├── notebooks/
│   └── 01_genai_fairness_audit.ipynb # Google Colab: OpenPose ControlNet + SD 1.5 fairness audit
├── docs/
│   ├── PROPOSAL_WEEK9.md             # 1-page Week 9 proposal
│   ├── MODEL_CARD.md                 # 1-page Responsible AI Model Card
│   ├── TECHNICAL_REPORT.md           # 3,000–4,000 word technical report draft
│   └── DEMO_AND_PRESENTATION.md      # 3–5 min demo video script & 10-min slide outline
└── LICENSE                           # MIT Licence + third-party model/dataset notices
```

---

## Quick Start (Run Locally)

Because ES modules and ONNX Runtime Web require HTTP (not `file://`), start any local static server from inside `formcoach/`:

```bash
# Option A: Python (built-in)
cd formcoach
python -m http.server 8000

# Option B: Node.js
npx serve . -l 8000
```

Then open **http://localhost:8000** in Chrome, Edge, or Safari.

---

## Deploy Free to GitHub Pages (2 Minutes)

1. Create a **public** GitHub repository (e.g., `formcoach-cv`).
2. Push the contents of `formcoach/` to the `main` branch:
   ```bash
   cd formcoach
   git init
   git add .
   git commit -m "Initial FormCoach release"
   git branch -M main
   git remote add origin https://github.com/mohitparag/CVA_Final_Project_Computer_Vision.git
   git push -u origin main
   ```
3. In GitHub, go to **Settings → Pages → Build and deployment**:
   - Source: **Deploy from a branch**
   - Branch: **`main`** / **`/ (root)`** → Click **Save**.
4. Within ~60 seconds, your HTTPS live app will be available at https://mohitparag.github.io/CVA_Final_Project_Computer_Vision/ (HTTPS is required by mobile & laptop browsers for webcam access).

---

## Reproducing ML Training, Video Evaluation & GenAI Fairness Audit

### 1. Retrain & Export ONNX Classifier
```bash
python ml/train_classifier.py
# Outputs: models/form_classifier.onnx (4.01 KB) & models/model_metadata.json
```

### 2. Evaluate on 30+ Open-Source Workout Clips
```bash
python ml/fetch_open_source_clips.py   # Creates data/clips/ and data/labels.csv template
pip install -r ml/requirements.txt
python ml/evaluate_videos.py --labels data/labels.csv --clips-dir data/clips --out-dir results/
```

### 3. Run Generative AI Fairness Audit (Free Google Colab T4 GPU)
Upload `notebooks/01_genai_fairness_audit.ipynb` to [Google Colab](https://colab.research.google.com/), set **Runtime → Change runtime type → T4 GPU**, and click **Run all**. The notebook synthesizes a pose-conditioned fairness test set using Stable Diffusion v1.5 + OpenPose ControlNet and outputs subgroup accuracy/MAE tables and charts.

---

## Licences & Responsible AI

- **Code:** MIT License (see `LICENSE`).
- **MediaPipe Pose Landmarker:** Apache License 2.0.
- **ONNX Runtime Web:** MIT License.
- **OpenPose ControlNet / Stable Diffusion v1.5:** Apache 2.0 / CreativeML Open RAIL-M (offline audit only).
- **Disclaimer:** FormCoach is an educational academic prototype and **not** medical or physiotherapy advice. See `docs/MODEL_CARD.md` for full intended use, subgroup fairness findings, and limitations.
