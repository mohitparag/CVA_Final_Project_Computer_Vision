# Week 9 One-Page Project Proposal — FormCoach

**Course:** MAIB AI 221 · Computer Vision with Applications  
**Project:** #10 · Sports & Wellness — *FormCoach: Exercise Rep Counter and Posture Feedback*  
**Student Name:** `[Your Name]` · **Student ID:** `[Your ID]`

---

## 1. Problem Statement & User
Millions of people exercise at home without access to a personal trainer or physiotherapist. Poor exercise technique—such as shallow squat depth, forward torso collapse, or sagging hips during push-ups—reduces training benefit and increases joint injury risk. **FormCoach** is a free, privacy-first webcam web application for home exercisers using a laptop or smartphone. It provides real-time repetition counting, skeleton overlays, posture cues, and a session summary entirely on-device.

**Cost of Errors:**
- *False Negative (missed form error):* User reinforces poor biomechanics (e.g., lumbar rounding or knee strain), increasing injury risk.
- *False Positive (incorrect warning):* Frustrates the user and erodes trust.
- *Visibility Failure:* Guessing angles when occluded produces erratic counts; mitigated via an explicit **"Not Sure — Low Visibility"** state (<60% landmark confidence).

## 2. Data & Licences
- **Pre-trained Pose Backbone:** MediaPipe Pose Landmarker Lite (33 3D/2D landmarks, **Apache 2.0** licence).
- **Evaluation Video Set (30+ clips):** Permissive open-source workout video clips covering Squats, Push-ups, and Lunges (correct form and common mistakes) sourced from **Kaggle Workout/Exercises Video Dataset (CC0 Public Domain)**, **Pexels (Pexels Free Licence)**, and **Pixabay (Pixabay Content Licence)**, supplemented with deliberate challenge clips (baggy clothing, side view, low light).
- **Synthetic Fairness Test Set (GenAI):** Generated offline on a free Google Colab T4 GPU using **Stable Diffusion v1.5 (CreativeML Open RAIL-M)** + **OpenPose ControlNet (Apache 2.0)**.

## 3. Computer Vision Methods & Generative AI Plan
- **Signal Processing:** One-Euro adaptive low-pass filter on raw 33 landmarks to eliminate jitter without lag.
- **Repetition State Machine:** Hysteresis state machine (`READY → DESCENDING → BOTTOM → ASCENDING`) with duration debounce.
- **Baseline Model:** Classical biomechanical joint-angle rule engine (knee, hip, elbow, torso tilt, plank line, knee-over-toe ratio).
- **Main CV Model (ONNX):** 3-layer Neural MLP classifier (`10 → 32 → 16 → 3`) trained on normalized pose features and exported to **ONNX (4.01 KB)**, running in-browser via **ONNX Runtime Web**.
- **Generative AI Innovation:** Pose-conditioned diffusion (OpenPose ControlNet) synthesizes identical exercise poses across skin tones, body types, ages, clothing styles, and lighting to conduct a controlled Week 10 subgroup fairness audit.

## 4. Evaluation Metrics & Deployment Plan
- **Quantitative Metrics:** Rep-count Mean Absolute Error (MAE), per-mistake Precision/Recall/Macro-F1 (Rule-Based vs. ONNX), Confusion Matrix, live FPS on laptop & phone (target ≥20 FPS), and subgroup detection/accuracy/angle-MAE on the synthetic fairness set.
- **Approved Free Deployment:** **GitHub Pages** (static HTTPS site; zero paid APIs, zero backend servers, 100% on-device execution). Local fallback via `python -m http.server 8000`.
