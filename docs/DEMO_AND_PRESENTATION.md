# Demo Video Script (3–5 Min) & Presentation Outline (10 Min)

## Part 1: Demo Video Script (Target: 4 Minutes)

Record your screen + voiceover using OBS Studio or Loom (free) while running the **live GitHub Pages URL** (`https://<username>.github.io/<repo>/`).

| Time | Segment | What to Show on Screen | Key Narration Points |
| :--- | :--- | :--- | :--- |
| **0:00 – 0:30** | **Intro & Privacy** | Open live GitHub Pages URL. Point to the green **On-Device Only** badge and open **Help & Safety** modal briefly. | "This is FormCoach, a 100% browser-based AI exercise coach deployed free on GitHub Pages. No video ever leaves your device." |
| **0:30 – 1:30** | **Case 1: Good-Form Squat (Live Webcam or Clean Clip)** | Select **Squat** + **Hybrid (Rules + ONNX ML)**. Toggle **Voice Cues ON**. Perform/play 3–4 good squats. Point out ≥20 FPS HUD, smooth skeleton, green angles, ONNX probability bars, and rep counter incrementing. | Show live joint angles, ONNX classifier agreeing with rules, and One-Euro smoothing toggle briefly to show jitter reduction. |
| **1:30 – 2:30** | **Case 2: Posture Mistake Detection (Push-up or Lunge)** | Switch to **Push-up** (or **Lunge**). Upload or perform a clip with **hips sagging** or **shallow depth**. Point out amber/red highlighted joints, real-time cue (*"Brace your core..."* / *"Go lower"*), and ONNX `Alignment` / `Depth` bar spiking. | Show switching between **Baseline: Rule-Based** and **Main Model: ONNX Classifier**, then click **Session Summary & PDF** to show the rep-by-rep report and Print-to-PDF preview. |
| **2:30 – 3:30** | **Case 3: Honest Failure Case (Required by Brief!)** | Upload or demonstrate a **low-light**, **baggy clothing**, or **strict side-view** clip (or step partially out of frame). Show landmark visibility dropping below 60% and the yellow **"Not Sure — Low Visibility"** banner pausing rep counting. | Explain *why* it fails (occluded hip/knee landmarks under baggy fabric or dim light) and how our Responsible AI visibility gate prevents wrong counts instead of hallucinating. |
| **3:30 – 4:00** | **GenAI Fairness Audit & Wrap-up** | Briefly show `fairness_audit_chart.png` from Google Colab (OpenPose ControlNet synthetic test set across skin tones, body types, ages, clothing, and lighting). | Summarize the key subgroup finding and reiterate that FormCoach is educational support, not medical advice. |

---

## Part 2: Final Presentation Outline (10 Minutes + 5 Minutes Q&A)

1. **Slide 1 — Title & Live URL** (30s): Project 10 FormCoach, live GitHub Pages QR code so the marker can test it on their phone!
2. **Slide 2 — Real-World Problem & Cost of Errors** (1m): Home fitness injury risk vs. inaccessible coaching; false negative vs. false positive costs; privacy requirement.
3. **Slide 3 — System Architecture (100% On-Device)** (1m): Browser pipeline diagram (`Webcam/Upload → MediaPipe Pose Landmarker → One-Euro Filter → 10-dim Biomechanics → Hysteresis State Machine + ONNX MLP → UI/Voice/PDF`).
4. **Slide 4 — Dataset & Licences** (1m): MediaPipe (Apache 2.0), 30+ permissive open-source workout clips (Kaggle CC0, Pexels, Pixabay), biomechanical feature dataset.
5. **Slide 5 — Baseline vs. Main ONNX Model** (1.5m): Classical angle rules vs. trained 3-layer MLP exported to ONNX (4.01 KB, 0.0012 ms CPU, 99.22% held-out Macro-F1); why Hybrid mode combines rule interpretability with ML generalization.
6. **Slide 6 — Generative AI Innovation: Pose-Conditioned Fairness Audit** (1.5m): Stable Diffusion v1.5 + OpenPose ControlNet on free Colab T4 GPU; show sample synthetic images holding the exact skeleton constant across skin tone, body type, age, clothing, and lighting.
7. **Slide 7 — Quantitative Evaluation & Subgroup Results** (1.5m): Rep-count MAE table, form-error Precision/Recall/Confusion Matrix, laptop & phone FPS, and the ControlNet fairness chart.
8. **Slide 8 — Error Analysis & Failure Gallery** (1m): Baggy clothing, strict side-view occlusion, low light, and how the <60% "Not Sure" visibility gate mitigates bad advice.
9. **Slide 9 — Responsible AI, Model Card & Limitations** (1m): On-device privacy, non-medical disclaimer, inclusive language, honest subgroup gaps, and future work (3D world-landmark multi-view fusion).

### Likely Q&A Questions to Prepare For
- *Why use a 10-dim biomechanical feature vector for the ONNX model instead of raw image pixels?* → View-invariant, tiny (4.01 KB), runs in <1 ms in-browser alongside MediaPipe without dropping below 20 FPS, and avoids memorizing background/demographic shortcuts.
- *Why did you need OpenPose ControlNet for the fairness audit instead of scraping more gym videos?* → Real online fitness videos are heavily skewed toward young, lean, well-lit athletes in fitted wear. ControlNet holds the **exact same ground-truth pose skeleton** constant while isolating one demographic or environmental variable at a time.
- *How do you prevent double-counting reps when landmarks jitter?* → Two layers: (1) One-Euro adaptive low-pass filtering on raw coordinates, and (2) a 4-state hysteresis machine with separate entry/exit angle thresholds and a 600–650 ms minimum rep duration debounce.
