# FormCoach — Presentation Speaker Notes (10 Minutes + Q&A)

**Presentation Deck URL (GitHub Pages):** https://mohitparag.github.io/CVA_Final_Project_Computer_Vision/presentation.html  
**Live App URL:** https://mohitparag.github.io/CVA_Final_Project_Computer_Vision/  
**Shortcuts inside `presentation.html`:** `→` / `←` navigate slides · `S` hide/show notes · `F` fullscreen

---

### Slide 1 — Title & Overview *(~0:45)*
"Good morning/afternoon everyone. Today I am presenting **FormCoach**, my final project for MAIB AI 221 under Project 10: Sports and Wellness. FormCoach is a **100% browser-based AI exercise repetition counter and real-time posture coach** for squats, push-ups, and lunges. Everything runs locally on the user's laptop or smartphone—zero video is ever uploaded, zero paid APIs are used, and it is deployed free on GitHub Pages. First, I will walk through how I built the computer vision pipeline, our trained ONNX model numbers, and our Generative AI fairness audit, and then I will demonstrate the working application live."

---

### Slide 2 — Problem Framing & Error Costs *(~1:00)*
"Why does this problem matter? Millions of people exercise at home without access to a personal trainer or physiotherapist. Poor form—like shallow squats, leaning too far forward, or sagging hips in a push-up—reduces benefit and increases joint injury risk. Cloud fitness apps typically charge subscriptions and stream home webcam video to remote servers, raising privacy concerns.

In FormCoach, I designed around the exact cost of errors:
- A **False Negative** (missing bad form) risks injury.
- A **False Positive** (spurious warnings) frustrates the user.
- **Occlusion or low light** causes hallucinated rep counts—which we explicitly prevent with a **'Not Sure — Low Visibility'** safety gate whenever key landmark confidence drops below 60%."

---

### Slide 3 — Six-Stage Client-Side Architecture *(~1:15)*
"Here is how I built the system end to end, with zero backend server:
1. **Media Input (`js/app.js`):** Captures frames from either a live webcam (with front/rear flip for phones) or uploaded video/image files.
2. **Pose Estimation (`js/poseEngine.js`):** Runs **MediaPipe Pose Landmarker Lite** (Apache 2.0) directly in the browser via WebGL GPU to extract 33 anatomical keypoints.
3. **Smoothing (`js/smoothing.js`):** Applies a **One-Euro adaptive low-pass filter** to eliminate frame-to-frame landmark jitter.
4. **Biomechanics (`js/geometry.js`):** Auto-selects the clearer body side (left or right) and extracts a **10-dimensional view-invariant feature vector**.
5. **Dual AI Engine (`js/exerciseLogic.js` + `js/onnxClassifier.js`):** Runs both our classical **Rule-Based Baseline** and our trained **4.01 KB ONNX neural classifier** via ONNX Runtime Web, coordinated by a 4-state hysteresis rep machine.
6. **Coach & Report (`js/voiceCoach.js`):** Renders color-coded skeletons, speaks real-time cues via the browser's Web Speech API, and exports a printable PDF session report.

Why use a 10-number biomechanical vector for the ONNX model instead of raw pixels? It is view-invariant, tiny (4 KB vs. the 100 MB limit), adds under 1 millisecond of latency alongside MediaPipe, and avoids learning background or demographic shortcuts."

---

### Slide 4 — Classical CV Engineering *(~1:00)*
"Looking closely at our classical CV components:
- **One-Euro Filter:** Raw MediaPipe coordinates jitter by 2–5 degrees. A normal moving average adds lag during fast reps; the One-Euro Filter stays smooth when still and adapts its cutoff frequency when you move fast. Users can toggle it on and off live.
- **Hysteresis Repetition State Machine:** Instead of a single angle threshold that double-counts when you hover near the boundary, we use four states—`READY → DESCENDING → BOTTOM → ASCENDING`—with separate entry/exit thresholds (for example, squat knee `<142°` to descend, `≤108°` for bottom depth, `≥158°` to finish) plus a 600–650 millisecond debounce."

---

### Slide 5 — Baseline vs. Trained ONNX Model Numbers *(~1:15)*
"To compare a classical baseline against a trained ONNX model, I trained a 3-layer Multi-Layer Perceptron (`10 → 32 ReLU → 16 ReLU → 3`) in Python (`ml/train_classifier.py`) on 4,500 noise-perturbed biomechanical pose vectors across three classes: `good_form`, `depth_issue`, and `alignment_issue` (80/20 stratified split).

On the **900 held-out test samples**:
- **Rule-Based Baseline:** **99.33%** accuracy and **0.9933** Macro-F1.
- **Trained ONNX MLP:** **99.22%** accuracy and **0.9922** Macro-F1 (296/300 good form, 298/300 depth issue, 299/300 alignment issue).
- **ONNX Export & Verification:** Exported to ONNX IR v8 at just **4.01 KB**, verified against ONNX Runtime with a maximum probability difference of **$1.19 \times 10^{-7}$** and **0.0012 ms** CPU latency per sample.
In the app, **Hybrid Mode** combines the anatomical specificity of the rules with the calibrated probability bars of the ONNX model."

---

### Slide 6 — Generative AI Innovation: OpenPose ControlNet *(~1:00)*
"For the Generative AI component, we tackled a real fairness gap: online workout datasets mostly show young, lean, light-skinned athletes in tight sportswear under bright lighting. To audit whether FormCoach works equitably across diverse users, I built a pose-conditioned diffusion pipeline on a **free Google Colab T4 GPU** (`notebooks/01_genai_fairness_audit.ipynb`).

We programmatically rendered **4 exact canonical OpenPose skeletons** (good squat, bad-back squat, good push-up, good lunge) with known ground-truth joint angles, then used **Stable Diffusion v1.5 + OpenPose ControlNet** (`lllyasviel/sd-controlnet-openpose`) to hold the exact skeleton constant while varying **11 subgroups across 5 axes**: skin tone, body type, age, fitted vs. baggy clothing, and bright vs. dim cluttered rooms—producing **88 controlled synthetic test images**."

---

### Slide 7 — Real Subgroup Fairness Audit Results *(~1:15)*
"Here are the actual quantitative results from running our exact browser MediaPipe model on all 88 ControlNet images:
1. **Biggest Failure Mode — Baggy Clothing:** Detection rate dropped **25 percentage points** (from 100% down to **75.0%**), mean landmark visibility fell sharply from **0.961 to 0.715**, and form accuracy fell to **50.0%**. This directly validated our fitted-wear camera guide and our `<60%` 'Not Sure' safety gate.
2. **Body Type & Age Parity:** Both `athletic_lean` and `plus_size` achieved **100% detection**, **87.5% form accuracy**, and our lowest angle error (~12° MAE). Older adults (65+) also achieved **87.5%** form accuracy vs. **75.0%** for young adults.
3. **Skin Tone Transparency:** All three skin tones achieved **100% pose detection** and tightly clustered angle error (20.2°–21.6° MAE), though `light_skin` scored **75.0%** form accuracy vs. **62.5%** for `medium_skin` and `dark_skin` (a 1-image difference on $n=8$), which we transparently document in our Model Card for larger real-world validation."

---

### Slide 8 — Responsible AI & Honest Limitations *(~0:45)*
"Privacy and safety are built in by design: zero bytes of video leave the browser, non-medical disclaimers appear on the start screen, Help modal, and PDF report, and all licences are open (MIT, Apache 2.0, Pexels Free). We also honestly document limitations—loose clothing, strict 90° side-view occlusion, and occasional diffusion pose drift—and our final exam-week step is completing ground-truth rep-count labelling on our 20+ collected Pexels workout clips (expanding to 30+)."

---

### Slide 9 — LIVE WORKING APPLICATION DEMO *(~2:00)*
*(Tip: Press `S` to hide speaker notes during the demo, or click **Launch Fullscreen App ↗**)*
1. "Now let's see FormCoach running live right inside the presentation."
2. Click **Upload Video / Image** → choose a squat clip from `formcoach/data/clips/squats/` (e.g., `6892548-uhd_2160_3840_25fps.mp4`).
   - Point out: **Live skeleton overlay**, **joint angle tags**, **FPS & ms latency HUD**, **rep counter incrementing**, and **ONNX Form Classifier probability bars** moving live.
3. Switch Exercise to **Push-up** or **Lunge**, upload a clip from `data/clips/push up/` or `data/clips/lunge/`.
   - Toggle **Form Engine** between *Baseline: Rule-Based* and *Main Model: ONNX Classifier*, and toggle **Voice Cues** ON briefly so the audience hears the spoken feedback.
4. Click **Session Summary & PDF** → show the rep-by-rep breakdown table and click **Export / Print PDF** to show the clean print-ready report.
5. Click **Help & Safety** briefly to show the camera guide, `<60%` Not Sure explanation, and privacy/medical disclaimer.

---

### Slide 10 — Summary & Q&A *(~0:30)*
"To wrap up, FormCoach meets all four MAIB AI 221 learning outcomes, both gate conditions—free live GitHub Pages deployment and a measured Generative AI fairness audit—plus both stretch goals (voice cues and PDF reporting). Thank you, and I'm happy to take any questions!"

---

### Quick Q&A Cheat Sheet (For the 5-Minute Q&A)
- **Q: Why did you choose GitHub Pages instead of Streamlit or Render?**  
  *A:* Privacy and speed. Webcam video stays 100% on the user's device (zero upload), there's no 1-minute cold-start sleep like Render/Streamlit, and WebGL MediaPipe runs at 20+ FPS locally with zero server RAM limits.
- **Q: Why did `bright_clean` have a higher angle MAE (25°) than `dim_cluttered` (14°) in the Colab audit?**  
  *A:* That's an honest limitation of synthetic diffusion auditing: Stable Diffusion v1.5 occasionally drifted slightly from the ControlNet conditioning skeleton on certain bright-room prompts, which is why we pair our synthetic audit with real-world Pexels video evaluation.
- **Q: How does the app handle someone facing left vs. right?**  
  *A:* `pickActiveSide()` in `js/geometry.js` computes mean visibility across left-side vs. right-side joints every frame and automatically evaluates the clearer side, and `kneeOverToeRatio` infers facing direction from the ankle-to-foot-index vector.
