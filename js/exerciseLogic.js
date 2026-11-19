/**
 * exerciseLogic.js
 * Implements:
 *  1. Hysteresis repetition state machine for Squat, Push-up, and Lunge.
 *  2. Rule-based biomechanical posture checker (Baseline model).
 *  3. Session analytics (rep log, good-form %, most common mistake).
 */

export const VISIBILITY_THRESHOLD = 0.60;

export const EXERCISE_CONFIG = {
  squat: {
    name: 'Squat',
    primaryAngleKey: 'kneeAngle',
    upThreshold: 158,       // Standing tall
    descendThreshold: 142,  // Began lowering
    bottomThreshold: 108,   // Reached valid bottom depth (or attempted depth)
    shallowThreshold: 112,  // If min knee angle > 112 at turnaround -> "Go lower"
    minRepDurationMs: 650,
  },
  pushup: {
    name: 'Push-up',
    primaryAngleKey: 'elbowAngle',
    upThreshold: 155,
    descendThreshold: 140,
    bottomThreshold: 105,
    shallowThreshold: 108,
    minRepDurationMs: 600,
  },
  lunge: {
    name: 'Lunge',
    primaryAngleKey: 'kneeAngle',
    upThreshold: 156,
    descendThreshold: 140,
    bottomThreshold: 108,
    shallowThreshold: 112,
    minRepDurationMs: 650,
  },
};

/**
 * Rule-based posture evaluation for a single frame (Baseline CV method).
 * Returns array of detected mistakes with inclusive coaching cues.
 */
export function evaluateRulesFrame(exercise, angles) {
  const issues = [];

  if (exercise === 'squat') {
    if (angles.torsoTiltDeg > 42) {
      issues.push({
        code: 'back_not_straight',
        label: 'Back not straight',
        cue: 'Keep your chest lifted and back straighter',
        severity: 'bad',
        joints: ['shoulder', 'hip'],
      });
    }
    if (angles.kneeOverToeRatio > 0.35 && angles.kneeAngle < 135) {
      issues.push({
        code: 'knees_past_toes',
        label: 'Knees past toes',
        cue: 'Shift your hips back — knees are tracking past toes',
        severity: 'warn',
        joints: ['knee', 'ankle'],
      });
    }
    if (angles.kneeAngle < 140 && angles.kneeAngle > 112) {
      issues.push({
        code: 'go_lower',
        label: 'Go lower',
        cue: 'Try lowering your hips a little deeper',
        severity: 'warn',
        joints: ['hip', 'knee'],
      });
    }
  } else if (exercise === 'pushup') {
    if (angles.bodyLineAngle < 156) {
      issues.push({
        code: 'hips_sagging',
        label: 'Back / hips not aligned',
        cue: 'Brace your core to keep a straight line from shoulders to ankles',
        severity: 'bad',
        joints: ['shoulder', 'hip', 'ankle'],
      });
    }
    if (angles.elbowAngle < 138 && angles.elbowAngle > 108) {
      issues.push({
        code: 'go_lower',
        label: 'Go lower',
        cue: 'Lower your chest a bit closer to the floor',
        severity: 'warn',
        joints: ['shoulder', 'elbow'],
      });
    }
  } else if (exercise === 'lunge') {
    if (angles.torsoTiltDeg > 26) {
      issues.push({
        code: 'back_not_straight',
        label: 'Torso leaning forward',
        cue: 'Keep your torso upright and shoulders stacked over hips',
        severity: 'bad',
        joints: ['shoulder', 'hip'],
      });
    }
    if (angles.kneeOverToeRatio > 0.32 && angles.kneeAngle < 135) {
      issues.push({
        code: 'knees_past_toes',
        label: 'Front knee past toes',
        cue: 'Step slightly wider so your front knee stays over your ankle',
        severity: 'warn',
        joints: ['knee', 'ankle'],
      });
    }
    if (angles.kneeAngle < 138 && angles.kneeAngle > 112) {
      issues.push({
        code: 'go_lower',
        label: 'Go lower',
        cue: 'Drop your back knee a little lower toward the floor',
        severity: 'warn',
        joints: ['hip', 'knee'],
      });
    }
  }

  return issues;
}

export class ExerciseTracker {
  constructor(exercise = 'squat') {
    this.setExercise(exercise);
  }

  setExercise(exercise) {
    this.exercise = exercise;
    this.config = EXERCISE_CONFIG[exercise] || EXERCISE_CONFIG.squat;
    this.reset();
  }

  reset() {
    this.phase = 'READY'; // READY | DESCENDING | BOTTOM | ASCENDING
    this.reps = 0;
    this.repHistory = [];
    this.mistakeCounts = {};
    this.repStartTime = null;
    this.minPrimaryAngleInRep = 180;
    this.activeRepMistakes = new Map(); // code -> label
    this.lastCompletedRepTime = 0;
    this.onnxProbsDuringBottom = null;
  }

  /**
   * Process one video frame's biomechanics + optional ONNX prediction.
   * engineMode: 'hybrid' | 'rules' | 'onnx'
   */
  update(bio, timestampMs, onnxResult = null, engineMode = 'hybrid') {
    const cfg = this.config;
    const primaryAngle = bio.angles[cfg.primaryAngleKey];

    // 1. Visibility / "Not Sure" gating
    if (bio.avgKeyVis < VISIBILITY_THRESHOLD || bio.minKeyVis < 0.42) {
      return {
        isNotSure: true,
        notSureReason: 'Key joints are partially hidden or dimly lit. Step back into full frame.',
        phase: this.phase,
        cue: { text: 'Waiting for clear full-body view...', level: 'neutral' },
        highlightJoints: [],
        repCompleted: null,
      };
    }

    // 2. Frame-level posture checks (Rules vs ONNX vs Hybrid)
    const ruleIssues = evaluateRulesFrame(this.exercise, bio.angles);
    const frameIssues = this._combineIssues(ruleIssues, onnxResult, engineMode, primaryAngle);

    // 3. Hysteresis Repetition State Machine
    let repCompleted = null;

    if (this.phase === 'READY') {
      if (primaryAngle < cfg.descendThreshold) {
        this.phase = 'DESCENDING';
        this.repStartTime = timestampMs;
        this.minPrimaryAngleInRep = primaryAngle;
        this.activeRepMistakes.clear();
        this.onnxProbsDuringBottom = null;
      }
    } else if (this.phase === 'DESCENDING') {
      if (primaryAngle < this.minPrimaryAngleInRep) {
        this.minPrimaryAngleInRep = primaryAngle;
        if (onnxResult) this.onnxProbsDuringBottom = onnxResult.probs;
      }

      // Record alignment mistakes during descent
      for (const iss of frameIssues) {
        if (iss.code !== 'go_lower') {
          this.activeRepMistakes.set(iss.code, iss.label);
        }
      }

      if (primaryAngle <= cfg.bottomThreshold) {
        this.phase = 'BOTTOM';
      } else if (primaryAngle > this.minPrimaryAngleInRep + 16 && this.minPrimaryAngleInRep < cfg.descendThreshold - 8) {
        // User reversed early without reaching full depth!
        this.phase = 'ASCENDING';
      }
    } else if (this.phase === 'BOTTOM') {
      if (primaryAngle < this.minPrimaryAngleInRep) {
        this.minPrimaryAngleInRep = primaryAngle;
        if (onnxResult) this.onnxProbsDuringBottom = onnxResult.probs;
      }
      for (const iss of frameIssues) {
        if (iss.code !== 'go_lower') {
          this.activeRepMistakes.set(iss.code, iss.label);
        }
      }
      if (primaryAngle > this.minPrimaryAngleInRep + 15) {
        this.phase = 'ASCENDING';
      }
    } else if (this.phase === 'ASCENDING') {
      for (const iss of frameIssues) {
        if (iss.code !== 'go_lower') {
          this.activeRepMistakes.set(iss.code, iss.label);
        }
      }

      if (primaryAngle >= cfg.upThreshold) {
        const duration = timestampMs - (this.repStartTime ?? timestampMs);
        if (duration >= cfg.minRepDurationMs && (timestampMs - this.lastCompletedRepTime) >= 400) {
          repCompleted = this._finalizeRep(bio.avgKeyVis, engineMode);
          this.lastCompletedRepTime = timestampMs;
        }
        this.phase = 'READY';
        this.minPrimaryAngleInRep = 180;
        this.activeRepMistakes.clear();
      }
    }

    // 4. Real-time coaching cue selection
    const cue = this._pickRealtimeCue(frameIssues, repCompleted);
    const highlightJoints = Array.from(new Set(frameIssues.flatMap((i) => i.joints || [])));

    return {
      isNotSure: false,
      notSureReason: '',
      phase: this.phase,
      cue,
      highlightJoints,
      repCompleted,
    };
  }

  /**
   * Evaluates a single static image (no rep cycle required).
   */
  evaluateStaticImage(bio, onnxResult = null, engineMode = 'hybrid') {
    if (bio.avgKeyVis < VISIBILITY_THRESHOLD) {
      return {
        isNotSure: true,
        notSureReason: 'Low pose visibility on this image (<60%).',
        phase: 'STATIC',
        cue: { text: 'Not sure — pose landmarks unclear', level: 'warn' },
        highlightJoints: [],
      };
    }
    const primaryAngle = bio.angles[this.config.primaryAngleKey];
    const ruleIssues = evaluateRulesFrame(this.exercise, bio.angles);
    const issues = this._combineIssues(ruleIssues, onnxResult, engineMode, primaryAngle);
    const highlightJoints = Array.from(new Set(issues.flatMap((i) => i.joints || [])));

    if (issues.length === 0) {
      return {
        isNotSure: false,
        phase: 'STATIC',
        cue: { text: 'Good posture detected in this frame!', level: 'good' },
        highlightJoints: [],
      };
    }
    const top = issues[0];
    return {
      isNotSure: false,
      phase: 'STATIC',
      cue: { text: top.cue, level: top.severity },
      highlightJoints,
    };
  }

  _combineIssues(ruleIssues, onnxResult, engineMode, primaryAngle) {
    if (engineMode === 'rules' || !onnxResult) {
      return ruleIssues;
    }

    const onnxIssues = [];
    // ONNX class indices: 0 = Good Form, 1 = Depth / Range Issue, 2 = Alignment / Back Issue
    const [pGood, pDepth, pAlign] = onnxResult.probs;
    const isWorking = primaryAngle < this.config.descendThreshold;

    if (isWorking && pAlign > 0.52) {
      onnxIssues.push({
        code: 'ml_alignment',
        label: 'Alignment / Back posture',
        cue: 'ONNX Coach: Keep your back straight and core aligned',
        severity: 'bad',
        joints: ['shoulder', 'hip'],
      });
    }
    if (isWorking && pDepth > 0.55 && pGood < 0.40) {
      onnxIssues.push({
        code: 'go_lower',
        label: 'Go lower',
        cue: 'ONNX Coach: Go a bit deeper for full range of motion',
        severity: 'warn',
        joints: ['hip', 'knee'],
      });
    }

    if (engineMode === 'onnx') {
      return onnxIssues;
    }

    // Hybrid mode: use rule-based specificity + ONNX confirmation/catch
    const merged = [...ruleIssues];
    const codes = new Set(merged.map((i) => i.code));
    for (const oi of onnxIssues) {
      if (oi.code === 'ml_alignment' && !codes.has('back_not_straight') && !codes.has('hips_sagging') && !codes.has('knees_past_toes')) {
        merged.push(oi);
      } else if (oi.code === 'go_lower' && !codes.has('go_lower')) {
        merged.push(oi);
      }
    }
    return merged;
  }

  _finalizeRep(avgVis, engineMode) {
    this.reps += 1;

    // Depth check at rep completion
    if (this.minPrimaryAngleInRep > this.config.shallowThreshold) {
      if (engineMode !== 'onnx') {
        this.activeRepMistakes.set('go_lower', 'Go lower');
      } else if (this.onnxProbsDuringBottom && this.onnxProbsDuringBottom[1] > 0.45) {
        this.activeRepMistakes.set('go_lower', 'Go lower');
      }
    }

    const mistakes = Array.from(this.activeRepMistakes.values());
    const isGood = mistakes.length === 0;

    for (const m of mistakes) {
      this.mistakeCounts[m] = (this.mistakeCounts[m] || 0) + 1;
    }

    const repRecord = {
      repNumber: this.reps,
      isGood,
      minAngle: Math.round(this.minPrimaryAngleInRep),
      mistakes,
      feedback: isGood ? 'Good form' : mistakes.join(', '),
      confidence: Math.round(avgVis * 100),
    };

    this.repHistory.push(repRecord);
    return repRecord;
  }

  _pickRealtimeCue(frameIssues, repCompleted) {
    if (repCompleted) {
      if (repCompleted.isGood) {
        return { text: `Rep ${repCompleted.repNumber} — Great form!`, level: 'good', speak: `Rep ${repCompleted.repNumber}, great form` };
      }
      return {
        text: `Rep ${repCompleted.repNumber} — ${repCompleted.mistakes[0]}`,
        level: 'warn',
        speak: repCompleted.mistakes[0],
      };
    }

    if (frameIssues.length > 0) {
      // Prioritize severe alignment issues over depth
      const severe = frameIssues.find((i) => i.severity === 'bad') || frameIssues[0];
      return { text: severe.cue, level: severe.severity, speak: severe.label };
    }

    if (this.phase === 'BOTTOM') {
      return { text: 'Good depth — drive smoothly back up!', level: 'good', speak: null };
    }
    if (this.phase === 'DESCENDING' || this.phase === 'ASCENDING') {
      return { text: 'Controlled movement — looking good!', level: 'good', speak: null };
    }
    return { text: `Ready for ${this.config.name}s — begin whenever you're ready`, level: 'neutral', speak: null };
  }

  getSummary() {
    const total = this.repHistory.length;
    const goodCount = this.repHistory.filter((r) => r.isGood).length;
    const goodPct = total > 0 ? Math.round((goodCount / total) * 100) : 0;

    let topMistake = 'None';
    let maxCount = 0;
    for (const [label, count] of Object.entries(this.mistakeCounts)) {
      if (count > maxCount) {
        maxCount = count;
        topMistake = `${label} (${count}x)`;
      }
    }

    return {
      exerciseName: this.config.name,
      totalReps: total,
      goodReps: goodCount,
      goodPct,
      topMistake,
      history: this.repHistory,
    };
  }
}
