/**
 * smoothing.js
 * Implements the One-Euro Filter (Casiez et al., CHI 2012) for real-time
 * landmark jitter reduction without introducing noticeable lag during fast reps.
 */

class LowPassFilter {
  constructor(alpha, initval = 0) {
    this.y = initval;
    this.s = initval;
    this.initialized = false;
    this.alpha = alpha;
  }

  filter(value, alpha = this.alpha) {
    if (!this.initialized) {
      this.s = value;
      this.initialized = true;
    } else {
      this.s = alpha * value + (1.0 - alpha) * this.s;
    }
    this.y = value;
    return this.s;
  }

  hasLastRawValue() {
    return this.initialized;
  }

  lastRawValue() {
    return this.y;
  }
}

export class OneEuroFilter {
  constructor(freq = 30, minCutoff = 1.2, beta = 0.02, dCutoff = 1.0) {
    this.freq = freq;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.x = new LowPassFilter(this.alpha(minCutoff));
    this.dx = new LowPassFilter(this.alpha(dCutoff));
    this.lastTime = null;
  }

  alpha(cutoff) {
    const te = 1.0 / this.freq;
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / te);
  }

  filter(value, timestampSec = null) {
    if (this.lastTime !== null && timestampSec !== null && timestampSec > this.lastTime) {
      this.freq = 1.0 / (timestampSec - this.lastTime);
    }
    this.lastTime = timestampSec;

    const dValue = this.x.hasLastRawValue()
      ? (value - this.x.lastRawValue()) * this.freq
      : 0.0;
    const edValue = this.dx.filter(dValue, this.alpha(this.dCutoff));
    const cutoff = this.minCutoff + this.beta * Math.abs(edValue);
    return this.x.filter(value, this.alpha(cutoff));
  }
}

/**
 * Smooths all 33 MediaPipe Pose landmarks (x, y, z) independently.
 */
export class PoseSmoother {
  constructor(numLandmarks = 33, minCutoff = 1.4, beta = 0.03) {
    this.numLandmarks = numLandmarks;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.reset();
  }

  reset() {
    this.filters = Array.from({ length: this.numLandmarks }, () => ({
      x: new OneEuroFilter(30, this.minCutoff, this.beta),
      y: new OneEuroFilter(30, this.minCutoff, this.beta),
      z: new OneEuroFilter(30, this.minCutoff, this.beta),
    }));
  }

  smooth(landmarks, timestampMs) {
    if (!landmarks || landmarks.length === 0) return landmarks;
    const tSec = timestampMs / 1000.0;
    return landmarks.map((lm, i) => {
      if (!this.filters[i]) return lm;
      return {
        x: this.filters[i].x.filter(lm.x, tSec),
        y: this.filters[i].y.filter(lm.y, tSec),
        z: this.filters[i].z.filter(lm.z ?? 0, tSec),
        visibility: lm.visibility ?? 1.0,
      };
    });
  }
}
