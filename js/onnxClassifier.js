/**
 * onnxClassifier.js
 * Loads and executes the student-trained ONNX pose-form classifier
 * directly in the user's browser via ONNX Runtime Web (WASM).
 */

export class OnnxFormClassifier {
  constructor() {
    this.session = null;
    this.metadata = null;
    this.ready = false;
    this.error = null;
  }

  async init(modelUrl = './models/form_classifier.onnx', metaUrl = './models/model_metadata.json') {
    try {
      if (typeof window.ort === 'undefined') {
        throw new Error('ONNX Runtime Web (ort) not loaded.');
      }

      // Configure WASM paths cleanly from CDN
      window.ort.env.wasm.numThreads = 1;

      const [metaResp, session] = await Promise.all([
        fetch(metaUrl).then((r) => {
          if (!r.ok) throw new Error(`Failed to fetch ${metaUrl}`);
          return r.json();
        }),
        window.ort.InferenceSession.create(modelUrl, {
          executionProviders: ['wasm'],
        }),
      ]);

      this.metadata = metaResp;
      this.session = session;
      this.ready = true;
      return true;
    } catch (err) {
      console.warn('[OnnxFormClassifier] Init failed, falling back to rules:', err);
      this.error = err.message;
      this.ready = false;
      return false;
    }
  }

  /**
   * Runs inference on a 10-dim Float32Array biomechanical feature vector.
   * Returns { probs: [pGood, pDepth, pAlign], predictedClass, confidence, latencyMs }
   */
  async predict(rawFeatures) {
    if (!this.ready || !this.session || !this.metadata) return null;

    const t0 = performance.now();
    const mean = this.metadata.feature_mean;
    const std = this.metadata.feature_std;
    const norm = new Float32Array(rawFeatures.length);
    for (let i = 0; i < rawFeatures.length; i++) {
      const s = std[i] > 1e-6 ? std[i] : 1.0;
      norm[i] = (rawFeatures[i] - mean[i]) / s;
    }

    const inputName = this.session.inputNames[0];
    const tensor = new window.ort.Tensor('float32', norm, [1, norm.length]);
    const results = await this.session.run({ [inputName]: tensor });
    const outputName = this.session.outputNames[0];
    const logits = Array.from(results[outputName].data);

    // Numerically stable softmax
    const maxLogit = Math.max(...logits);
    const exps = logits.map((v) => Math.exp(v - maxLogit));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((e) => e / sumExp);

    let bestIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }

    return {
      probs,
      predictedIdx: bestIdx,
      predictedClass: this.metadata.classes[bestIdx] || `class_${bestIdx}`,
      confidence: probs[bestIdx],
      latencyMs: performance.now() - t0,
    };
  }
}
