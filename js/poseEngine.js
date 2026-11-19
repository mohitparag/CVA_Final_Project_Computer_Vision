/**
 * poseEngine.js
 * Wrapper around MediaPipe Pose Landmarker (Tasks for Web, Apache 2.0).
 * Runs 100% locally in the browser using WebGL GPU (with WASM CPU fallback).
 */

import {
  FilesetResolver,
  PoseLandmarker,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm';

const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

export const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Upper body
  [11, 23], [12, 24], [23, 24],                     // Torso
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31], // Left leg
  [24, 26], [26, 28], [28, 30], [30, 32], [28, 32], // Right leg
];

export class PoseEngine {
  constructor() {
    this.landmarker = null;
    this.runningMode = 'VIDEO';
    this.ready = false;
  }

  async init(mode = 'VIDEO') {
    this.runningMode = mode;
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

    try {
      this.landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: 'GPU',
        },
        runningMode: mode,
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    } catch (gpuErr) {
      console.warn('[PoseEngine] GPU delegate unavailable, falling back to CPU:', gpuErr);
      this.landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: 'CPU',
        },
        runningMode: mode,
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    }

    this.ready = true;
    return true;
  }

  async setMode(mode) {
    if (!this.landmarker || this.runningMode === mode) return;
    await this.landmarker.setOptions({ runningMode: mode });
    this.runningMode = mode;
  }

  detectVideo(videoEl, timestampMs) {
    if (!this.ready || !this.landmarker) return null;
    const res = this.landmarker.detectForVideo(videoEl, timestampMs);
    return res?.landmarks?.[0] || null;
  }

  detectImage(imgEl) {
    if (!this.ready || !this.landmarker) return null;
    const res = this.landmarker.detect(imgEl);
    return res?.landmarks?.[0] || null;
  }
}
