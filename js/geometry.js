/**
 * geometry.js
 * Biomechanical joint-angle calculations and normalized feature extraction
 * from MediaPipe's 33 Pose Landmarks.
 */

export const LM = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32,
};

/**
 * Computes interior angle (in degrees, 0-180) at vertex B formed by A-B-C.
 */
export function angleBetween(a, b, c) {
  const bax = a.x - b.x;
  const bay = a.y - b.y;
  const bcx = c.x - b.x;
  const bcy = c.y - b.y;
  const magBA = Math.hypot(bax, bay);
  const magBC = Math.hypot(bcx, bcy);
  if (magBA < 1e-6 || magBC < 1e-6) return 180;
  let cosTheta = (bax * bcx + bay * bcy) / (magBA * magBC);
  cosTheta = Math.max(-1, Math.min(1, cosTheta));
  return (Math.acos(cosTheta) * 180) / Math.PI;
}

/**
 * Angle of vector (from -> to) relative to upward vertical (0 deg = straight up, 90 deg = horizontal).
 */
export function angleFromVertical(hip, shoulder) {
  const vx = shoulder.x - hip.x;
  const vy = shoulder.y - hip.y; // y increases downward in image coordinates
  const mag = Math.hypot(vx, vy);
  if (mag < 1e-6) return 0;
  // Upward vertical vector is (0, -1)
  let cosTheta = (-vy) / mag;
  cosTheta = Math.max(-1, Math.min(1, cosTheta));
  return (Math.acos(cosTheta) * 180) / Math.PI;
}

/**
 * Selects the side (left or right) with higher average landmark visibility
 * for reliable 2D side/45-degree pose evaluation.
 */
export function pickActiveSide(landmarks) {
  const leftIds = [LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST, LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE];
  const rightIds = [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST, LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE];

  const leftVis = leftIds.reduce((s, id) => s + (landmarks[id]?.visibility ?? 0), 0) / leftIds.length;
  const rightVis = rightIds.reduce((s, id) => s + (landmarks[id]?.visibility ?? 0), 0) / rightIds.length;

  return leftVis >= rightVis
    ? { side: 'left', visibility: leftVis }
    : { side: 'right', visibility: rightVis };
}

/**
 * Extracts all biomechanical angles, visibility confidence, and the 10-dim ONNX feature vector.
 */
export function extractBiomechanics(landmarks, exercise) {
  const { side, visibility } = pickActiveSide(landmarks);
  const isLeft = side === 'left';

  const shoulder = landmarks[isLeft ? LM.LEFT_SHOULDER : LM.RIGHT_SHOULDER];
  const elbow = landmarks[isLeft ? LM.LEFT_ELBOW : LM.RIGHT_ELBOW];
  const wrist = landmarks[isLeft ? LM.LEFT_WRIST : LM.RIGHT_WRIST];
  const hip = landmarks[isLeft ? LM.LEFT_HIP : LM.RIGHT_HIP];
  const knee = landmarks[isLeft ? LM.LEFT_KNEE : LM.RIGHT_KNEE];
  const ankle = landmarks[isLeft ? LM.LEFT_ANKLE : LM.RIGHT_ANKLE];
  const foot = landmarks[isLeft ? LM.LEFT_FOOT_INDEX : LM.RIGHT_FOOT_INDEX];

  const oppHip = landmarks[isLeft ? LM.RIGHT_HIP : LM.LEFT_HIP];
  const oppKnee = landmarks[isLeft ? LM.RIGHT_KNEE : LM.LEFT_KNEE];
  const oppAnkle = landmarks[isLeft ? LM.RIGHT_ANKLE : LM.LEFT_ANKLE];

  // Key visibility check tailored per exercise
  let keyLandmarks = [shoulder, hip, knee, ankle];
  if (exercise === 'pushup') {
    keyLandmarks = [shoulder, elbow, wrist, hip, ankle];
  } else if (exercise === 'lunge') {
    keyLandmarks = [shoulder, hip, knee, ankle, oppKnee, oppAnkle];
  }
  const minKeyVis = Math.min(...keyLandmarks.map((p) => p?.visibility ?? 0));
  const avgKeyVis = keyLandmarks.reduce((s, p) => s + (p?.visibility ?? 0), 0) / keyLandmarks.length;

  // Joint angles
  const kneeAngle = angleBetween(hip, knee, ankle);
  const oppKneeAngle = angleBetween(oppHip, oppKnee, oppAnkle);
  const hipAngle = angleBetween(shoulder, hip, knee);
  const elbowAngle = angleBetween(shoulder, elbow, wrist);
  const torsoTiltDeg = angleFromVertical(hip, shoulder);
  const bodyLineAngle = angleBetween(shoulder, hip, ankle);

  // Facing direction using foot index vs ankle (or nose vs shoulder)
  const facingDir = (foot.x - ankle.x) >= 0 ? 1 : -1;
  const shinLength = Math.max(Math.hypot(knee.x - ankle.x, knee.y - ankle.y), 0.05);
  // Positive when knee extends horizontally beyond toe in facing direction
  const kneeOverToeRatio = ((knee.x - foot.x) * facingDir) / shinLength;

  // For lunges, identify front knee (higher foot forward or smaller horizontal dist to nose)
  const activeKneeAngle = exercise === 'lunge' ? Math.min(kneeAngle, oppKneeAngle) : kneeAngle;
  const backKneeAngle = exercise === 'lunge' ? Math.max(kneeAngle, oppKneeAngle) : kneeAngle;

  // 10-dim feature vector matching ml/train_classifier.py
  const featureVector = new Float32Array([
    exercise === 'squat' ? 1.0 : 0.0,
    exercise === 'pushup' ? 1.0 : 0.0,
    exercise === 'lunge' ? 1.0 : 0.0,
    activeKneeAngle,
    hipAngle,
    elbowAngle,
    torsoTiltDeg,
    bodyLineAngle,
    kneeOverToeRatio,
    backKneeAngle,
  ]);

  return {
    side,
    visibility,
    minKeyVis,
    avgKeyVis,
    joints: { shoulder, elbow, wrist, hip, knee, ankle, foot },
    angles: {
      kneeAngle: activeKneeAngle,
      backKneeAngle,
      hipAngle,
      elbowAngle,
      torsoTiltDeg,
      bodyLineAngle,
      kneeOverToeRatio,
    },
    featureVector,
  };
}
