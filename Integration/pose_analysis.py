"""
pose_analysis.py
================
Real-time pose estimation using MediaPipe Pose + OpenCV.

Detects two things:
  1. Sitting vs. Standing  – ratio of torso height to total body height
  2. Looking Away          – horizontal nose-to-ear-midpoint offset vs inter-ear span

Bonus:
  • FPS counter
  • Press '+' / '-' to raise / lower the looking-away yaw threshold at runtime
  • Exponential smoothing on confidence scores to kill flicker

Dependencies (see requirements.txt):
  pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import collections
import sys

# ─── MediaPipe setup ───────────────────────────────────────────────────────────
mp_pose   = mp.solutions.pose
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ─── Landmark indices we care about ────────────────────────────────────────────
IDX_NOSE        = 0
IDX_LEFT_EAR    = 7
IDX_RIGHT_EAR   = 8
IDX_LEFT_SHOULDER  = 11
IDX_RIGHT_SHOULDER = 12
IDX_LEFT_HIP    = 23
IDX_RIGHT_HIP   = 24
IDX_LEFT_ANKLE  = 27
IDX_RIGHT_ANKLE = 28

# ─── Thresholds (tunable at runtime via +/-) ───────────────────────────────────
# Sitting/Standing:
#   ratio = |shoulder_mid_y - hip_mid_y| / |shoulder_mid_y - ankle_mid_y|
#   If ratio < SIT_RATIO_THRESHOLD → sitting (torso takes up little of body)
SIT_RATIO_THRESHOLD = 0.40   # empirically chosen; lower = easier to classify as sitting

# Looking Away:
#   offset_ratio = |nose_x - ear_mid_x| / inter_ear_distance
#   Values above LOOK_AWAY_THRESHOLD indicate significant yaw
LOOK_AWAY_THRESHOLD = 0.30   # ~30 % of inter-ear span; increase → less sensitive

# Minimum landmark visibility score to trust a landmark
MIN_VISIBILITY = 0.50

# Exponential smoothing factor (0 = no smoothing, 1 = no update)
ALPHA = 0.25   # new_smooth = ALPHA * raw + (1-ALPHA) * old_smooth

# Threshold on smoothed confidence to flip the label
DECISION_THRESHOLD = 0.50

# ─── Smoothing state ───────────────────────────────────────────────────────────
# Store smoothed confidence that a person is SITTING (0=standing, 1=sitting)
smooth_sit_conf    = 0.5
# Store smoothed confidence that person is LOOKING AWAY (0=at camera, 1=away)
smooth_look_conf   = 0.5


def visible(lm, min_vis=MIN_VISIBILITY):
    """Return True if a single landmark has sufficient visibility."""
    return lm.visibility >= min_vis


def midpoint_2d(a, b):
    """Return (x, y) midpoint of two landmarks (normalised coords)."""
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def detect_sitting_standing(landmarks):
    """
    Classify the pose as 'Sitting' or 'Standing'.

    Strategy
    --------
    We compare the vertical span of the torso (shoulders → hips) against the
    vertical span of the whole body (shoulders → ankles).

        ratio = (hip_y - shoulder_y) / (ankle_y - shoulder_y)

    In image coordinates, y increases downward, so a standing person has a
    large ankle_y and a moderate hip_y, giving a ratio around 0.45–0.55.
    When someone sits, the ankles are often out of frame or close to the hips,
    making the ratio small.  Values below SIT_RATIO_THRESHOLD → Sitting.

    Returns
    -------
    (label: str, sit_confidence: float)
        sit_confidence in [0, 1]; higher = more likely sitting.
    """
    lm = landmarks.landmark

    # Check visibility of key joints
    required = [IDX_LEFT_SHOULDER, IDX_RIGHT_SHOULDER,
                IDX_LEFT_HIP,      IDX_RIGHT_HIP]
    if not all(visible(lm[i]) for i in required):
        return "Unknown", 0.5

    # Shoulder midpoint (y is the vertical component in normalised coords)
    _, sh_y = midpoint_2d(lm[IDX_LEFT_SHOULDER], lm[IDX_RIGHT_SHOULDER])
    _, hip_y = midpoint_2d(lm[IDX_LEFT_HIP],     lm[IDX_RIGHT_HIP])

    # Try to use ankles; fall back to hips + fixed offset if not visible
    ankles_ok = (visible(lm[IDX_LEFT_ANKLE]) and visible(lm[IDX_RIGHT_ANKLE]))
    if ankles_ok:
        _, ank_y = midpoint_2d(lm[IDX_LEFT_ANKLE], lm[IDX_RIGHT_ANKLE])
    else:
        # Estimate ankle position as 2× the torso length below the shoulder
        ank_y = sh_y + 2.0 * (hip_y - sh_y)

    torso_span = hip_y - sh_y          # positive (hips are below shoulders)
    body_span  = ank_y - sh_y          # positive (ankles are below shoulders)

    if body_span < 1e-4:               # degenerate frame
        return "Unknown", 0.5

    ratio = torso_span / body_span
    # ratio ≈ 0.45–0.60 when standing, ≈ 0.60–0.90 when sitting
    # (sitting compresses the visible body height or raises the ankles)
    # Map ratio to a sit_confidence: linearly scale around the threshold
    sit_conf = np.clip((ratio - SIT_RATIO_THRESHOLD) /
                       (1.0 - SIT_RATIO_THRESHOLD), 0.0, 1.0)

    label = "Sitting" if sit_conf >= DECISION_THRESHOLD else "Standing"
    return label, float(sit_conf)


def detect_looking_away(landmarks):
    """
    Detect whether the person is looking toward the camera or away.

    Strategy
    --------
    Use the 2-D (x, y) positions of:
      • Nose       (landmark 0)
      • Left ear   (landmark 7)
      • Right ear  (landmark 8)

    When facing the camera both ears are roughly equidistant from the nose
    in the x-direction.  When the person yaws (turns their head), the nose
    moves closer to the ear in the direction they are turning, and the
    ear-midpoint shifts away from the nose.

    We compute:
        offset_ratio = |nose_x - ear_mid_x| / inter_ear_distance

    This value is ~0 when facing forward and rises toward 0.5 as the head
    turns 90°.  We flag "Looking away" when offset_ratio > LOOK_AWAY_THRESHOLD.

    Returns
    -------
    (label: str, look_away_confidence: float)
        look_away_confidence in [0, 1]; higher = more likely looking away.
    """
    lm = landmarks.landmark

    # Need nose and at least one ear; prefer both ears for accuracy
    if not visible(lm[IDX_NOSE]):
        return "Unknown", 0.5

    nose_x = lm[IDX_NOSE].x

    left_ear_ok  = visible(lm[IDX_LEFT_EAR])
    right_ear_ok = visible(lm[IDX_RIGHT_EAR])

    if left_ear_ok and right_ear_ok:
        ear_mid_x      = (lm[IDX_LEFT_EAR].x + lm[IDX_RIGHT_EAR].x) / 2.0
        inter_ear_dist = abs(lm[IDX_LEFT_EAR].x - lm[IDX_RIGHT_EAR].x)
    elif left_ear_ok:
        # Only left ear visible → person likely looking right (away from us)
        ear_mid_x      = lm[IDX_LEFT_EAR].x
        inter_ear_dist = abs(nose_x - lm[IDX_LEFT_EAR].x) * 1.5  # estimate
    elif right_ear_ok:
        ear_mid_x      = lm[IDX_RIGHT_EAR].x
        inter_ear_dist = abs(nose_x - lm[IDX_RIGHT_EAR].x) * 1.5
    else:
        return "Unknown", 0.5

    if inter_ear_dist < 1e-4:
        return "Unknown", 0.5

    offset_ratio = abs(nose_x - ear_mid_x) / inter_ear_dist

    # Map offset_ratio to look_away_confidence
    look_conf = np.clip((offset_ratio - LOOK_AWAY_THRESHOLD) /
                        (0.50 - LOOK_AWAY_THRESHOLD), 0.0, 1.0)

    label = "Looking away" if look_conf >= DECISION_THRESHOLD else "Looking at camera"
    return label, float(look_conf)


def smooth_update(smooth_val, raw_conf):
    """
    Exponential moving average:
        new = ALPHA * raw + (1-ALPHA) * old
    Returns updated smoothed value.
    """
    return ALPHA * raw_conf + (1.0 - ALPHA) * smooth_val


def draw_confidence_bar(frame, x, y, width, height, value, label, color_high, color_low):
    """
    Draw a small horizontal bar representing a confidence value [0,1].
    color_high is drawn on the 'high confidence' side.
    """
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    fill = int(value * width)
    color = color_high if value >= DECISION_THRESHOLD else color_low
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (180, 180, 180), 1)
    cv2.putText(frame, label, (x + width + 6, y + height - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)


def draw_overlay(frame, sit_label, sit_conf, look_label, look_conf,
                 person_detected, fps):
    """Render the status panel onto the frame (top-left)."""
    h, w = frame.shape[:2]

    # ── Background panel ──────────────────────────────────────────────────────
    panel_w, panel_h = 360, 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if not person_detected:
        cv2.putText(frame, "Person not fully detected",
                    (18, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (50, 180, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (18, 126), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (150, 150, 150), 1, cv2.LINE_AA)
        return

    # ── Posture label ─────────────────────────────────────────────────────────
    sit_color   = (60, 200, 80)  if sit_label  == "Standing"        else (50, 120, 255)
    look_color  = (60, 200, 80)  if look_label == "Looking at camera" else (50, 80,  255)

    status_line = f"{sit_label}  |  {look_label}"
    cv2.putText(frame, "Status:", (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, status_line, (18, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 230, 230), 2, cv2.LINE_AA)

    # ── Confidence bars ───────────────────────────────────────────────────────
    # Sitting confidence bar  (high = sitting)
    draw_confidence_bar(frame, 18, 68, 160, 14,
                        sit_conf,   "Sit conf",
                        (50, 120, 255), (60, 200, 80))
    # Looking-away confidence bar  (high = looking away)
    draw_confidence_bar(frame, 18, 88, 160, 14,
                        look_conf,  "Look-away conf",
                        (50,  80, 255), (60, 200, 80))

    # ── Threshold hints ───────────────────────────────────────────────────────
    cv2.putText(frame,
                f"Thresholds  sit={SIT_RATIO_THRESHOLD:.2f}  yaw={LOOK_AWAY_THRESHOLD:.2f}",
                (18, 114),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1, cv2.LINE_AA)

    # ── FPS ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (18, 130), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (130, 130, 130), 1, cv2.LINE_AA)

    # ── Corner key hint ───────────────────────────────────────────────────────
    cv2.putText(frame, "+/- : adjust look-away sens   q : quit",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 100), 1, cv2.LINE_AA)


# ─── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global smooth_sit_conf, smooth_look_conf
    global LOOK_AWAY_THRESHOLD, SIT_RATIO_THRESHOLD

    # Accept an optional video path as argv[1]; default to webcam (0)
    source = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() \
             else (sys.argv[1] if len(sys.argv) > 1 else 0)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    # FPS tracking
    prev_time = time.time()
    fps = 0.0

    with mp_pose.Pose(
        model_complexity        = 1,          # 0=Lite, 1=Full, 2=Heavy
        smooth_landmarks        = True,        # MediaPipe's own temporal smoothing
        enable_segmentation     = False,
        min_detection_confidence= 0.55,
        min_tracking_confidence = 0.55,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream or cannot read frame.")
                break

            # Mirror webcam for natural feel
            frame = cv2.flip(frame, 1)

            # ── FPS ───────────────────────────────────────────────────────────
            now      = time.time()
            fps      = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # ── MediaPipe inference ───────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            person_detected = results.pose_landmarks is not None

            if person_detected:
                # Draw skeleton
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

                # ── Sitting / Standing ────────────────────────────────────────
                sit_label_raw, sit_conf_raw = detect_sitting_standing(
                    results.pose_landmarks)

                # ── Looking Away ──────────────────────────────────────────────
                look_label_raw, look_conf_raw = detect_looking_away(
                    results.pose_landmarks)

                # ── Temporal smoothing ────────────────────────────────────────
                if sit_conf_raw != 0.5:    # only update when we got a real read
                    smooth_sit_conf  = smooth_update(smooth_sit_conf,  sit_conf_raw)
                if look_conf_raw != 0.5:
                    smooth_look_conf = smooth_update(smooth_look_conf, look_conf_raw)

                # Final labels come from the smoothed confidence
                sit_label  = "Sitting"   if smooth_sit_conf  >= DECISION_THRESHOLD \
                             else "Standing"
                look_label = "Looking away" if smooth_look_conf >= DECISION_THRESHOLD \
                             else "Looking at camera"
            else:
                sit_label  = "Unknown"
                look_label = "Unknown"

            # ── HUD overlay ───────────────────────────────────────────────────
            draw_overlay(frame,
                         sit_label,  smooth_sit_conf,
                         look_label, smooth_look_conf,
                         person_detected, fps)

            cv2.imshow("Pose Analysis  (q=quit  +/-=sensitivity)", frame)

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                LOOK_AWAY_THRESHOLD = min(LOOK_AWAY_THRESHOLD + 0.02, 0.70)
                print(f"[INFO] Look-away threshold → {LOOK_AWAY_THRESHOLD:.2f}")
            elif key == ord('-') or key == ord('_'):
                LOOK_AWAY_THRESHOLD = max(LOOK_AWAY_THRESHOLD - 0.02, 0.05)
                print(f"[INFO] Look-away threshold → {LOOK_AWAY_THRESHOLD:.2f}")
            elif key == ord(']'):
                SIT_RATIO_THRESHOLD = min(SIT_RATIO_THRESHOLD + 0.02, 0.80)
                print(f"[INFO] Sit ratio threshold → {SIT_RATIO_THRESHOLD:.2f}")
            elif key == ord('['):
                SIT_RATIO_THRESHOLD = max(SIT_RATIO_THRESHOLD - 0.02, 0.10)
                print(f"[INFO] Sit ratio threshold → {SIT_RATIO_THRESHOLD:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
