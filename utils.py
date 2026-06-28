import numpy as np

def calculate_ear(landmarks, eye_indices) -> float:
    try:
        p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * h + 1e-6)
    except Exception:
        return 0.3

def estimate_head_pose(landmarks, img_w, img_h):
    try:
        # Landmarker indices are stable across tasks versions
        nose = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        n_p = np.array([nose.x * img_w, nose.y * img_h])
        l_p = np.array([left_eye.x * img_w, left_eye.y * img_h])
        r_p = np.array([right_eye.x * img_w, right_eye.y * img_h])
        
        eye_center = (l_p + r_p) / 2.0
        dist_ratio = np.linalg.norm(n_p - eye_center) / (np.linalg.norm(l_p - r_p) + 1e-6)
        
        if dist_ratio > 0.65 or abs(l_p[1] - r_p[1]) > 25:
            return True
        return False
    except Exception:
        return False

def compute_attention_score(metrics: dict) -> float:
    score = 100.0
    if metrics.get("Phone Usage"): score -= 40.0
    if metrics.get("Drowsy"): score -= 30.0
    if metrics.get("Looking Away"): score -= 20.0
    if metrics.get("Laptop Usage"): score -= 5.0
    if metrics.get("Reading") or metrics.get("Writing"): score += 10.0
    return float(max(0.0, min(100.0, score)))