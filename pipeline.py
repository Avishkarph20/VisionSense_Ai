import cv2
import os
import urllib.request
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import calculate_ear, estimate_head_pose, compute_attention_score
from database import log_behaviour

class ClassroomMonitorPipeline:
    def __init__(self):
        # Auto-download models to make this completely self-contained
        self.face_model_path = "face_landmarker.task"
        self.pose_model_path = "pose_landmarker.task"
        self._download_assets()

        # Initialize modern YOLO11
        self.model = YOLO("yolo11n.pt")
        self.target_classes = [0, 63, 67] # person, laptop, cell phone
        
        # Build Modern MediaPipe Tasks API Options
        base_face_options = python.BaseOptions(model_asset_path=self.face_model_path)
        face_options = vision.FaceLandmarkerOptions(base_options=base_face_options, output_face_blendshapes=False, num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        
        base_pose_options = python.BaseOptions(model_asset_path=self.pose_model_path)
        pose_options = vision.PoseLandmarkerOptions(base_options=base_pose_options, output_segmentation_masks=False)
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # Eye landmark tracking indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def _download_assets(self):
        urls = {
            self.face_model_path: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            self.pose_model_path: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
        }
        for path, url in urls.items():
            if not os.path.exists(path):
                print(f"Downloading required engine asset: {path}...")
                urllib.request.urlretrieve(url, path)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w, _ = frame.shape
        results = self.model.track(source=frame, persist=True, classes=self.target_classes, tracker="bytetrack.yaml", verbose=False)
        
        students = {}
        objects = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                track_id = int(box.id[0].item()) if box.id is not None else None
                
                if cls_id == 0 and track_id is not None:
                    students[track_id] = xyxy
                elif cls_id in [63, 67]:
                    objects.append({"type": "laptop" if cls_id == 63 else "phone", "box": xyxy})

        for student_id, bbox in students.items():
            sx1, sy1, sx2, sy2 = bbox
            sx1, sy1, sx2, sy2 = max(0, sx1), max(0, sy1), min(w, sx2), min(h, sy2)
            
            crop = frame[sy1:sy2, sx1:sx2]
            if crop.size == 0: continue
                
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            ch, cw, _ = crop.shape

            behaviour_flags = {"Drowsy": False, "Looking Away": False, "Reading": False, "Writing": False, "Laptop Usage": False, "Phone Usage": False}

            for obj in objects:
                ox1, oy1, ox2, oy2 = obj["box"]
                ix1, iy1 = max(sx1, ox1), max(sy1, oy1)
                ix2, iy2 = min(sx2, ox2), min(sy2, oy2)
                if ix2 > ix1 and iy2 > iy1:
                    if obj["type"] == "phone": behaviour_flags["Phone Usage"] = True
                    elif obj["type"] == "laptop": behaviour_flags["Laptop Usage"] = True

            # 1. New Tasks API Face Landmark processing
            face_result = self.face_landmarker.detect(mp_image)
            if face_result.face_landmarks:
                landmarks = face_result.face_landmarks[0]
                left_ear = calculate_ear(landmarks, self.LEFT_EYE)
                right_ear = calculate_ear(landmarks, self.RIGHT_EYE)
                if (left_ear + right_ear) / 2.0 < 0.21:
                    behaviour_flags["Drowsy"] = True
                if estimate_head_pose(landmarks, cw, ch):
                    behaviour_flags["Looking Away"] = True

            # 2. New Tasks API Pose Landmark processing
            pose_result = self.pose_landmarker.detect(mp_image)
            if pose_result.pose_landmarks:
                p_landmarks = pose_result.pose_landmarks[0]
                rwrist = p_landmarks[16] # Static task keypoint for Right Wrist
                lwrist = p_landmarks[15] # Static task keypoint for Left Wrist
                nose = p_landmarks[0]    # Nose
                
                if rwrist.y > 0.0 and lwrist.y > 0.0:
                    if abs(rwrist.y - nose.y) < 0.25 or abs(lwrist.y - nose.y) < 0.25:
                        if not behaviour_flags["Laptop Usage"]: behaviour_flags["Reading"] = True
                    elif rwrist.y > nose.y and rwrist.y < 0.85:
                        behaviour_flags["Writing"] = True

            score = compute_attention_score(behaviour_flags)
            log_behaviour(student_id, behaviour_flags, score)

            color = (0, 255, 0) if score > 70 else (0, 165, 255) if score > 40 else (0, 0, 255)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
            cv2.putText(frame, f"ID: {student_id} | {int(score)}%", (sx1, max(sy1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            idx = 0
            for action, flag in behaviour_flags.items():
                if flag:
                    cv2.putText(frame, f"• {action}", (sx1 + 5, sy1 + 20 + (idx * 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    idx += 1
        return frame

    def close(self):
        self.face_landmarker.close()
        self.pose_landmarker.close()