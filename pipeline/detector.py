"""
Integrated pipeline: ShanghaiTech (object detection) + UMN (abnormal behavior detection)

Stage 1: ShanghaiTech model detects all people in the frame
Stage 2: Extract ROIs around detected people
Stage 3: Preprocess ROIs (CLAHE, resize to 224x224)
Stage 4: Feed to UMN-trained model for abnormal behavior classification
Stage 5: Visualize results with dual scores
"""

import cv2
import numpy as np
import math
import argparse
import os
import json
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model


def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def apply_clahe(image, clip_limit=2.0, tile_size=(8, 8)):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)"""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(image)


def extract_and_preprocess_roi(frame, x1, y1, x2, y2, target_size=(224, 224)):
    """Extract ROI from frame, apply CLAHE, and resize to target size"""
    # Clamp coordinates to frame bounds
    h, w = frame.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    
    # Extract ROI
    roi = frame[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return None
    
    # Apply CLAHE
    roi = apply_clahe(roi)
    
    # Resize to target size
    roi = cv2.resize(roi, target_size)
    
    return roi


def normalize_image(image):
    """Normalize image to [0, 1] range"""
    return image.astype(np.float32) / 255.0


def compute_optical_flow_score(prev_gray, curr_gray):
    """Returns a chaos score 0-1 based on optical flow"""
    if prev_gray is None:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_magnitude = np.mean(magnitude)
    return min(avg_magnitude / 8.0, 1.0)


def main(source, yolo_model_path, umn_model_path=None, output_path=None, max_rois=10):
    """
    Run integrated detection pipeline.
    
    Args:
        source: video file or camera index
        yolo_model_path: path to trained ShanghaiTech YOLO model
        umn_model_path: path to trained UMN abnormal behavior model
        output_path: optional output video file path
        max_rois: maximum number of ROIs to process per frame (UMN limitation)
    """
    
    # Load YOLO model for object detection
    if os.path.exists(yolo_model_path):
        yolo_model = YOLO(yolo_model_path)
        print(f"Loaded YOLO model: {yolo_model_path}")
    else:
        print(f"Warning: {yolo_model_path} not found. Using default yolov8n.pt")
        yolo_model = YOLO("yolov8n.pt")
    
    # Load UMN model for abnormal behavior detection (if provided)
    umn_model = None
    if umn_model_path and os.path.exists(umn_model_path):
        umn_model = load_model(umn_model_path)
        print(f"Loaded UMN model: {umn_model_path}")
    else:
        print("Warning: UMN model not provided or not found. Using placeholder anomaly scores.")
    
    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Setup video writer
    writer = None
    if output_path is not None:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps == 0:
            fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    
    # Tracking and history
    track_history = defaultdict(lambda: [])
    HISTORY_LEN = 15
    SPEED_THRESH = 15.0
    MAX_CAPACITY = 30
    prev_gray = None
    
    # Data logging
    log_data = {
        'frame_count': 0,
        'total_detections': 0,
        'avg_density': [],
        'avg_speed': [],
        'avg_abnormality': [],
        'risk_levels': [],
        'max_risk': 0.0
    }
    
    print("Press 'q' to quit.")
    frame_count = 0
    skip_frames = 0  # Process every other frame for speed
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        skip_frames += 1
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (640, 360))
        curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # ──── STAGE 1: YOLO Object Detection ────
        results = yolo_model.track(
            small_frame, persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            conf=0.3,
            iou=0.45,
            classes=[0]
        )
        
        frame_density = 0
        current_speeds = []
        detected_boxes = []
        track_centers = []
        
        if results and results[0].boxes is not None and results[0].boxes.xywh is not None:
            if results[0].boxes.xywh.shape[0] > 0:
                boxes = results[0].boxes.xywh.cpu().tolist()
                xyxy = results[0].boxes.xyxy.cpu().tolist()
                frame_density = len(boxes)
                
                if hasattr(results[0].boxes, "id") and results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                else:
                    track_ids = list(range(frame_density))
                
                for idx, (box, box_xyxy, track_id) in enumerate(zip(boxes, xyxy, track_ids)):
                    if idx >= max_rois:
                        break
                    
                    x, y, w, h = map(float, box)
                    center = (x, y)
                    x1, y1, x2, y2 = map(float, box_xyxy)
                    
                    track_centers.append(center)
                    history = track_history[track_id]
                    history.append(center)
                    if len(history) > HISTORY_LEN:
                        history.pop(0)
                    track_history[track_id] = history
                    
                    # Speed calculation
                    if len(history) > 2:
                        old_center = history[0]
                        dist = calculate_distance(old_center, center)
                        frames_passed = len(history) - 1
                        current_speeds.append(dist / frames_passed)
                    
                    # Scale coordinates back to original frame size
                    scale_x = frame.shape[1] / small_frame.shape[1]
                    scale_y = frame.shape[0] / small_frame.shape[0]
                    x1_scaled = x1 * scale_x
                    y1_scaled = y1 * scale_y
                    x2_scaled = x2 * scale_x
                    y2_scaled = y2 * scale_y
                    center_scaled = (center[0] * scale_x, center[1] * scale_y)
                    
                    detected_boxes.append({
                        'id': track_id,
                        'x1': x1_scaled, 'y1': y1_scaled, 'x2': x2_scaled, 'y2': y2_scaled,
                        'center': center_scaled,
                        'history': history
                    })
                
                # Draw trails
                for track_id, pts in track_history.items():
                    for i in range(1, len(pts)):
                        cv2.line(frame,
                                 (int(pts[i-1][0]), int(pts[i-1][1])),
                                 (int(pts[i][0]), int(pts[i][1])),
                                 (0, 255, 0), 1)
        
        avg_speed = np.mean(current_speeds) if current_speeds else 0
        
        # ──── STAGE 2-4: UMN Abnormal Behavior Detection ────
        abnormality_scores = {}
        abnormal_track_ids = []  # Track IDs with abnormal behavior
        
        if umn_model is not None and len(detected_boxes) > 0:
            for box_info in detected_boxes:
                track_id = box_info['id']
                x1, y1, x2, y2 = int(box_info['x1']), int(box_info['y1']), int(box_info['x2']), int(box_info['y2'])
                
                try:
                    # Extract and preprocess ROI
                    roi = extract_and_preprocess_roi(frame, x1, y1, x2, y2)
                    if roi is not None and roi.size > 0:
                        roi_normalized = normalize_image(roi)
                        roi_batch = np.expand_dims(roi_normalized, axis=0)
                        
                        # Predict abnormality with lower verbosity
                        prediction = umn_model.predict(roi_batch, verbose=0)
                        
                        # Handle different prediction formats
                        if isinstance(prediction, np.ndarray):
                            abnormality_score = float(prediction[0, 0]) if prediction.shape[1] > 0 else 0.0
                        elif isinstance(prediction, list):
                            abnormality_score = float(prediction[0][0]) if len(prediction) > 0 else 0.0
                        else:
                            abnormality_score = float(prediction[0]) if prediction else 0.0
                        
                        abnormality_scores[track_id] = abnormality_score
                        if abnormality_score > 0.5:
                            abnormal_track_ids.append(track_id)
                except Exception as e:
                    abnormality_scores[track_id] = 0.0
        
        # ──── STAGE 5: Optical Flow Chaos Score (skip alternate frames for speed) ────
        flow_score = 0.0
        if prev_gray is not None and frame_count % 2 == 0:
            flow_score = compute_optical_flow_score(prev_gray, curr_gray)
        prev_gray = curr_gray
        
        # ──── Compute Risk Scores ────
        density_score = min(frame_density / MAX_CAPACITY, 1.0)
        speed_score = min(avg_speed / SPEED_THRESH, 1.0)
        motion_score = (speed_score + flow_score) / 2.0
        
        base_risk = (0.6 * density_score) + (0.4 * motion_score)
        
        # Enhance risk with abnormality scores
        if abnormality_scores:
            avg_abnormality = np.mean(list(abnormality_scores.values()))
            combined_risk = min(base_risk * 0.6 + avg_abnormality * 0.4, 1.0)
        else:
            combined_risk = base_risk
            avg_abnormality = 0.0
        
        # Log data
        log_data['frame_count'] = frame_count
        log_data['total_detections'] += frame_density
        log_data['avg_density'].append(density_score)
        log_data['avg_speed'].append(speed_score)
        log_data['avg_abnormality'].append(avg_abnormality)
        log_data['risk_levels'].append(combined_risk)
        log_data['max_risk'] = max(log_data['max_risk'], combined_risk)
        
        # ──── Alert Level ────
        if combined_risk >= 0.75:
            alert_text = "CRITICAL - STAMPEDE + ANOMALY RISK"
            alert_color = (0, 0, 255)
        elif combined_risk >= 0.50:
            alert_text = "WARNING - CROWD + ANOMALY DETECTED"
            alert_color = (0, 140, 255)
        else:
            alert_text = "SAFE"
            alert_color = (0, 200, 0)
        
        # ──── Render Bounding Boxes: GREEN for all detections ────
        for box_info in detected_boxes:
            track_id = box_info['id']
            x1, y1, x2, y2 = int(box_info['x1']), int(box_info['y1']), int(box_info['x2']), int(box_info['y2'])
            
            # Draw GREEN box for all person detections
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
            label = f"ID:{track_id}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # ──── Overlay BLUE boxes for abnormal detections only ────
        for track_id in abnormal_track_ids:
            # Find the corresponding box
            for box_info in detected_boxes:
                if box_info['id'] == track_id:
                    x1, y1, x2, y2 = int(box_info['x1']), int(box_info['y1']), int(box_info['x2']), int(box_info['y2'])
                    abn_score = abnormality_scores.get(track_id, 0.0)
                    
                    # Draw BLUE overlay box for abnormal behavior
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue
                    abn_label = f"ABN:{abn_score:.2f}"
                    cv2.putText(frame, abn_label, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    break
        
        # ──── UI: Info Panel ────
        cv2.rectangle(frame, (0, 0), (420, 200), (0, 0, 0), -1)
        y_offset = 28
        cv2.putText(frame, f"People Detected:  {frame_density}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y_offset += 28
        cv2.putText(frame, f"Abnormal Count:   {len(abnormal_track_ids)}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)
        y_offset += 28
        cv2.putText(frame, f"Avg Speed:        {avg_speed:.1f} px/f",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y_offset += 28
        cv2.putText(frame, f"Flow Chaos:       {flow_score:.2f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y_offset += 28
        cv2.putText(frame, f"Abnormality Avg:  {avg_abnormality:.2f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y_offset += 28
        cv2.putText(frame, f"Combined Risk:    {combined_risk:.2f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, alert_color, 2)
        
        # ──── Alert Banner ────
        h_frame = frame.shape[0]
        w_frame = frame.shape[1]
        cv2.rectangle(frame, (0, h_frame - 50), (w_frame, h_frame), (0, 0, 0), -1)
        cv2.putText(frame, alert_text,
                    (10, h_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, alert_color, 3)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}",
                    (w_frame - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Write or display
        if writer is not None:
            writer.write(frame)

    
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    
    # Save summary data
    summary = {
        'total_frames': log_data['frame_count'],
        'total_detections': log_data['total_detections'],
        'avg_density': np.mean(log_data['avg_density']) if log_data['avg_density'] else 0,
        'avg_speed': np.mean(log_data['avg_speed']) if log_data['avg_speed'] else 0,
        'avg_abnormality': np.mean(log_data['avg_abnormality']) if log_data['avg_abnormality'] else 0,
        'max_risk': log_data['max_risk'],
        'risk_distribution': {
            'safe': sum(1 for r in log_data['risk_levels'] if r < 0.33),
            'warning': sum(1 for r in log_data['risk_levels'] if 0.33 <= r < 0.67),
            'critical': sum(1 for r in log_data['risk_levels'] if r >= 0.67)
        }
    }
    with open('processing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Pipeline complete. Summary saved to processing_summary.json")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Max risk: {summary['max_risk']:.2f}")

# ======================================
# 🔥 FLASK WRAPPERS (IMPORTANT)
# ======================================

def process_video(input_path, output_path):
    print("🔥 Running Integrated Pipeline on Video")

    main(
        source=input_path,
        yolo_model_path="models/best.pt",
        umn_model_path="models/umn_abnormal_model.h5",
        output_path=output_path
    )

    print("✅ Done processing video")


def run_webcam():
    print("🔥 Running Integrated Pipeline on Webcam")

    main(
        source=0,
        yolo_model_path="models/best.pt",
        umn_model_path="models/umn_abnormal_model.h5",
        output_path=None
    )

#====================================================================
#====================================================================
#====================================================================



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrated pipeline: ShanghaiTech detection + UMN anomaly detection"
    )
    parser.add_argument("--source", type=str, default="0",
                        help="'0' for webcam or path to video file")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                        help="Path to ShanghaiTech-trained YOLO model")
    parser.add_argument("--umn-model", type=str, default=None,
                        help="Path to UMN-trained abnormal behavior model (.h5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save output video to this path (optional)")
    parser.add_argument("--max-rois", type=int, default=10,
                        help="Max number of ROIs to process per frame (UMN bottleneck)")
    args = parser.parse_args()
    
    source = int(args.source) if args.source.isdigit() else args.source
    main(source, args.yolo_model, args.umn_model, args.output, args.max_rois)
