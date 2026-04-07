import cv2
import math
import numpy as np
import argparse
import os
from collections import defaultdict
from ultralytics import YOLO

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def normalize_vector(v):
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-4 else np.zeros_like(v)

def compute_directional_coherence(velocities):
    if len(velocities) < 2:
        return 0.0
    unit_vectors = np.array([normalize_vector(v) for v in velocities])
    mean_vector = unit_vectors.mean(axis=0)
    return float(np.linalg.norm(mean_vector))

def compute_zone_density(centers, frame_shape, rows=3, cols=3):
    counts = np.zeros((rows, cols), dtype=int)
    h, w = frame_shape[:2]
    zone_h = h / rows
    zone_w = w / cols
    for x, y in centers:
        row = min(int(y / zone_h), rows - 1)
        col = min(int(x / zone_w), cols - 1)
        counts[row, col] += 1
    return counts

def draw_zone_grid(frame, zone_counts, zone_capacity):
    rows, cols = zone_counts.shape
    h, w = frame.shape[:2]
    zone_h = int(h / rows)
    zone_w = int(w / cols)
    overlay = frame.copy()
    for r in range(rows):
        for c in range(cols):
            count = zone_counts[r, c]
            ratio = min(count / max(zone_capacity, 1), 1.0)
            color = (
                int(255 * ratio),
                int(64 * (1 - ratio)),
                int(50 + 150 * ratio)
            )
            x1, y1 = c * zone_w, r * zone_h
            x2, y2 = x1 + zone_w, y1 + zone_h
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(frame, str(count),
                        (x1 + 8, y1 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

def draw_density_heatmap(frame, centers):
    if not centers:
        return
    heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
    for x, y in centers:
        cv2.circle(heatmap, (int(x), int(y)), 45, 1.0, -1)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=30)
    heatmap = np.uint8(np.clip(heatmap * 255, 0, 255))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)
    cv2.addWeighted(heatmap_color, 0.30, frame, 0.70, 0, frame)

def compute_optical_flow_score(prev_gray, curr_gray):
    """Returns a chaos score 0-1 based on how sudden/chaotic motion is"""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_magnitude = np.mean(magnitude)
    # Normalize to 0-1 (values above 8 pixels/frame = very chaotic)
    return min(avg_magnitude / 8.0, 1.0)

def get_alert_level(risk_score, coherence_score, potential_score):
    if potential_score >= 0.80 or (risk_score >= 0.75 and coherence_score >= 0.60):
        return "CRITICAL - STAMPEDE POTENTIAL", (0, 0, 255)
    elif risk_score >= 0.50 or potential_score >= 0.65:
        return "WARNING - CROWD BUILDING", (0, 140, 255)
    else:
        return "SAFE", (0, 200, 0)

def main(source , model_path, output_path=None):
    # Load model
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"Loaded model: {model_path}")
    else:
        print(f"Warning: {model_path} not found. Using default yolov8n.pt")
        model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Setup video writer if output is specified
    writer = None
    if output_path is not None:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps == 0: fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    # Tracking config
    track_history  = defaultdict(lambda: [])
    HISTORY_LEN    = 15
    SPEED_THRESH   = 15.0
    DENSITY_THRESH = 10
    MAX_CAPACITY   = 30   # tune this for your venue size

    # Optical flow setup
    prev_gray = None

    print("Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Layer 1: YOLOv8 density + speed (ShanghaiTech logic) ──
        results = model.track(frame, persist=True,
                       tracker="bytetrack.yaml", verbose=False,
                       conf=0.25, iou=0.45, classes=[0])
        frame_density  = 0
        current_speeds = []
        track_centers  = []
        velocities     = []

        if results and results[0].boxes is not None and results[0].boxes.xywh is not None and results[0].boxes.xywh.shape[0] > 0:
            boxes = results[0].boxes.xywh.cpu().tolist()
            frame_density = len(boxes)

            if hasattr(results[0].boxes, "id") and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = list(range(frame_density))

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = map(float, box)
                center = (x, y)
                track_centers.append(center)

                history = track_history[track_id]
                if history and len(history) > 0:
                    prev_center = history[-1]
                    velocities.append(np.array([x - prev_center[0], y - prev_center[1]], dtype=np.float32))

                history.append(center)
                if len(history) > HISTORY_LEN:
                    history.pop(0)
                track_history[track_id] = history

                # Draw box and ID
                pt1 = (int(x - w/2), int(y - h/2))
                pt2 = (int(x + w/2), int(y + h/2))
                cv2.rectangle(frame, pt1, pt2, (255, 165, 0), 2)
                cv2.putText(frame, f"ID:{track_id}",
                            (pt1[0], pt1[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,165,0), 1)

                # Speed calculation
                if len(history) > 2:
                    old_center = history[0]
                    dist = calculate_distance(old_center, center)
                    frames_passed = len(history) - 1
                    current_speeds.append(dist / frames_passed)

            # Draw trails
            for track_id, pts in track_history.items():
                for i in range(1, len(pts)):
                    cv2.line(frame,
                             (int(pts[i-1][0]), int(pts[i-1][1])),
                             (int(pts[i][0]),   int(pts[i][1])),
                             (0, 255, 0), 1)

        avg_speed = np.mean(current_speeds) if current_speeds else 0

        # ── Layer 2: Optical flow chaos score (UMN logic) ──
        flow_score = 0.0
        if prev_gray is not None:
            flow_score = compute_optical_flow_score(prev_gray, curr_gray)
        prev_gray = curr_gray

        # ── Layer 3: Novel crowd pressure and stampede potential
        density_score = min(frame_density / MAX_CAPACITY, 1.0)
        speed_score   = min(avg_speed / SPEED_THRESH, 1.0)
        motion_score  = (speed_score + flow_score) / 2.0
        coherence_score = compute_directional_coherence(velocities)

        dispersion = 1.0
        if len(track_centers) > 1:
            center_array = np.array(track_centers, dtype=np.float32)
            spread = np.linalg.norm(np.std(center_array, axis=0))
            dispersion = min(spread / math.hypot(frame.shape[1], frame.shape[0]), 1.0)

        zone_counts = compute_zone_density(track_centers, frame.shape)
        zone_capacity = max(1, MAX_CAPACITY // (zone_counts.size))
        zone_pressure = min(np.max(zone_counts) / zone_capacity, 1.0)
        crowd_pressure = min(0.45 * density_score + 0.30 * coherence_score + 0.15 * (1.0 - dispersion) + 0.10 * zone_pressure, 1.0)
        stampede_potential = min(0.5 * crowd_pressure + 0.5 * motion_score, 1.0)

        risk_score = max((0.6 * density_score) + (0.4 * motion_score), stampede_potential)
        alert_text, alert_color = get_alert_level(risk_score, coherence_score, stampede_potential)

        draw_density_heatmap(frame, track_centers)
        draw_zone_grid(frame, zone_counts, zone_capacity)

        # ── UI: info panel (top left) ──
        cv2.rectangle(frame, (0, 0), (330, 150), (0, 0, 0), -1)
        cv2.putText(frame, f"People:            {frame_density}",
                    (10, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Avg Speed:         {avg_speed:.1f} px/f",
                    (10, 56),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Flow Chaos:        {flow_score:.2f}",
                    (10, 84),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Direction Coherence: {coherence_score:.2f}",
                    (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Risk Score:        {risk_score:.2f}",
                    (10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)

        # ── UI: alert banner (bottom of frame) ──
        h_frame = frame.shape[0]
        w_frame = frame.shape[1]
        cv2.rectangle(frame, (0, h_frame-50), (w_frame, h_frame), (0,0,0), -1)
        cv2.putText(frame, alert_text,
                    (10, h_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, alert_color, 3)

        if writer is not None:
            writer.write(frame)
        else:
            cv2.imshow("Crowd Stampede Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

# ===============================
# FUNCTIONS FOR WEB APP
# ===============================

def process_video(input_path, output_path):
    print(f"[INFO] Processing video: {input_path}")
    main(input_path, "models/best.pt", output_path)
    return output_path


def run_webcam():
    print("[INFO] Starting webcam...")
    main(0, "models/best.pt", None)

#=================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0",
                        help="'0' for webcam or path to video file")
    parser.add_argument("--model",  type=str, default="best.pt",
                        help="Path to your YOLOv8 model weights")
    parser.add_argument("--output", type=str, default=None,
                        help="Save output video to this path (optional)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    main(source, args.model, args.output)