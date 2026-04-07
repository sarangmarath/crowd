# Integrated Crowd Detection + Abnormal Behavior Detection Pipeline

This pipeline combines **ShanghaiTech crowd detection** with **UMN abnormal behavior detection** for a comprehensive stampede prevention system.

## Architecture

```
Video Frame
    ↓
[Stage 1: YOLO Detection] 
    → Detects all people in frame
    → Counts density, measures speed
    ↓
[Stage 2-4: UMN Anomaly Detection]
    → Extracts ROIs around each person (limited to max_rois)
    → Applies CLAHE preprocessing
    → Classifies as normal/abnormal behavior
    ↓
[Stage 5: Risk Scoring]
    → Combines crowd metrics + abnormality scores
    → Generates SAFE / WARNING / CRITICAL alert
    ↓
Output: Annotated frame with dual scores
```

## Files

- `train_yolo.py` - Train ShanghaiTech YOLO object detector
- `train_umn_model.py` - Train UMN abnormal behavior classifier
- `integrated_pipeline.py` - Run both models together

## Step 1: Train Object Detection (YOLO)

If you have the ShanghaiTech dataset:

```bash
python train_yolo.py \
    --shanghaitech-dir "C:/C Model/ShanghaiTech_Crowd_Counting_Dataset" \
    --target-dir "./datasets/crowd" \
    --epochs 50 \
    --batch 16 \
    --imgsz 640
```

Output: `./runs/train/crowd_yolov8/weights/best.pt`

If you already have a YOLO dataset YAML:

```bash
python train_yolo.py \
    --data-yaml "./datasets/crowd/crowd_dataset.yaml" \
    --epochs 50
```

## Step 2: Train Abnormal Behavior Detection (UMN)

Using CLAHE-preprocessed UMN dataset:

```bash
python train_umn_model.py \
    --train-dir "CrowdAnomalyDetection_DeepLearning-master/UMN/Final UMN dataset/UMN_CHALE_5159/train_UMN_clahe" \
    --test-dir "CrowdAnomalyDetection_DeepLearning-master/UMN/Final UMN dataset/UMN_CHALE_5159/test_UMN_clahe" \
    --output-model "umn_abnormal_model.h5" \
    --epochs 30 \
    --batch-size 32
```

Output: `umn_abnormal_model.h5`

## Step 3: Run Integrated Pipeline

**With webcam:**

```bash
python integrated_pipeline.py \
    --source 0 \
    --yolo-model "best.pt" \
    --umn-model "umn_abnormal_model.h5" \
    --max-rois 10
```

**With video file:**

```bash
python integrated_pipeline.py \
    --source "stampede_test.mp4" \
    --yolo-model "best.pt" \
    --umn-model "umn_abnormal_model.h5" \
    --output "output_annotated.mp4" \
    --max-rois 10
```

## Arguments

### `integrated_pipeline.py`

- `--source` (default: "0") - Webcam (0) or video file path
- `--yolo-model` (default: "best.pt") - Path to trained YOLO weights
- `--umn-model` (default: None) - Path to trained UMN .h5 model
- `--output` (default: None) - Save output video (optional)
- `--max-rois` (default: 10) - Max ROIs to process per frame (UMN bottleneck)

## Output Annotations

Each frame displays:

**Left Panel (Black background):**
- People Detected: Count of detected individuals
- Avg Speed: Average pixel velocity per frame
- Flow Chaos: Optical flow magnitude (0-1)
- Abnormality Avg: Average abnormality score from UMN model
- Combined Risk: Final risk score (0-1)

**Per-Person Boxes:**
- Orange box with ID: Detected person + tracking ID
- Green text "Abn: X.XX": Normal behavior (< 0.5)
- Red text "Abn: X.XX": Abnormal behavior (> 0.5)

**Bottom Banner:**
- **SAFE**: Low risk (combined_risk < 0.5)
- **WARNING**: Moderate risk (0.5 ≤ combined_risk < 0.75)
- **CRITICAL**: High risk with anomalies (combined_risk ≥ 0.75)

## How the Pipeline Works

1. **YOLO detects** all people in the frame (no limit)
2. **Limited ROI extraction**: Only processes first `max_rois` people (default: 10)
3. **For each ROI**:
   - Extract bounding box region
   - Apply CLAHE (histogram equalization)
   - Resize to 224×224
   - Feed to UMN model
   - Get abnormality score (0=normal, 1=abnormal)
4. **Risk calculation**: Blends crowd metrics (density/speed/flow) with abnormality scores
5. **Alert thresholds**: CRITICAL if high risk + anomalies detected together

## Tips for Better Results

### Improve YOLO Detection:
- Use more training data
- Train longer (increase `--epochs`)
- Use larger model (`yolov8m.pt` or `yolov8l.pt` instead of yolov8n.pt)

### Improve UMN Abnormality:
- Ensure CLAHE images are properly preprocessed
- Train with balanced classes (script handles oversampling)
- Test different `--batch-size` values (32, 64)
- Monitor validation metrics

### Adjust Pipeline Parameters:
- Increase `--max-rois` if you have GPU memory (slower but more thorough)
- Decrease `--max-rois` for real-time performance
- Tune alert thresholds in code (lines ~330-340)

## Expected Performance

- **Real-time**: 10-15 FPS with max_rois=10 on modern GPU
- **Accuracy**: Depends on training data quality and epochs
- **Latency**: UMN model runs per-person, so 10 people = 10 forward passes

## Troubleshooting

### "No module named tensorflow"
```bash
pip install tensorflow
```

### "UMN model not found"
Provide `--umn-model` path or train first with `train_umn_model.py`

### Slow performance
- Reduce `--max-rois` (e.g., 5 or 3)
- Use `device='gpu'` if available
- Reduce frame resolution

### Low detection accuracy
- Check YOLO training loss curves
- Ensure ShanghaiTech data is properly converted
- Train with more epochs and larger batch size

## References

- ShanghaiTech Dataset: Crowd counting benchmark
- UMN Dataset: Abnormal crowd behavior in surveillance
- CLAHE: Contrast-limited adaptive histogram equalization for preprocessing
