# Video Object Detection

## 1. Library / Solution Advantages

**Libraries:** PyTorch Hub (YOLOv5) + OpenCV

Advantages:
- Same YOLOv5 model reused — consistent detection quality.
- OpenCV provides fast frame-by-frame video I/O.
- Frame-level detection can be extended to tracking (DeepSORT, ByteTrack).

## 2. Model Principle and Selection Reasoning

**How it works:**
- OpenCV reads the video frame-by-frame using `cv2.VideoCapture`.
- Each frame is passed to YOLOv5 (same model as Assignment 3).
- Detected boxes are rendered on the frame using `results.render()`.
- The annotated frames are written to a new video file via `cv2.VideoWriter`.

**Why this combination:**
- YOLOv5 is fast enough for near-real-time CPU inference on 720p video.
- OpenCV is the standard tool for video processing in Python.

## 3. Dataset Structure

**For inference:**
- Input: video file (MP4, AVI).
- Processing: frame extraction at original FPS.
- Output: annotated video + per-frame detection list.

**For training:** Same COCO dataset as Assignment 3.

## 4. Quality Metrics

- **Frames Per Second (FPS)** of the processing pipeline.
- **Per-frame mAP** (if ground truth is available).
- **Detection consistency** across frames (temporal stability).
- **End-to-end latency** for a video of given length.

## 5. Implementation Features and Efficiency

- Video is processed sequentially; no parallelization in this demo.
- CPU inference: ~2–5 FPS for 720p.
- For production: use GPU, batch frames, or deploy YOLOv5 in TensorRT.
- Output video uses `mp4v` codec for broad compatibility.