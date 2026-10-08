# Image Object Detection

## 1. Library / Solution Advantages

**Library:** PyTorch Hub
**Model:** YOLOv5s (`ultralytics/yolov5`)

Advantages:
- One-line load: `torch.hub.load('ultralytics/yolov5', 'yolov5s')`.
- YOLOv5 is extremely well-optimized for CPU and GPU.
- Detects 80 COCO classes out of the box.
- Extensive documentation and active community.

## 2. Model Principle and Selection Reasoning

**How YOLOv5 works:**
- The image is resized and passed through a CNN backbone (CSPDarknet).
- A Feature Pyramid Network (FPN) merges features at multiple scales.
- The detection head predicts bounding boxes, class probabilities, and objectness scores in a single forward pass ("You Only Look Once").
- Non-Maximum Suppression (NMS) removes overlapping boxes.

**Why YOLOv5s:**
- Small model (~15 MB) — fast even on CPU.
- Good accuracy/speed trade-off.
- Easier to install and use than YOLOv8 in this context.

## 3. Dataset Structure

**For inference:**
- Input: RGB image (JPG / PNG).
- Preprocessing: resize to 640×640, normalization.
- Output: list of boxes with `(xmin, ymin, xmax, ymax, confidence, class)`.

**For training (background):**
- COCO dataset: 118,000 training images, 80 object classes.
- YOLOv5s pre-trained weights are used as-is.

## 4. Quality Metrics

- **mAP@0.5** (mean Average Precision at IoU 0.5): ~37% for YOLOv5s.
- **mAP@0.5:0.95**: ~56% for YOLOv5s.
- **Inference speed**: ~20–40 ms per image on CPU.
- **Confidence threshold**: default 0.25.

## 5. Implementation Features and Efficiency

- Model is cached after first `torch.hub.load()` call.
- Uses OpenCV + PIL for image I/O.
- Output video/image is saved with bounding boxes drawn.
- For production: batch processing, GPU acceleration, or export to ONNX/TensorRT.