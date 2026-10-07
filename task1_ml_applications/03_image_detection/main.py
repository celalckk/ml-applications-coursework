"""
Task 1 - Assignment 3: Image Object Detection
Library: PyTorch Hub
Model: YOLOv5s (ultralytics/yolov5)
"""

import torch
import cv2

print("Loading YOLOv5 model from PyTorch Hub...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
print("Model ready!\n")

image_path = "task1_ml_applications/03_image_detection/sample.jpg"
print(f"Running detection on: {image_path}")

results = model(image_path)

print("\n" + "=" * 60)
print("DETECTION RESULTS")
print("=" * 60)

# Print detected objects
df = results.pandas().xyxy[0]
for index, row in df.iterrows():
    print(f"  {row['name']:15s} | Confidence: {row['confidence']:.2%} | Box: ({row['xmin']:.0f},{row['ymin']:.0f})-({row['xmax']:.0f},{row['ymax']:.0f})")

print(f"\nTotal objects detected: {len(df)}")

# Save annotated image
results.save(save_dir="task1_ml_applications/03_image_detection/")
print("\nAnnotated image saved to: task1_ml_applications/03_image_detection/")

print("=" * 60)
print("Detection completed!")
print("=" * 60)