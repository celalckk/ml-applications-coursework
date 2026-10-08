"""
Task 1 - Assignment 4: Video Object Detection
Library: PyTorch Hub (YOLOv5) + OpenCV
"""

import torch
import cv2
import os

print("Loading YOLOv5 model from PyTorch Hub...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
print("Model ready!\n")


input_video = "task1_ml_applications/04_video_detection/sample.mp4"
output_video = "task1_ml_applications/04_video_detection/output.mp4"


cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {input_video}")


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("=" * 60)
print(f"Input video : {input_video}")
print(f"Resolution  : {width}x{height}")
print(f"FPS         : {fps}")
print(f"Total frames: {total_frames}")
print("=" * 60)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_count = 0
all_detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

  
    results = model(frame)

 
    annotated_frame = results.render()[0]

   
    out.write(annotated_frame)

    
    df = results.pandas().xyxy[0]
    for _, row in df.iterrows():
        all_detections.append(row['name'])

  
    if frame_count % 10 == 0:
        print(f"  Processed {frame_count}/{total_frames} frames...")

cap.release()
out.release()


print("\n" + "=" * 60)
print("DETECTION SUMMARY")
print("=" * 60)
print(f"Total frames processed: {frame_count}")

if all_detections:
    from collections import Counter
    counts = Counter(all_detections)
    print("\nDetected object types across all frames:")
    for obj, count in counts.most_common():
        print(f"  {obj:15s} : {count} detections")

print(f"\nOutput video saved to: {output_video}")
print("=" * 60)
print("Video detection completed!")
print("=" * 60)