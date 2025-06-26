# FootballIQ: AI-Powered Player & Ball Analytics for Football

[![Model Download Link](https://img.shields.io/badge/Download%20Models-Google%20Drive-blue)](https://drive.google.com/drive/folders/11AkiuTpAJ3GOGeFYlXz3qbRCr8StLNdT?usp=sharing)


## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Object Detection with YOLOv8](#object-detection-with-yolov8)
  - [Why YOLOv8?](#why-yolov8)
  - [Implementation Details](#implementation-details)
  - [Challenges and Solutions](#challenges-and-solutions)
- [Custom Training with Roboflow](#custom-training-with-roboflow)
  - [Dataset Preparation](#dataset-preparation)
  - [Training Results](#training-results)
- [Object Tracking & Unique ID Assignment](#object-tracking--unique-id-assignment)
  - [Tracking Algorithms Used](#tracking-algorithms-used)
  - [Visualizing Tracking](#visualizing-tracking)
- [Team Assignment using KMeans Clustering](#team-assignment-using-kmeans-clustering)
- [Camera Motion Estimation with Optical Flow](#camera-motion-estimation-with-optical-flow)
- [Perspective Transformation for Real-World Metrics](#perspective-transformation-for-real-world-metrics)
- [Player Speed & Distance Calculation](#player-speed--distance-calculation)
- [Full Pipeline Walkthrough](#full-pipeline-walkthrough)
- [Conclusion & Future Work](#conclusion--future-work)
- [References](#references)

---

## Introduction

**FootballIQ** is an end-to-end AI/Computer Vision system designed to analyze football games from broadcast video. It detects and tracks players, referees, and the ball, then extracts advanced analytics such as player speed, distance covered, and team control. The system leverages the latest advances in deep learning, including YOLOv8 for object detection, custom training for football-specific classes, advanced tracking, clustering, optical flow, and perspective transformation.

This project demonstrates a complete real-world application of computer vision, machine learning, and data analytics in sports. It is accessible for both beginners and experienced ML engineers, and extensible for other sports or analytics use-cases.

---

## Project Overview

**Key Capabilities:**
1. **Object Detection:** Detects players, referees, and ball using YOLOv8.
2. **Custom Training:** Fine-tunes detection models on football-specific datasets for higher accuracy.
3. **Object Tracking:** Assigns unique IDs to each player and ball, maintaining identity across frames.
4. **Team Assignment:** Uses pixel clustering (KMeans) to segment and assign players to teams based on jersey color.
5. **Camera Motion Correction:** Estimates and removes camera movement using optical flow.
6. **Real-World Analytics:** Measures player speed and distance covered in meters via perspective transformation.
7. **Visualization:** Annotates frames with bounding boxes, IDs, team assignments, and calculated stats.

---

## Environment Setup

- Python 3.8+
- Jupyter Notebook (main notebooks)
- Required libraries: `ultralytics`, `opencv-python`, `supervision`, `scikit-learn`, `numpy`, `matplotlib`
- Pretrained/finetuned model weights: [Download from Google Drive](https://drive.google.com/drive/folders/11AkiuTpAJ3GOGeFYlXz3qbRCr8StLNdT?usp=sharing)

Install dependencies:
```bash
pip install ultralytics opencv-python supervision scikit-learn numpy matplotlib
```

---

## Object Detection with YOLOv8

### Why YOLOv8?

YOLO (You Only Look Once) is a family of state-of-the-art, real-time object detectors. YOLOv8, provided by Ultralytics, offers fast and accurate detection, making it ideal for analyzing football matches where both speed and accuracy are crucial.

**Model Used:** `yolov8m` (medium-sized model, pre-trained on COCO)

- Detects 80 common classes, including `person` and `sports ball`.
- Outputs: bounding box coordinates (`x1, y1, x2, y2`), class, and confidence probability.

### Implementation Details

- Load YOLOv8 via Ultralytics API.
- Run inference on each frame of the input video.
- Collect bounding boxes, class labels, and probabilities.
- Draw results with OpenCV for visualization.

### Challenges and Solutions

#### Challenge 1: Ball Detection Inconsistency

- **Problem:** The ball was detected in only a few frames; sometimes missed due to size, occlusions, or similarity to the background.
- **Solution:** Switched to YOLOv5 for ball detection, as it exhibited higher robustness for small and fast-moving objects in football scenarios.

#### Challenge 2: False Positives (People Outside Field)

- **Problem:** `person` class detected audience or people outside the playing field.
- **Solution:** Trained a custom model using the [Roboflow Football Player Detection Dataset](https://public.roboflow.com/object-detection/football-players), which focuses on in-field players and football-specific classes.

---

## Custom Training with Roboflow

### Dataset Preparation

- Downloaded annotated football images from Roboflow.
- Annotated classes: `{0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}`
- Split into train/validation sets.
- Used YOLOv8's training script to finetune on the new dataset.

### Training Results

- Training and validation losses decreased steadily, indicating effective learning.
- Precision, recall, and mAP metrics improved for all classes.

![image](https://github.com/user-attachments/assets/994814c2-c9b4-49b9-b63f-88d5532112eb)

---

## Object Tracking & Unique ID Assignment

Detection alone is insufficient; we need to track individuals and the ball across frames to analyze movement, speed, and interactions.

### Tracking Algorithms Used

- Used the [Supervision](https://github.com/roboflow/supervision) library for object tracking.
- Each detected player or ball receives a unique tracker ID.
- Custom logic ensures correct identity assignment, even during occlusions or close encounters.

**Note:** Closest bounding box assignment can fail in close calls; advanced matching algorithms (e.g., Hungarian algorithm) are used for robustness.

### Visualizing Tracking

- Players are annotated with an ellipse and their unique tracker ID.
- The ball is annotated with a triangle.
- Tracks are drawn over time to visualize player and ball movement paths.

---

## Team Assignment using KMeans Clustering

Assigning detected players to teams is done by analyzing the dominant color of their jerseys.

- Extract the pixel region inside each player's bounding box.
- Use KMeans clustering to segment pixels and identify the dominant color.
- Compare colors to known team palettes to assign a team label.
- Visualize with colored bounding boxes or labels.

---

## Camera Motion Estimation with Optical Flow

To measure player movement accurately, we must isolate movement due to camera panning/zooming.

- Calculate optical flow between consecutive frames using OpenCV's `calcOpticalFlowFarneback` or similar methods.
- Estimate the average camera movement in X and Y.
- Subtract camera motion from each player's movement to get true player displacement.

---

## Perspective Transformation for Real-World Metrics

Pixel movement is not equivalent to real-world distances due to perspective.

- Use OpenCV's perspective transformation (`cv2.getPerspectiveTransform`, `cv2.warpPerspective`) with manually selected reference points (e.g., corners of the field).
- Transform image coordinates into a "bird's-eye view" where distances are proportional to meters.
- All player movement is now measured in meters.

---

## Player Speed & Distance Calculation

- For each tracked player, calculate the displacement between frames (in meters) and divide by the frame interval to estimate speed (m/s or km/h).
- Aggregate distances to compute total distance covered over time.
- Visualize per-player stats directly on the frame, as shown below.

![image](https://github.com/user-attachments/assets/7433d24a-ff87-448e-b972-0612ea11396c)


---

## Full Pipeline Walkthrough

1. **Input Video** → Extract frames.
2. **YOLOv8 Detection** → Detect players, referees, and ball on each frame.
3. **Custom Dataset Inference** → Improve detection for football scenario.
4. **Tracking** → Assign unique IDs and track objects across frames.
5. **Team Color Clustering** → Assign team labels using KMeans on jersey colors.
6. **Camera Motion Estimation** → Compute and remove camera movement.
7. **Perspective Transform** → Map pixel positions to real-world coordinates.
8. **Calculate Speed/Distance** → For each player, compute and annotate stats.
9. **Visualization** → Draw bounding boxes, IDs, team colors, speeds, and distances on frames.
10. **Analytics Output** → Save annotated video and statistical summaries.

---

## Conclusion & Future Work

This project demonstrates a comprehensive, real-world application of deep learning and computer vision in sports analytics. By combining state-of-the-art detection, custom training, advanced tracking, clustering, and geometric transformations, we deliver actionable insights from football video footage.

**Possible Extensions:**
- Add event detection (passes, goals, fouls).
- Expand to support other sports.
- Deploy as a web app for real-time analytics.
- Integrate with broadcast overlays for live matches.

---

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Football Dataset](https://public.roboflow.com/object-detection/football-players)
- [Supervision Library](https://github.com/roboflow/supervision)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

---

For questions, issues, or contributions, please open an [issue](https://github.com/Akshatkt/FootballIQ-AI-Powered-Player-Ball-Analytics/issues) or submit a pull request.

---
