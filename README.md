# 🚗 CrashSight – Real-Time Car Accident Detection with AI

CrashSight is a real-time car crash detection system that combines fast object tracking with deep scene understanding to identify accidents as they happen. It’s designed for real-world deployment on traffic footage — smart intersections, fleet monitoring, or dashcams.

---

##  What It Does

- Detects vehicle collisions from raw video using computer vision
- Tracks object motion and dynamics across time
- Classifies crash patterns using an attention-enhanced CNN
- Outputs annotated clips, attention maps, and structured crash reports

---

##  System Overview

CrashSight runs a two-stage accident detection pipeline:

### 1. Vehicle Tracking & Motion Risk Analysis

- **YOLOv8** detects vehicles in each frame  
- A custom **Centroid Tracker** estimates speed, acceleration, and trajectory angles  
- Collision candidates are flagged when:
  - Bounding boxes intersect
  - Motion shows abnormal deceleration or direction change

### 2. Scene Understanding via Attention CNN

- **ResNet18 + TSM + CBAM + Multi-Head Attention**
- Learns crash semantics from over 1,000 labeled videos
- Produces heatmaps (via Grad-CAM++) showing model attention

### Triggering Logic

By default, Module 2 runs **only when Module 1 flags risk** — either via bounding box collision or abnormal motion score.  
If hardware permits, both modules can run in parallel for full-score fusion:

```python
final_score = α * motion_score + β * scene_score
```

### Outputs

- 📹 Annotated video with bounding boxes, speed labels, collision tags  
- 🌡️ Attention heatmaps (visual explanations)  
- 📝 Structured crash reports (JSON & text)  
- 🎞️ Accident clips in H.264 (annotated + raw)

---

##  Key Features

- Real-time: ~30 FPS (GPU) or 10–15 FPS (CPU)
- Risk scoring based on motion vectors + collision angles
- Scene classification with attention maps
- Velocity and acceleration tracking per vehicle
- Collision type estimation (e.g. rear-end, side-impact)
- Auto-generated video segments for detected crashes

---

##  Project Structure

```
main.py                 # Core crash detection pipeline
models/                 # YOLO + CNN weights
app/                    # Web/API interface
utils/                  # Video processing tools
static/                 # Outputs, heatmaps, visual assets
```

---

##  Technical Stack

| Component               | Tech Used                              |
|------------------------|-----------------------------------------|
| Object Detection        | YOLOv8                                  |
| Motion Tracking         | Custom CentroidTracker                  |
| Scene Classifier        | ResNet18 + TSM + CBAM + MHA             |
| Video Processing        | OpenCV, FFmpeg                          |
| Heatmap Visualization   | Grad-CAM++, Matplotlib                  |
| Backend/API             | FastAPI                                 |
| Deployment Format       | ONNX, MP4, JSON                         |

---

##  Results

- **99% accuracy** on test set (1,000+ crash/no-crash clips)
- Detects edge cases missed by motion-only or scene-only models
- Auto-tags: rear-end, side-impact, angled collisions
- Generates structured reports (with velocity & class info)

---

##  Usage

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Run Detection on a Video

```bash
python main.py
```

Modify the paths in `main.py` to point to your input `.mp4` file and output directory.

### 3. Output Files

After processing, you'll get:

- `output_full.mp4` – Full annotated video  
- `*_accident_*.mp4` – Short clips for detected events  
- `collision_report.txt` – Human-readable crash logs  
- `attention_summary.png` – CNN attention visualization  

---

##  Real-World Applications

- **Traffic Surveillance:** Monitor intersections for accidents in real time  
- **Fleet Monitoring:** Detect high-risk behavior across company vehicles  
- **Insurance:** Automatic documentation for claims  
- **Smart Cities:** Real-time road safety analysis  
- **Research:** Study vehicle dynamics in real-world crashes  

---

##  Future Improvements

- Integrate LSTM/GRU for longer temporal context  
- Expand crash taxonomy (e.g., near-miss, pedestrian events)  
- Optimize for edge deployment 

---
