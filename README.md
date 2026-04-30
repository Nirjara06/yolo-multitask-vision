# 🎯 YOLO Multi-Task Vision System

> **Deep Learning Lab Assignment 3** — Multi-Task Computer Vision using YOLOv8
> MIT Academy of Engineering, Pune | Department of Computer Software Engineering 

---

## 👤 Student Details

| Field | Details |
|---|---|
| **Name** | Nirjara More |
| **PRN** | 202301100049 |
| **Subject** | Deep Learning|
| **Semester** | VI |


---

## 📌 Objective

Design and implement a **multi-task computer vision system** using YOLOv8 (nano/lightweight models) to perform:

- **Part A** — Object Detection (Vehicle Detection)
- **Part B** — Image Classification (Plant Disease Classification)
- **Part C** — Pose Estimation (Human Pose Estimation)
- **Part D** — Oriented Bounding Boxes / OBB (Text Detection)
- **Deployment** — Flask API + Streamlit Web App on localhost

---

## 💻 System Configuration

| Field | Details |
|---|---|
| **Python** | 3.11.13 |
| **PyTorch** | 2.11.0 (MPS) |
| **Ultralytics** | 8.4.36 |
| **Training Device** | Apple MPS (Metal Performance Shaders) |

> ⚠️ All training was performed **locally** on a personal machine. No cloud platforms (Colab/Kaggle) were used.

---

## 📊 Results Summary

| Part | Task | Model | Dataset | Key Metric |
|---|---|---|---|---|
| A | Vehicle Detection | `yolov8n` | vehicles.v1i.yolov8 | mAP50: **0.009** |
| B | Plant Disease Classification | `yolov8n-cls` | Plant Disease.v1i.folder | Top-1 Acc: **100%** |
| C | Human Pose Estimation | `yolov8n-pose` | Human Pose.v1i.yolov8 | mAP50: **0.955** |
| D | Text Detection OBB | `yolov8n-obb` | Text Detection.v1i.yolov8-obb | mAP50: **0.819** |

---

## 📁 Project Structure

```
yolo_multitask/
│
├── train_detection.py          # Part A — Vehicle Detection training script
├── train_classification.py     # Part B — Plant Disease Classification training
├── train_pose.py               # Part C — Human Pose Estimation training
├── train_obb.py                # Part D — Text Detection OBB training
├── run_all_tasks.py            # Combined runner for all 4 parts
│
├── app_streamlit.py            # Streamlit web app for inference UI
│
├── deployment/
│   └── app.py                  # Flask REST API for localhost deployment
│
├── partA_object_detection.ipynb
├── partB_classification.ipynb
├── partC_pose_estimation.ipynb
├── partD_obb.ipynb
├── partE_deployment.ipynb
│
├── models/                     # ⚠️ Not pushed (file size too large)
│   ├── yolov8n.pt
│   ├── yolov8n-cls.pt
│   ├── yolov8n-pose.pt
│   └── yolov8n-obb.pt
│
├── datasets/                   # ⚠️ Not pushed (sourced from Roboflow)
│
├── runs/                       # ⚠️ Not pushed (generated during training)
│
└── README.md
```

---

## 🗂️ Datasets (Roboflow)

All datasets were sourced from **Roboflow** in YOLO-compatible export format.

| Part | Dataset | Roboflow Link |
|---|---|---|
| A — Detection | vehicles.v1i.yolov8 | https://universe.roboflow.com |
| B — Classification | Plant Disease.v1i.folder | https://universe.roboflow.com |
| C — Pose | Human Pose.v1i.yolov8 | https://universe.roboflow.com |
| D — OBB | Text Detection.v1i.yolov8-obb | https://universe.roboflow.com |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Lab-Assignment-3--Yolo-Model.git
cd Lab-Assignment-3--Yolo-Model
```

### 2. Create Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install ultralytics flask streamlit opencv-python pillow
```

---

## 🏋️ Training

Run each part individually:

```bash
# Part A — Object Detection
python train_detection.py

# Part B — Classification
python train_classification.py

# Part C — Pose Estimation
python train_pose.py

# Part D — OBB
python train_obb.py

# OR run all parts at once
python run_all_tasks.py
```

---

## 🚀 Deployment

### Option 1 — Flask REST API (Localhost)

```bash
cd deployment
python app.py
```

API runs at: `http://localhost:5000`

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/predict` | POST | Send image → get detections as JSON |

**Test the API:**
```bash
curl -X POST http://localhost:5000/predict \
     -F "image=@test_image.jpg"
```

**Sample Response:**
```json
{
  "total": 3,
  "detections": [
    {
      "class": "car",
      "confidence": 0.874,
      "bbox": [120.5, 230.1, 400.2, 560.8]
    }
  ]
}
```

### Option 2 — Streamlit Web App

```bash
streamlit run app_streamlit.py
```

Live deployment: **https://yolo-multitask-vision.streamlit.app/**

---

## 🧠 Model Architecture

| Model | Task | Parameters | Input Size |
|---|---|---|---|
| `yolov8n` | Detection | ~3.2M | 640×640 |
| `yolov8n-cls` | Classification | ~2.7M | 224×224 |
| `yolov8n-pose` | Pose Estimation | ~3.3M | 640×640 |
| `yolov8n-obb` | Oriented BBox | ~3.1M | 640×640 |

All models use the **YOLOv8 Nano** variant — optimized for speed and lightweight deployment.

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `partA_object_detection.ipynb` | Step-by-step detection training, validation, inference |
| `partB_classification.ipynb` | Classification training with Top-1/Top-5 accuracy |
| `partC_pose_estimation.ipynb` | Pose training with 17-keypoint skeleton visualization |
| `partD_obb.ipynb` | OBB training on aerial/text data with rotated box output |
| `partE_deployment.ipynb` | Flask API deployment with live curl + requests testing |

---

## 🔑 Key Concepts

**Part A — Object Detection**
- YOLO performs detection in a single forward pass
- Outputs: bounding boxes, class labels, confidence scores
- Metric: mAP50 (mean Average Precision at IoU 0.5)

**Part B — Image Classification**
- Assigns a single class label to the whole image
- No bounding boxes — pure image-level prediction
- Metric: Top-1 and Top-5 accuracy

**Part C — Pose Estimation**
- Detects 17 human body keypoints (nose, eyes, shoulders, elbows, wrists, hips, knees, ankles)
- Each keypoint has (x, y) coordinates + confidence
- Metric: OKS-based mAP (Object Keypoint Similarity)

**Part D — Oriented Bounding Boxes (OBB)**
- Regular boxes are axis-aligned; OBB adds rotation angle θ
- Defined as: `(cx, cy, width, height, θ)`
- Essential for tilted/rotated objects like aerial vehicles or text lines

---

## 🛠️ Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/) with Apple MPS backend
- [Flask](https://flask.palletsprojects.com/) — REST API
- [Streamlit](https://streamlit.io/) — Web UI
- [Roboflow](https://roboflow.com/) — Dataset sourcing
- [OpenCV](https://opencv.org/) — Image processing
- [Pillow](https://pillow.readthedocs.io/) — Image I/O

---

## ⚠️ Notes

- `models/` folder is not pushed due to file size. Download base models via `ultralytics` or from the [Ultralytics releases](https://github.com/ultralytics/assets/releases).
- `datasets/` folder is not pushed. Download from Roboflow using the links above.
- `runs/` folder is generated locally during training.

---

## 📜 References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [Roboflow Universe](https://universe.roboflow.com/)
- [COCO Dataset](https://cocodataset.org/)
- [DOTA Dataset](https://captain-whu.github.io/DOTA/)

---

*Lab Assignment 3 | Deep Learning | MIT Academy of Engineering, Pune*
