# Taekwondo Pose Detection

**A Computer Vision Project for Human Pose Estimation and Taekwondo Movement Analysis**

## Overview

Taekwondo Pose Detection is a Python-based computer vision project designed to detect and visualize human body landmarks in Taekwondo videos and through a live webcam.

The application uses **MediaPipe Pose Landmarker** for human pose estimation and **OpenCV** for video processing. A graphical user interface built with Tkinter allows users to interact with the application.

## Features

- **Graphical User Interface:** A desktop interface for interacting with the application.
- **Video Processing:** Select and process video files.
- **Pose Estimation:** Detect human body landmarks in video frames.
- **Skeleton Visualization:** Display detected body landmarks and pose connections.
- **Live Webcam Detection:** Visualize human body pose landmarks in real time using a webcam.
- **Machine Learning Pipeline:** Includes a Random Forest training script and saved model artifacts.

> **Current Status:** The application supports pose estimation and skeleton visualization. Automatic Taekwondo movement classification and real-time movement recognition are not yet implemented.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| MediaPipe | Human pose estimation |
| OpenCV | Image and video processing |
| NumPy | Numerical computing |
| Scikit-learn | Machine learning |
| Joblib | Model serialization |
| Tkinter | Graphical user interface |
| Pillow | Image handling |

## Project Structure

```text
Taekwondo-Pose-Detection/
│
├── models/
│   └── pose_landmarker_full.task
│
├── taekwondo_ai_interface.py
├── predict_video.py
├── train_final_model.py
├── taekwondo_rf_final.pkl
├── class_names.json
├── model_info.json
├── selected_features.npy
├── X_temporal.npy
├── y.npy
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/M-Mahmoudy/Taekwondo-Pose-Detection.git
cd Taekwondo-Pose-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

To launch the graphical application:

```bash
python taekwondo_ai_interface.py
```

After launching the application, you can use the interface to select a video or access the webcam pose estimation feature.

## Machine Learning

The project includes a Random Forest model and related training artifacts.

The current application primarily focuses on human pose estimation and skeleton visualization. Movement classification is a planned development stage.

## Dataset

The original Taekwondo video dataset is not included in this repository.

## Future Improvements

- Automatic Taekwondo movement classification
- Real-time movement recognition
- Improved classification accuracy
- Expanded training dataset
- Performance optimization

## License

No license has been specified for this project yet.