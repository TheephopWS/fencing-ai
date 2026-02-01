# Fencing AI App

A real-time fencing analysis application that uses AI-powered pose estimation to detect and track fencers' movements, identify attacking patterns, and provide visual feedback during fencing matches.

## Overview

This application analyzes fencing videos or live camera feeds using MediaPipe's pose estimation to:
- Track two fencers simultaneously
- Detect attacking movements based on footwork patterns
- Display real-time scoring overlay
- Process and annotate fencing videos

## Features

- **Pose Estimation**: Uses MediaPipe pose landmarker to track 33 body keypoints per fencer
- **Attack Detection**: Analyzes foot positioning and movement to determine which fencer is attacking
- **Video Processing**: Upload and process fencing match videos with annotations
- **Real-time Analysis**: Support for live camera feeds and IP camera streams
- **Streamlit Web Interface**: User-friendly web UI for video upload and processing

## Project Structure

```
fencing-v1/
├── app.py                          # Main Streamlit web application
├── ProcessVid.py                   # Video processing pipeline
├── requirements.txt                # Python dependencies
├── landmark.json                   # Pose landmark configuration
├── PoseEstimation/
│   ├── Training.py                 # Main pose estimation and analysis logic
│   ├── PoseEstimator.py           # OpenPose-based estimator (alternative)
│   ├── pose_landmarker_full.task  # MediaPipe pose model
│   ├── OpenPoseEstimator.ipynb    # Jupyter notebook for experimentation
│   └── models/                     # Model files (graph_opt.pb, etc.)
├── utils/
│   ├── Camera.py                   # Camera utility functions
│   └── ClearFiles.py              # File cleanup utilities
└── uploads/                        # Directory for uploaded/processed videos
```

## Prerequisites

- Python 3.8 or higher
- Webcam or IP camera (for live analysis)
- CUDA-capable GPU (optional, for faster processing with OpenPose)

## Installation

### 1. Clone or Download the Repository

```
git clone https://github.com/TheephopWS/fencing-ai.git
```

### 2. Create a Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The required packages include:
- `streamlit` - Web interface framework
- `opencv-python` - Computer vision and video processing
- `numpy` - Numerical computations
- `torch` - PyTorch for deep learning models
- `scipy` - Scientific computing
- `matplotlib` - Visualization
- `mediapipe` - Google's pose estimation framework

## Usage

### Running the Streamlit Web App

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the Web Interface**:
   - Open your browser and navigate to `http://localhost:8501`

3. **Upload and Process Videos**:
   - Click "Browse files" to upload an MP4 video file
   - Wait for processing to complete
   - View the annotated video with pose landmarks and attack indicators
   - Use "Delete Old Files" button to clear the uploads directory

### Running Real-time Analysis

For live camera analysis, you can run the Training.py script directly:

```bash
python PoseEstimation/Training.py
```

**Note**: By default, it attempts to connect to an IP camera at `http://172.27.132.75:8080/video`. To use a different source:

Edit [PoseEstimation/Training.py](PoseEstimation/Training.py#L232):
```python
# For local webcam (camera index 0):
cap = cv2.VideoCapture(0)

# For IP camera:
cap = cv2.VideoCapture("http://YOUR_IP:PORT/video")
```

Press 'q' to quit the live analysis window.

## Configuration

### Model Settings

In [PoseEstimation/Training.py](PoseEstimation/Training.py#L8-L12), you can adjust:

```python
NUM_POSES = 2                           # Number of people to track
MIN_POSE_DETECTION_CONFIDENCE = 0.5     # Detection confidence threshold
MIN_POSE_PRESENCE_CONFIDENCE = 0.5      # Presence confidence threshold
MIN_TRACKING_CONFIDENCE = 0.5           # Tracking confidence threshold
```

### Video Settings

```python
WIDTH = 640      # Frame width
HEIGHT = 480     # Frame height
start_time_seconds = 180  # Match duration in seconds
```

## How It Works

### Attack Detection Algorithm

The system analyzes foot positioning to determine attacks:

1. **Distance Calculation**: Measures distances between:
   - Lead foot and lead hip
   - Follow foot and follow hip
   - Distance between both feet

2. **Movement Analysis**: Compares current frame with previous frame to detect:
   - Forward foot movement
   - Stance expansion/compression
   - Direction of attack

3. **Attacker Identification**: 
   - Left fencer (index 0) or Right fencer (index 1)
   - Displays "Attack" indicator on the appropriate side
   - Manages attack buffer to prevent rapid switching

### Visual Overlay

The processed video includes:
- Pose landmarks (33 keypoints) for both fencers
- Skeletal connections between landmarks
- Score display (0:0 format)
- Timer countdown
- "Attack" indicator (yellow text) for the attacking fencer

## Camera Utilities

To check available cameras on your system:

```bash
python utils/Camera.py
```

This will list all available camera indices that can be used with `cv2.VideoCapture()`.

## Troubleshooting

### Common Issues

1. **ImportError for mediapipe**:
   - Ensure mediapipe is properly installed: `pip install mediapipe --upgrade`

2. **Model file not found**:
   - Verify `pose_landmarker_full.task` exists in the PoseEstimation directory

3. **Camera not opening**:
   - Check camera index in Training.py
   - Run Camera.py to find available cameras
   - Ensure camera permissions are granted

4. **Slow processing**:
   - Reduce video resolution in Training.py
   - Lower confidence thresholds slightly
   - Use GPU acceleration if available

5. **Streamlit not starting**:
   ```bash
   pip install streamlit --upgrade
   streamlit run app.py
   ```

## Development

### Adding New Features

- **Custom pose analysis**: Modify the `get_distance_info()` and `get_attacking()` functions in Training.py
- **Different sports**: Adjust landmark indices and distance calculations for other sports
- **Enhanced UI**: Customize the Streamlit interface in app.py

### Model Files

The project supports two pose estimation approaches:
1. **MediaPipe** (default): Fast, accurate, runs on CPU
2. **OpenPose**: More detailed, requires GPU, uses TensorFlow model in `models/graph_opt.pb`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

