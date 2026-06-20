# Helmet Detection Monitoring System

A real-time helmet detection system using YOLOv11, OpenCV, and Gradio for industrial safety monitoring.

## Features

| Feature | Description |
|---------|-------------|
| **Real-time Detection** | Detects helmets and no-helmet violations in real-time using YOLOv11 |
| **Multi-format Support** | Process images, videos, and live webcam feeds |
| **Alarm System** | Thread-safe audio alarm with cooldown period for violations |
| **Violation Logging** | Automatic CSV logging of all violations with timestamps |
| **Industrial Dashboard** | Professional Gradio interface with status indicators and statistics |
| **Flexible Class Names** | Supports multiple dataset naming conventions (helmet/hardhat, etc.) |
| **Confidence Thresholding** | Configurable confidence threshold (default: 0.40) |
| **Visual Feedback** | Color-coded bounding boxes (green for helmet, red for no-helmet) |

## Folder Structure

```
helmet-detection/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── violations.csv      # Violation log file
├── README.md           # This file
├── .gitignore          # Git ignore rules
├── outputs/            # Processed video outputs
│   └── .gitkeep
├── uploads/            # Temporary upload storage
│   └── .gitkeep
└── assets/             # Static assets (logos, etc.)
    └── .gitkeep
```

## Prerequisites

Before running the application, you must add the following files to the root directory:

- **best.pt** - Your trained YOLOv11 model file
- **alarm.wav** - Audio file for violation alerts

## Local Setup Instructions

### 1. Navigate to the project directory

```bash
cd helmet-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add your model and alarm files

Place `best.pt` and `alarm.wav` in the root `helmet-detection/` directory.

### 6. Run the application

```bash
python app.py
```

### 7. Access the web interface

Open your browser and navigate to:
```
http://localhost:7860
```

## GitHub Upload Instructions

### 1. Initialize Git repository

```bash
git init
```

### 2. Configure Git LFS for large files

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.wav"
git lfs track "*.mp4"
```

### 3. Add all files

```bash
git add .
git add .gitattributes
```

### 4. Commit changes

```bash
git commit -m "Initial commit: Helmet Detection Monitoring System"
```

### 5. Create repository on GitHub

Create a new repository on GitHub (don't initialize with README)

### 6. Push to GitHub

```bash
git remote add origin https://github.com/your-username/helmet-detection.git
git branch -M main
git push -u origin main
```

## Hugging Face Spaces Deployment

### 1. Create a new Space

- Go to [Hugging Face Spaces](https://huggingface.co/spaces)
- Click "Create new Space"
- Choose "Gradio" as the SDK
- Name your space (e.g., `helmet-detection`)

### 2. Create `README.md` with YAML header

Add this to the top of your README.md:

```yaml
---
title: Helmet Detection Monitoring System
emoji: ⛑️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---
```

### 3. Upload files to Hugging Face

You can either:
- Use the Hugging Face web interface to upload files
- Use Git to push to your Space

**Using Git:**
```bash
git clone https://huggingface.co/spaces/your-username/helmet-detection
cd helmet-detection
# Copy your files here
git add .
git commit -m "Initial deployment"
git push
```

### 4. Upload large files using Hugging Face CLI

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload your-username/helmet-detection best.pt ./best.pt
huggingface-cli upload your-username/helmet-detection alarm.wav ./alarm.wav
```

### 5. Your Space will automatically build and deploy

Access your deployed app at: `https://huggingface.co/spaces/your-username/helmet-detection`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Model not found error** | Ensure `best.pt` is in the root directory. Check the file path in `app.py`. |
| **Alarm not playing** | Verify `alarm.wav` exists. On Windows, ensure audio drivers are working. Try using a different audio format. |
| **Webcam not working** | Check browser permissions. Ensure no other application is using the camera. Try a different browser. |
| **Video processing slow** | Reduce video resolution or use a smaller model. Adjust `CONF_THRESHOLD` to filter low-confidence detections. |
| **Import errors** | Ensure all dependencies are installed: `pip install -r requirements.txt`. Check Python version (3.8+ recommended). |
| **Gradio interface not loading** | Check if port 7860 is available. Try changing `server_port` in `app.py`. |
| **CSV logging errors** | Ensure `violations.csv` has write permissions. Check that the directory exists. |
| **Memory errors** | Reduce video resolution or process shorter video segments. Close other applications. |
| **False detections** | Adjust `CONF_THRESHOLD` in `app.py` (increase to reduce false positives). Retrain model with better data. |
| **Class name not recognized** | Add your class name to `HELMET_ALIASES` or `NO_HELMET_ALIASES` in `app.py`. |

## Configuration

You can modify these constants in `app.py`:

- `CONF_THRESHOLD` - Minimum confidence for detections (default: 0.40)
- `ALARM_COOLDOWN` - Seconds between alarm triggers (default: 3.0)
- `GREEN` and `RED` - BGR color values for bounding boxes
- `HELMET_ALIASES` and `NO_HELMET_ALIASES` - Class name mappings

## Supported Class Name Conventions

The system automatically normalizes various class name formats:

**Helmet classes:**
- helmet, hardhat, with helmet, with_helmet, head, person_with_helmet

**No-helmet classes:**
- no_helmet, no-helmet, without helmet, without_helmet, no_hardhat, no-hardhat, person_without_helmet

## License

MIT License - Feel free to use this project for your safety monitoring needs.

## Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) - Object detection model
- [Gradio](https://gradio.app/) - Web interface framework
- [OpenCV](https://opencv.org/) - Computer vision library
