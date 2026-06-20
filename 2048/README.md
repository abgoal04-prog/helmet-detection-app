# Construction Site Safety Monitor

A Streamlit-based AI-powered construction site safety monitoring system that uses YOLO computer vision to detect Personal Protective Equipment (PPE) violations in real-time.

[![Deploy to Streamlit Cloud](https://deploy.streamlit.io/badge.svg)](https://deploy.streamlit.io/)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new)

## Features

- Real-time PPE detection using YOLO models
- Video upload and processing
- Live webcam monitoring
- Violation tracking and reporting
- Interactive dashboard with statistics
- Cloud-ready deployment

## Quick Start

### Clone and Run Locally

```bash
git clone https://github.com/your-username/construction-safety-monitor.git
cd construction-safety-monitor
pip install -r requirements.txt
streamlit run app.py
```

The application will automatically create all required directories (`models/`, `uploads/`, `outputs/`, `data/`) on first run.

### Open in GitHub Codespaces

Click the "Open in GitHub Codespaces" button above to launch a fully configured development environment in your browser.

### Deploy to Streamlit Cloud

#### Step-by-Step Deployment Instructions

1. **Push your repository to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/your-username/construction-safety-monitor.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repository
   - Select `app.py` as the main file
   - Click "Deploy"

The app will automatically install dependencies from `requirements.txt` and start running on Streamlit Cloud.

## Custom Models

You can add custom YOLO PPE detection models by placing them in the `models/` directory:

- Place your custom model file as `models/best.pt`
- The application will automatically load this model if available
- If no custom model is found, the app will use YOLOv8n as a demo model

**Note:** Large model files are ignored by git by default. If you want to commit custom models, use [Git LFS](https://git-lfs.github.com/).

## Project Structure

```
construction-safety-monitor/
├── .streamlit/          # Streamlit configuration
├── .devcontainer/       # GitHub Codespaces configuration
├── models/              # YOLO model files
├── uploads/             # Uploaded video files
├── outputs/             # Processed output files
├── data/                # Database files
├── app.py               # Main Streamlit application
├── detector.py          # YOLO detection logic
├── database.py          # SQLite database operations
└── video_processor.py   # Video processing logic
```

## Requirements

- Python 3.11+
- See `requirements.txt` for full dependencies

## Troubleshooting

### Common Deployment Issues

**Issue:** ModuleNotFoundError on Streamlit Cloud
- **Solution:** Ensure all dependencies are listed in `requirements.txt` with pinned versions

**Issue:** Database locked error on cloud
- **Solution:** The app handles this gracefully with cloud-safe database operations

**Issue:** Model not loading
- **Solution:** The app falls back to YOLOv8n demo model if custom model is missing

**Issue:** Webcam not working on Streamlit Cloud
- **Solution:** Webcam requires HTTPS. Streamlit Cloud provides this automatically. The app falls back to camera input if webrtc fails

**Issue:** Video processing timeout on free tier
- **Solution:** Videos are automatically truncated at 500 frames to avoid timeout. Upgrade to paid tier for longer videos

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
