"""
Helmet Detection Monitoring System
A real-time helmet detection system using YOLOv11, OpenCV, and Gradio
"""

import os
import cv2
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional

import gradio as gr
from ultralytics import YOLO

# ==================== CONFIGURATION CONSTANTS ====================
# File paths
MODEL_PATH = "best.pt"
ALARM_PATH = "alarm.wav"
VIOLATIONS_CSV = "violations.csv"
OUTPUTS_DIR = "outputs"
UPLOADS_DIR = "uploads"

# Detection thresholds
CONF_THRESHOLD = 0.40

# Colors for bounding boxes (BGR format)
GREEN = (0, 200, 0)      # Helmet detected
RED = (0, 0, 255)        # No helmet detected

# CSV headers
CSV_HEADERS = ["Date", "Time", "Detection Type", "Confidence Score", "Status"]

# Class name aliases for different datasets
# Maps various possible class names to standardized "helmet" or "no_helmet"
HELMET_ALIASES = {
    "helmet", "hardhat", "with helmet", "with_helmet", "head", "person_with_helmet"
}

NO_HELMET_ALIASES = {
    "no_helmet", "no-helmet", "without helmet", "without_helmet", 
    "no_hardhat", "no-hardhat", "person_without_helmet"
}

# ==================== MODEL LOADING ====================
def load_model():
    """
    Load YOLO model from best.pt with error handling.
    Returns the model if successful, None otherwise.
    """
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        print(f"⚠️  WARNING: Could not load model from {MODEL_PATH}")
        print(f"Error: {e}")
        print("Please ensure best.pt is in the root directory.")
        return None

# Load the model at startup
model = load_model()
model_warning = None if model is not None else """
<div style='background-color: #ffcccc; padding: 15px; border-radius: 5px; border: 2px solid #ff0000;'>
    <h3 style='color: #cc0000; margin: 0;'>⚠️ MODEL NOT FOUND</h3>
    <p style='color: #cc0000; margin: 5px 0 0 0;'>Please add your <strong>best.pt</strong> file to the root directory.</p>
</div>
"""

# ==================== ALARM CONTROLLER ====================
class AlarmController:
    """
    Thread-safe alarm controller that plays alarm sounds without overlapping.
    Supports both playsound and Windows winsound.
    """
    
    def __init__(self):
        self.is_playing = False
        self.lock = threading.Lock()
        self.alarm_file = ALARM_PATH
    
    def play_alarm(self):
        """
        Play the alarm sound in a separate thread to avoid blocking.
        Will not overlap if already playing.
        """
        with self.lock:
            if self.is_playing or not os.path.exists(self.alarm_file):
                return
            
            self.is_playing = True
        
        # Start alarm in a separate thread
        thread = threading.Thread(target=self._play_alarm_thread, daemon=True)
        thread.start()
    
    def _play_alarm_thread(self):
        """Internal method to play alarm in a thread."""
        try:
            # Try Windows winsound first (more reliable on Windows)
            try:
                import winsound
                winsound.PlaySound(self.alarm_file, winsound.SND_FILENAME)
            except (ImportError, Exception):
                # Fallback to playsound
                from playsound import playsound
                playsound(self.alarm_file)
        except Exception as e:
            print(f"Error playing alarm: {e}")
        finally:
            with self.lock:
                self.is_playing = False
    
    def reset(self):
        """Reset the alarm state."""
        with self.lock:
            self.is_playing = False

# Create global alarm controller instance
alarm_controller = AlarmController()

# ==================== LOGGING FUNCTIONS ====================
def log_violation(confidence: float):
    """
    Log a no-helmet detection to the violations CSV file.
    
    Args:
        confidence: Confidence score of the detection (0-1)
    """
    try:
        now = datetime.now()
        date_str = now.strftime("%d-%m-%Y")
        time_str = now.strftime("%H:%M:%S")
        
        # Format confidence as percentage
        conf_percent = f"{confidence:.2f}"
        
        # Create the log entry
        log_entry = f"{date_str},{time_str},No Helmet,{conf_percent},UNSAFE\n"
        
        # Append to CSV
        with open(VIOLATIONS_CSV, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Error logging violation: {e}")

def load_violations_log() -> pd.DataFrame:
    """
    Load the violations log into a pandas DataFrame.
    Returns the log sorted with newest entries first.
    
    Returns:
        DataFrame with violation records
    """
    try:
        if os.path.exists(VIOLATIONS_CSV):
            df = pd.read_csv(VIOLATIONS_CSV)
            if not df.empty:
                df = df.iloc[::-1]  # Reverse to show newest first
            return df
        else:
            # Create empty DataFrame with correct columns
            return pd.DataFrame(columns=CSV_HEADERS)
    except Exception as e:
        print(f"Error loading violations log: {e}")
        return pd.DataFrame(columns=CSV_HEADERS)

def clear_violations_log():
    """Clear the violations CSV file (keep only header)."""
    try:
        with open(VIOLATIONS_CSV, "w", encoding="utf-8") as f:
            f.write(",".join(CSV_HEADERS) + "\n")
        return True
    except Exception as e:
        print(f"Error clearing violations log: {e}")
        return False

# ==================== CLASS NAME NORMALIZATION ====================
def normalize_class_name(class_name: str) -> Optional[str]:
    """
    Normalize model class names into "helmet" or "no_helmet".
    Works with various dataset naming conventions.
    
    Args:
        class_name: The raw class name from the model
        
    Returns:
        "helmet", "no_helmet", or None if not recognized
    """
    class_name_lower = class_name.lower().strip()
    
    # Check against helmet aliases
    for alias in HELMET_ALIASES:
        if alias in class_name_lower or class_name_lower in alias:
            return "helmet"
    
    # Check against no-helmet aliases
    for alias in NO_HELMET_ALIASES:
        if alias in class_name_lower or class_name_lower in alias:
            return "no_helmet"
    
    # If not recognized, return None
    return None

# ==================== DETECTION FUNCTIONS ====================
def detect_on_frame(frame: np.ndarray, model: YOLO) -> Tuple[np.ndarray, Dict, bool]:
    """
    Run YOLO detection on a single frame.
    
    Args:
        frame: Input frame (numpy array)
        model: YOLO model instance
        
    Returns:
        Tuple of (annotated_frame, stats_dict, is_unsafe)
        - annotated_frame: Frame with bounding boxes drawn
        - stats_dict: Dictionary with detection statistics
        - is_unsafe: Boolean indicating if any no-helmet detected
    """
    if model is None:
        return frame, {"persons": 0, "helmet": 0, "no_helmet": 0, "violations": 0}, False
    
    # Run YOLO inference
    results = model(frame, verbose=False)
    
    # Initialize counters
    helmet_count = 0
    no_helmet_count = 0
    total_persons = 0
    is_unsafe = False
    
    # Process detections
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get confidence and class
                conf = float(box.conf)
                class_id = int(box.cls)
                class_name = model.names[class_id]
                
                # Filter by confidence threshold
                if conf < CONF_THRESHOLD:
                    continue
                
                # Normalize class name
                normalized = normalize_class_name(class_name)
                
                if normalized == "helmet":
                    helmet_count += 1
                    total_persons += 1
                    color = GREEN
                    label = f"Helmet {conf:.2f}"
                elif normalized == "no_helmet":
                    no_helmet_count += 1
                    total_persons += 1
                    color = RED
                    label = f"No Helmet {conf:.2f}"
                    is_unsafe = True
                else:
                    # Unrecognized class, skip
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Create stats dictionary
    stats = {
        "persons": total_persons,
        "helmet": helmet_count,
        "no_helmet": no_helmet_count,
        "violations": no_helmet_count
    }
    
    return frame, stats, is_unsafe

# ==================== PROCESSING FUNCTIONS ====================
def process_image(image_path: str) -> Tuple[np.ndarray, str, str]:
    """
    Process a single image for helmet detection.
    Returns ONLY the processed output image.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Tuple of (processed_image, status_html, stats_html)
    """
    if model is None:
        return None, model_warning, "<div style='color: red;'>Model not loaded</div>"
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, "<div style='color: red;'>Error: Could not read image</div>", ""
        
        # Run detection
        annotated_frame, stats, is_unsafe = detect_on_frame(image, model)
        
        # Generate status HTML
        if is_unsafe:
            status_html = """
            <div style='background-color: #ffcccc; padding: 20px; border-radius: 10px; border: 3px solid #ff0000; text-align: center;'>
                <h2 style='color: #cc0000; margin: 0;'>🚨 UNSAFE</h2>
                <p style='color: #cc0000; margin: 5px 0 0 0;'>No helmet detected!</p>
            </div>
            """
        else:
            status_html = """
            <div style='background-color: #ccffcc; padding: 20px; border-radius: 10px; border: 3px solid #00cc00; text-align: center;'>
                <h2 style='color: #006600; margin: 0;'>✅ SAFE</h2>
                <p style='color: #006600; margin: 5px 0 0 0;'>All persons wearing helmets</p>
            </div>
            """
        
        # Generate statistics HTML
        stats_html = f"""
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
            <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center;'>
                <h3 style='margin: 0; color: #333;'>Total Persons</h3>
                <p style='font-size: 24px; margin: 5px 0 0 0; color: #0066cc;'>{stats['persons']}</p>
            </div>
            <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center;'>
                <h3 style='margin: 0; color: #333;'>Helmet</h3>
                <p style='font-size: 24px; margin: 5px 0 0 0; color: #00cc00;'>{stats['helmet']}</p>
            </div>
            <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center;'>
                <h3 style='margin: 0; color: #333;'>No Helmet</h3>
                <p style='font-size: 24px; margin: 5px 0 0 0; color: #cc0000;'>{stats['no_helmet']}</p>
            </div>
            <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center;'>
                <h3 style='margin: 0; color: #333;'>Violations</h3>
                <p style='font-size: 24px; margin: 5px 0 0 0; color: #cc0000;'>{stats['violations']}</p>
            </div>
        </div>
        """
        
        # Log violations if any
        if stats['violations'] > 0:
            for _ in range(stats['violations']):
                log_violation(0.95)  # Use default confidence for logging
        
        # Convert BGR to RGB for Gradio
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        return annotated_frame, status_html, stats_html
    
    except Exception as e:
        error_html = f"<div style='color: red;'>Error: {str(e)}</div>"
        return None, error_html, ""

def process_video(video_path: str) -> Tuple[str, str, str]:
    """
    Process a video file for helmet detection.
    Writes output video to outputs directory.
    
    Args:
        video_path: Path to the input video
        
    Returns:
        Tuple of (output_video_path, status_html, stats_html)
    """
    if model is None:
        return None, model_warning, "<div style='color: red;'>Model not loaded</div>"
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "<div style='color: red;'>Error: Could not open video</div>", ""
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUTS_DIR, f"output_{timestamp}.mp4")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize counters
        total_violations = 0
        total_persons = 0
        total_helmet = 0
        total_no_helmet = 0
        
        # Process frames
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            annotated_frame, stats, is_unsafe = detect_on_frame(frame, model)
            
            # Update counters
            total_violations += stats['violations']
            total_persons += stats['persons']
            total_helmet += stats['helmet']
            total_no_helmet += stats['no_helmet']
            
            # Log violations
            if stats['violations'] > 0:
                for _ in range(stats['violations']):
                    log_violation(0.95)
                # Trigger alarm
                alarm_controller.play_alarm()
            
            # Write frame
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Progress update every 30 frames
                print(f"Processed {frame_count} frames...")
        
        # Release resources
        cap.release()
        out.release()
        
        # Reset alarm
        alarm_controller.reset()
        
        # Generate status HTML
        if total_violations > 0:
            status_html = f"""
            <div style='background-color: #ffcccc; padding: 20px; border-radius: 10px; border: 3px solid #ff0000; text-align: center;'>
                <h2 style='color: #cc0000; margin: 0;'>🚨 UNSAFE</h2>
                <p style='color: #cc0000; margin: 5px 0 0 0;'>{total_violations} violations detected!</p>
            </div>
            """
        else:
            status_html = """
            <div style='background-color: #ccffcc; padding: 20px; border-radius: 10px; border: 3px solid #00cc00; text-align: center;'>
                <h2 style='color: #006600; margin: 0;'>✅ SAFE</h2>
                <p style='color: #006600; margin: 5px 0 0 0;'>No violations detected</p>
            </div>
            """
        
        # Generate statistics HTML
        stats_html = f"""
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
            <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center;'>
                <h3 style='margin: 0; color: #333;'>Total Persons</h3>
                <p style='font-size: 24px; margin: 5px 0 0 0; color: #0066cc;'>{total_persons}</p>
            </div>
            <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center;'>
                <h3 style='margin: 0; color: #333;'>Helmet</h3>
                <p style='font-size: 24px; margin: 5px 0 0 0; color: #00cc00;'>{total_helmet}</p>
            </div>
            <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center;'>
                <h3 style='margin: 0; color: #333;'>No Helmet</h3>
                <p style='font-size: 24px; margin: 5px 0 0 0; color: #cc0000;'>{total_no_helmet}</p>
            </div>
            <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center;'>
                <h3 style='margin: 0; color: #333;'>Violations</h3>
                <p style='font-size: 24px; margin: 5px 0 0 0; color: #cc0000;'>{total_violations}</p>
            </div>
        </div>
        """
        
        return output_path, status_html, stats_html
    
    except Exception as e:
        error_html = f"<div style='color: red;'>Error: {str(e)}</div>"
        return None, error_html, ""

# Global variables for webcam cooldown
last_alarm_time = 0
ALARM_COOLDOWN = 3.0  # 3 seconds cooldown

def process_webcam_frame(frame: np.ndarray) -> Tuple[np.ndarray, str, str]:
    """
    Process a single webcam frame for real-time detection.
    Includes 3-second alarm cooldown.
    
    Args:
        frame: Input frame from webcam
        
    Returns:
        Tuple of (annotated_frame, status_html, stats_html)
    """
    global last_alarm_time
    
    if model is None:
        return frame, model_warning, "<div style='color: red;'>Model not loaded</div>"
    
    try:
        # Run detection
        annotated_frame, stats, is_unsafe = detect_on_frame(frame, model)
        
        # Handle alarm with cooldown
        current_time = time.time()
        if is_unsafe and (current_time - last_alarm_time) > ALARM_COOLDOWN:
            alarm_controller.play_alarm()
            last_alarm_time = current_time
            
            # Log violations
            for _ in range(stats['violations']):
                log_violation(0.95)
        
        # Generate status HTML
        if is_unsafe:
            status_html = """
            <div style='background-color: #ffcccc; padding: 20px; border-radius: 10px; border: 3px solid #ff0000; text-align: center;'>
                <h2 style='color: #cc0000; margin: 0;'>🚨 UNSAFE</h2>
                <p style='color: #cc0000; margin: 5px 0 0 0;'>No helmet detected!</p>
            </div>
            """
        else:
            status_html = """
            <div style='background-color: #ccffcc; padding: 20px; border-radius: 10px; border: 3px solid #00cc00; text-align: center;'>
                <h2 style='color: #006600; margin: 0;'>✅ SAFE</h2>
                <p style='color: #006600; margin: 5px 0 0 0;'>All persons wearing helmets</p>
            </div>
            """
        
        # Generate statistics HTML
        stats_html = f"""
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
            <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center;'>
                <h3 style='margin: 0; color: #333;'>Total Persons</h3>
                <p style='font-size: 24px; margin: 5px 0 0 0; color: #0066cc;'>{stats['persons']}</p>
            </div>
            <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center;'>
                <h3 style='margin: 0; color: #333;'>Helmet</h3>
                <p style='font-size: 24px; margin: 5px 0 0 0; color: #00cc00;'>{stats['helmet']}</p>
            </div>
            <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center;'>
                <h3 style='margin: 0; color: #333;'>No Helmet</h3>
                <p style='font-size: 24px; margin: 5px 0 0 0; color: #cc0000;'>{stats['no_helmet']}</p>
            </div>
            <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center;'>
                <h3 style='margin: 0; color: #333;'>Violations</h3>
                <p style='font-size: 24px; margin: 5px 0 0 0; color: #cc0000;'>{stats['violations']}</p>
            </div>
        </div>
        """
        
        # Convert BGR to RGB for Gradio
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        return annotated_frame, status_html, stats_html
    
    except Exception as e:
        error_html = f"<div style='color: red;'>Error: {str(e)}</div>"
        return frame, error_html, ""

# ==================== GRADIO INTERFACE ====================
def refresh_violations_table():
    """Refresh the violations table with latest data."""
    df = load_violations_log()
    return df

def create_interface():
    """
    Create and return the Gradio interface for the helmet detection system.
    """
    
    # Custom CSS for industrial safety theme
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .header h1 {
        color: white;
        margin: 0;
        font-size: 32px;
    }
    .logo-area {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .logo-icon {
        font-size: 48px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Helmet Detection Monitoring System") as interface:
        
        # Header with title and logo
        gr.HTML("""
        <div class="header">
            <div class="logo-area">
                <span class="logo-icon">⛑️</span>
                <h1>Helmet Detection Monitoring System</h1>
            </div>
        </div>
        """)
        
        # Warning banner if model not loaded
        if model_warning:
            gr.HTML(model_warning)
        
        with gr.Row():
            # LEFT PANEL: Detection tabs
            with gr.Column(scale=2):
                with gr.Tabs():
                    # Image Detection Tab
                    with gr.Tab("Image Detection"):
                        image_input = gr.Image(type="filepath", label="Upload Image")
                        image_button = gr.Button("Run Detection", variant="primary")
                        image_output = gr.Image(label="Detection Result")
                        image_status = gr.HTML()
                        image_stats = gr.HTML()
                        
                        image_button.click(
                            fn=process_image,
                            inputs=image_input,
                            outputs=[image_output, image_status, image_stats]
                        )
                    
                    # Video Detection Tab
                    with gr.Tab("Video Detection"):
                        video_input = gr.Video(label="Upload Video")
                        video_button = gr.Button("Process Video", variant="primary")
                        video_output = gr.Video(label="Processed Video")
                        video_status = gr.HTML()
                        video_stats = gr.HTML()
                        
                        video_button.click(
                            fn=process_video,
                            inputs=video_input,
                            outputs=[video_output, video_status, video_stats]
                        )
                    
                    # Live Webcam Tab
                    with gr.Tab("Live Webcam"):
                        webcam_button = gr.Button("Start Webcam", variant="primary")
                        webcam_output = gr.Image(label="Live Detection")
                        webcam_status = gr.HTML()
                        webcam_stats = gr.HTML()
                        
                        webcam_stream = gr.Image(
                            source="webcam",
                            streaming=True,
                            label="Live Webcam Feed"
                        )
                        
                        # Use streaming with frame processing
                        webcam_stream.stream(
                            fn=process_webcam_frame,
                            inputs=webcam_stream,
                            outputs=[webcam_output, webcam_status, webcam_stats],
                            stream_every=0.2
                        )
            
            # CENTER PANEL: Output display
            with gr.Column(scale=2):
                gr.Markdown("### Detection Output")
                # Output is shown in the tabs above
            
            # RIGHT PANEL: Status and Statistics
            with gr.Column(scale=1):
                gr.Markdown("### Live Status")
                live_status = gr.HTML()
                
                gr.Markdown("### Statistics")
                live_stats = gr.HTML()
        
        # BOTTOM PANEL: Violation History
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Violation History")
                with gr.Row():
                    refresh_button = gr.Button("Refresh", size="sm")
                    clear_button = gr.Button("Clear History", variant="stop", size="sm")
                
                violations_table = gr.Dataframe(
                    label="Violation Records",
                    headers=CSV_HEADERS,
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False
                )
                
                refresh_button.click(
                    fn=refresh_violations_table,
                    outputs=violations_table
                )
                
                clear_button.click(
                    fn=clear_violations_log,
                    outputs=[]
                ).then(
                    fn=refresh_violations_table,
                    outputs=violations_table
                )
        
        # Initialize violations table on load
        interface.load(
            fn=refresh_violations_table,
            outputs=violations_table
        )
    
    return interface

# ==================== MAIN ENTRY POINT ====================
if __name__ == "__main__":
    # Create interface
    interface = create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
