"""
AI-Powered Construction Site Safety Monitoring System
Uses YOLO to detect safety helmets and head violations
Features: Image/Video/Webcam detection, database logging, reporting
"""

import streamlit as st
from pathlib import Path
from datetime import datetime

# Core dependencies with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.error("OpenCV not installed. Please run: pip install opencv-python-headless")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    st.error("NumPy not installed. Please run: pip install numpy")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    st.error("Pandas not installed. Please run: pip install pandas")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Plotly not installed. Please run: pip install plotly")

# Module imports with error handling
try:
    from detector import SafetyDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False
    st.error("Detector module not found")

try:
    from database import DatabaseManager, DetectionDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    st.error("Database module not found")

try:
    from video_processor import VideoProcessor
    VIDEO_PROCESSOR_AVAILABLE = True
except ImportError:
    VIDEO_PROCESSOR_AVAILABLE = False
    st.error("Video processor module not found")

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Initialize paths using pathlib
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

# Auto-create all required folders on startup
MODELS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Streamlit page configuration
st.set_page_config(
    page_title="Construction Site Safety Monitor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
with open(PROJECT_ROOT / ".streamlit" / "style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def get_detector():
    """Initialize and cache the PPE detector"""
    return SafetyDetector(project_root=PROJECT_ROOT)

@st.cache_resource
def get_database():
    """Initialize and cache the database manager"""
    db_path = DATA_DIR / "safety_monitor.db"
    return DatabaseManager(db_path=str(db_path))

@st.cache_resource
def get_detection_database():
    """Initialize and cache the detection database"""
    return DetectionDatabase(project_root=PROJECT_ROOT)

@st.cache_resource
def get_video_processor():
    """Initialize and cache the video processor"""
    return VideoProcessor(project_root=PROJECT_ROOT)

# Initialize components
detector = get_detector()
db_manager = get_database()
detection_db = get_detection_database()
video_processor = get_video_processor()

# Initialize database tables
db_manager.initialize_database()

# Initialize all required session_state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'uploaded_video' not in st.session_state:
    st.session_state.uploaded_video = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'annotated_output' not in st.session_state:
    st.session_state.annotated_output = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'should_stop' not in st.session_state:
    st.session_state.should_stop = False
if 'safety_status' not in st.session_state:
    st.session_state.safety_status = "SITE SAFE"
if 'total_persons' not in st.session_state:
    st.session_state.total_persons = 0
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
if 'total_violations' not in st.session_state:
    st.session_state.total_violations = 0
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = "Image"
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None


def process_image_detection(image_file):
    """Process uploaded image for PPE detection"""
    if image_file is None:
        return None, "No image uploaded"
    
    # Read and process image
    image_bytes = image_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run detection
    annotated_pil, detections, safety_status, total_persons, total_violations = detector.detect_image(image)
    
    # Update session state
    st.session_state.annotated_output = annotated_pil
    st.session_state.detection_results = detections
    st.session_state.safety_status = safety_status
    st.session_state.total_persons = total_persons
    st.session_state.total_detections = len(detections)
    st.session_state.total_violations = total_violations
    
    # Save annotated image to outputs/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / f"annotated_{timestamp}.jpg"
    annotated_bgr = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), annotated_bgr)
    
    # Save to detection database
    record = {
        'input_type': 'image',
        'file_name': image_file.name,
        'safety_status': safety_status,
        'total_persons': total_persons,
        'total_detections': len(detections),
        'total_violations': total_violations,
        'detections': detections,
        'output_path': str(output_path)
    }
    detection_db.save_detection(record)
    
    # Convert PIL to numpy array for display
    annotated_image_rgb = np.array(annotated_pil)
    
    return annotated_image_rgb, f"Image processed successfully - {safety_status}"


def process_video_detection(video_file):
    """Process uploaded video for PPE detection"""
    if video_file is None:
        return None, "No video uploaded"
    
    st.session_state.is_running = True
    st.session_state.should_stop = False
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    try:
        # Process video with VideoProcessor
        output_path, total_dets, total_vios, total_pers, safety_status = video_processor.process_uploaded_video(
            video_file,
            progress_bar=progress_bar
        )
        
        # Update session state
        st.session_state.output_video_path = output_path
        st.session_state.total_detections = total_dets
        st.session_state.total_violations = total_vios
        st.session_state.total_persons = total_pers
        st.session_state.safety_status = safety_status
        st.session_state.annotated_output = output_path  # Store video path
        
        # Save to detection database
        record = {
            'input_type': 'video',
            'file_name': video_file.name,
            'safety_status': safety_status,
            'total_persons': total_pers,
            'total_detections': total_dets,
            'total_violations': total_vios,
            'detections': [],
            'output_path': output_path
        }
        detection_db.save_detection(record)
        
        return output_path, f"Video processed successfully - {safety_status}"
        
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None, f"Error: {e}"
    finally:
        st.session_state.is_running = False
        progress_bar.empty()


def process_webcam_detection():
    """Process webcam for real-time PPE detection"""
    st.session_state.webcam_active = True
    st.session_state.should_stop = False
    
    try:
        if WEBRTC_AVAILABLE:
            # Try streamlit-webrtc first (works on HTTPS)
            rtc_config = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            
            webrtc_ctx = webrtc_streamer(
                key="webcam",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=None,
                async_processing=True
            )
            
            if webrtc_ctx.state.playing:
                st.info("📷 Webcam active - Real-time detection enabled")
                
                # Display live safety status
                status_placeholder = st.empty()
                
                while not st.session_state.should_stop and webrtc_ctx.state.playing:
                    # Update live safety status
                    if st.session_state.total_violations > 0:
                        status_placeholder.error(f"🔴 LIVE - UNSAFE ({st.session_state.total_violations} violations)")
                    else:
                        status_placeholder.success("🟢 LIVE - SITE SAFE")
                    
                    st.session_state.is_running = True
                    st.time.sleep(0.5)
            
        else:
            # Fallback to st.camera_input
            st.warning("streamlit-webrtc not available, using camera input fallback")
            camera_image = st.camera_input("Take a photo for detection")
            
            if camera_image:
                # Process the captured image
                image_array = np.array(camera_image)
                annotated_pil, detections, safety_status, total_persons, total_violations = detector.detect_image(image_array)
                
                # Update session state
                st.session_state.annotated_output = annotated_pil
                st.session_state.detection_results = detections
                st.session_state.safety_status = safety_status
                st.session_state.total_persons = total_persons
                st.session_state.total_detections = len(detections)
                st.session_state.total_violations = total_violations
                
                # Save to detection database
                record = {
                    'input_type': 'webcam',
                    'file_name': f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    'safety_status': safety_status,
                    'total_persons': total_persons,
                    'total_detections': len(detections),
                    'total_violations': total_violations,
                    'detections': detections,
                    'output_path': None
                }
                detection_db.save_detection(record)
                
                st.image(annotated_pil, caption="Detection Result", use_container_width=True)
                st.success(f"Detection complete - {safety_status}")
                
    except Exception as e:
        st.error(f"Webcam error: {e}")
    finally:
        st.session_state.webcam_active = False
        st.session_state.is_running = False


def reset_detection():
    """Reset detection state"""
    st.session_state.uploaded_file = None
    st.session_state.uploaded_video = None
    st.session_state.detection_results = None
    st.session_state.annotated_output = None
    st.session_state.is_running = False
    st.session_state.should_stop = False
    st.session_state.safety_status = "SITE SAFE"
    st.session_state.total_persons = 0
    st.session_state.total_detections = 0
    st.session_state.total_violations = 0
    st.session_state.webcam_active = False
    st.session_state.output_video_path = None


def main():
    """Main application function"""
    
    # Check if all required dependencies are available
    if not all([CV2_AVAILABLE, NUMPY_AVAILABLE, PANDAS_AVAILABLE, PLOTLY_AVAILABLE, 
                DETECTOR_AVAILABLE, DATABASE_AVAILABLE, VIDEO_PROCESSOR_AVAILABLE]):
        st.error("❌ Missing required dependencies. Please install all required packages.")
        st.stop()
    
    # Display demo banner if using demo model
    demo_banner = detector.get_demo_banner_message()
    if demo_banner:
        st.warning(demo_banner)
    
    # Top navigation tabs
    tab1, tab2, tab3 = st.tabs(["Detection Dashboard", "History & Reports", "Overall Statistics"])
    
    # Detection Dashboard Tab
    with tab1:
        # Two equal columns
        col_left, col_right = st.columns(2)
        
        # LEFT COLUMN - Input Controls
        with col_left:
            st.markdown("### Input Controls")
            
            # Segmented input selector
            input_mode = st.segmented_control(
                "Input Type",
                ["Image", "Video", "Webcam"],
                selection_mode="single",
                default="Image"
            )
            st.session_state.input_mode = input_mode
            
            if input_mode == "Image":
                # Image upload area
                uploaded_file = st.file_uploader(
                    "Drop Image Here - or - Click to Upload",
                    type=["jpg", "jpeg", "png"],
                    help="Upload an image of a construction site"
                )
                st.session_state.uploaded_file = uploaded_file
                
            elif input_mode == "Video":
                # Video upload area
                uploaded_video = st.file_uploader(
                    "Drop Video Here - or - Click to Upload",
                    type=["mp4", "avi", "mov"],
                    help="Upload a video of a construction site"
                )
                st.session_state.uploaded_video = uploaded_video
                
            elif input_mode == "Webcam":
                st.info("Webcam feature - click to start live detection")
            
            st.markdown("---")
            
            # Buttons
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                start_btn = st.button("Start Detection", type="primary", use_container_width=True)
            
            with col_btn2:
                stop_btn = st.button("Stop Detection", use_container_width=True)
                if stop_btn:
                    st.session_state.should_stop = True
                    st.session_state.is_running = False
                    st.warning("Detection stopped")
            
            with col_btn3:
                reset_btn = st.button("Reset", use_container_width=True)
            
            if reset_btn:
                reset_detection()
                st.rerun()
            
            st.markdown("---")
            st.markdown("### Dashboard Statistics")
            
            # Safety status with colored dot
            if st.session_state.total_violations > 0:
                st.markdown("🔴 **UNSAFE**")
            else:
                st.markdown("🟢 **SITE SAFE**")
            
            # Statistics metrics
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Total Persons", st.session_state.total_persons)
            
            with col_stat2:
                st.metric("Total Detections", st.session_state.total_detections)
            
            with col_stat3:
                st.metric("Total Violations", st.session_state.total_violations, delta_color="inverse")
        
        # RIGHT COLUMN - Detection Output
        with col_right:
            st.markdown("### Detection Output")
            
            # Annotated Output placeholder
            if st.session_state.annotated_output is not None:
                # Check if it's a video path or image
                if isinstance(st.session_state.annotated_output, str) and st.session_state.annotated_output.endswith('.mp4'):
                    st.video(st.session_state.annotated_output, caption="Processed Video", use_container_width=True)
                else:
                    st.image(st.session_state.annotated_output, caption="Annotated Output", use_container_width=True)
            else:
                st.markdown("""
                <div style='border: 2px dashed #ccc; padding: 40px; text-align: center; border-radius: 10px;'>
                    <p style='color: #888; font-size: 16px;'>Annotated Output</p>
                    <p style='color: #aaa; font-size: 14px;'>Upload and process an image to see results</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detection Results table placeholder
            if st.session_state.detection_results is not None and len(st.session_state.detection_results) > 0:
                st.markdown("### Detection Results")
                df = pd.DataFrame(st.session_state.detection_results)
                st.dataframe(df, use_container_width=True)
            else:
                st.markdown("""
                <div style='border: 2px dashed #ccc; padding: 40px; text-align: center; border-radius: 10px;'>
                    <p style='color: #888; font-size: 16px;'>Detection Results</p>
                    <p style='color: #aaa; font-size: 14px;'>No detection results available</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Handle Start Detection button
        if start_btn:
            if input_mode == "Image" and st.session_state.uploaded_file:
                with st.spinner("Processing image..."):
                    result_image, message = process_image_detection(st.session_state.uploaded_file)
                    if result_image is not None:
                        st.success(message)
                        st.rerun()
            elif input_mode == "Video" and st.session_state.uploaded_video:
                with st.spinner("Processing video..."):
                    result_video, message = process_video_detection(st.session_state.uploaded_video)
                    if result_video is not None:
                        st.success(message)
                        st.rerun()
            elif input_mode == "Webcam":
                process_webcam_detection()
            else:
                st.warning("Please upload a file first")
    
    # History & Reports Tab
    with tab2:
        st.markdown("### History & Reports")
        
        # Model info card
        model_info_col1, model_info_col2 = st.columns(2)
        with model_info_col1:
            st.info(f"🤖 **Model**: {MODELS_DIR / 'best.pt' if (MODELS_DIR / 'best.pt').exists() else 'YOLOv8n (Demo)'}")
        with model_info_col2:
            if detector.is_demo_model:
                st.warning("⚠️ Using demo model - add custom model for production")
            else:
                st.success("✅ Using custom model")
        
        st.markdown("---")
        
        hist_tab1, hist_tab2 = st.tabs(["Detection History", "Clear History"])
        
        with hist_tab1:
            st.subheader("Detection History")
            
            # Filters
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            with col_filter1:
                safety_filter = st.selectbox("Filter by Safety Status", ["All", "SITE SAFE", "UNSAFE"])
            with col_filter2:
                input_type_filter = st.selectbox("Filter by Input Type", ["All", "image", "video", "webcam"])
            with col_filter3:
                if st.button("Load History"):
                    df = detection_db.get_all_detections()
                    
                    if not df.empty:
                        # Apply filters
                        if safety_filter != "All":
                            df = df[df['safety_status'] == safety_filter]
                        if input_type_filter != "All":
                            df = df[df['input_type'] == input_type_filter]
                        
                        if not df.empty:
                            st.dataframe(df, use_container_width=True)
                            
                            # Export to CSV
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Export to CSV",
                                csv,
                                "detection_history.csv",
                                "text/csv",
                                key="download-history-csv"
                            )
                        else:
                            st.info("No records match the selected filters")
                    else:
                        st.info("No detection history found")
        
        with hist_tab2:
            st.subheader("Clear History")
            st.warning("⚠️ This action cannot be undone!")
            
            clear_confirmation = st.checkbox("I understand and want to clear all detection history")
            
            if clear_confirmation:
                if st.button("Clear All History", type="primary"):
                    success = detection_db.clear_history(confirmed=True)
                    if success:
                        st.success("History cleared successfully")
                        st.rerun()
                    else:
                        st.error("Failed to clear history")
    
    # Overall Statistics Tab
    with tab3:
        st.markdown("### Overall Statistics")
        
        # Get statistics from DetectionDatabase
        stats = detection_db.get_summary_stats()
        df = detection_db.get_all_detections()
        
        if not df.empty:
            # Metric cards
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Scans", stats["total_scans"])
            
            with col2:
                st.metric("Safe Scans", stats["safe_scans"], delta_color="normal")
            
            with col3:
                st.metric("Unsafe Scans", stats["unsafe_scans"], delta_color="inverse")
            
            with col4:
                st.metric("Total Violations", stats["total_violations"], delta_color="inverse")
            
            with col5:
                st.metric("Avg Violations/Scan", f"{stats['avg_violations_per_scan']:.2f}")
            
            st.markdown("---")
            
            # Plotly Charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.subheader("Safe/Unsafe Distribution")
                fig_pie = px.pie(
                    values=[stats["safe_scans"], stats["unsafe_scans"]],
                    names=["Safe Scans", "Unsafe Scans"],
                    title="Safety Status Distribution",
                    color_discrete_map={"Safe Scans": "#00CC96", "Unsafe Scans": "#EF553B"}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with chart_col2:
                st.subheader("Input Type Distribution")
                input_type_counts = df['input_type'].value_counts()
                fig_bar = px.bar(
                    x=input_type_counts.index,
                    y=input_type_counts.values,
                    title="Detection by Input Type",
                    labels={"x": "Input Type", "y": "Count"}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("---")
            
            # Activity over time line chart
            st.subheader("Activity Over Time")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            daily_counts = df.groupby('date').size().reset_index(name='count')
            
            fig_line = px.line(
                daily_counts,
                x='date',
                y='count',
                title="Detection Activity Over Time",
                labels={"date": "Date", "count": "Number of Detections"}
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
            st.markdown("---")
            
            # Violation count bar chart
            st.subheader("Violation Count by Scan")
            violation_data = df[df['total_violations'] > 0][['timestamp', 'total_violations']].copy()
            violation_data['timestamp'] = pd.to_datetime(violation_data['timestamp'])
            
            if not violation_data.empty:
                fig_violation = px.bar(
                    violation_data,
                    x='timestamp',
                    y='total_violations',
                    title="Violations per Scan",
                    labels={"timestamp": "Scan Time", "total_violations": "Number of Violations"}
                )
                st.plotly_chart(fig_violation, use_container_width=True)
            else:
                st.info("No violations recorded yet")
            
        else:
            st.info("📊 No detection history available yet. Start detection to generate statistics.")


if __name__ == "__main__":
    main()
