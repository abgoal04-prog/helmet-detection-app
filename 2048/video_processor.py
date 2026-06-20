"""
Video Processing Module for Construction Site Safety Monitoring
Handles video file processing for PPE detection

NOTE: Long videos will be truncated on free Streamlit Cloud to avoid timeouts.
Consider processing shorter videos or upgrading to a paid tier for longer content.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from detector import SafetyDetector
import streamlit as st


class VideoProcessor:
    """Processes video files for PPE detection with Streamlit integration"""
    
    def __init__(self, detector: Optional[SafetyDetector] = None, project_root: Optional[Path] = None):
        """
        Initialize the video processor
        
        Args:
            detector: SafetyDetector instance. If None, creates a new one
            project_root: Path to project root. If None, uses current file's parent directory
        """
        if project_root is None:
            project_root = Path(__file__).parent.resolve()
        
        self.project_root = project_root
        self.uploads_dir = project_root / "uploads"
        self.outputs_dir = project_root / "outputs"
        
        # Auto-create uploads and outputs folders if they don't exist
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        self.detector = detector
    
    def process_uploaded_video(self, uploaded_file, progress_bar: Optional[st.progress] = None) -> Tuple[str, int, int, int, str]:
        """
        Process uploaded video file for PPE detection
        
        Args:
            uploaded_file: Streamlit uploaded file object
            progress_bar: Optional Streamlit progress bar for showing processing progress
            
        Returns:
            Tuple of (output_video_path, total_detections, total_violations, total_persons, safety_status)
        """
        # Validate file type
        allowed_extensions = ['.mp4', '.avi', '.mov']
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}. Allowed types: {allowed_extensions}")
        
        # Save uploaded file temporarily to uploads/
        temp_upload_path = self.uploads_dir / uploaded_file.name
        with open(temp_upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Process the video
            output_path, total_detections, total_violations, total_persons, safety_status = self._process_video(
                str(temp_upload_path),
                progress_bar=progress_bar
            )
            
            return str(output_path), total_detections, total_violations, total_persons, safety_status
            
        finally:
            # Clean up temporary upload file after processing
            if temp_upload_path.exists():
                temp_upload_path.unlink()
    
    def _process_video(self, video_path: str, progress_bar: Optional[st.progress] = None) -> Tuple[Path, int, int, int, str]:
        """
        Process video file frame by frame for PPE detection
        
        Args:
            video_path: Path to the video file
            progress_bar: Optional Streamlit progress bar
            
        Returns:
            Tuple of (output_path, total_detections, total_violations, total_persons, safety_status)
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Generate output path
        timestamp = cv2.getTickCount()
        output_path = self.outputs_dir / f"processed_{timestamp}.mp4"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Initialize statistics
        total_detections = 0
        total_violations = 0
        total_persons = 0
        frame_count = 0
        
        # Process every 2nd frame for cloud performance
        frame_skip = 2
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Only process every Nth frame for performance
                if frame_count % frame_skip == 0:
                    # Process frame with detector
                    if self.detector:
                        try:
                            annotated_pil, detections, safety_status, persons, violations = self.detector.detect_image(frame)
                            
                            # Convert PIL back to numpy array (BGR)
                            annotated_frame = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
                            
                            # Update statistics
                            total_detections += len(detections)
                            total_violations += violations
                            total_persons += persons
                            
                            # Determine final safety status
                            final_safety_status = "SITE SAFE" if total_violations == 0 else "UNSAFE"
                            
                        except Exception as e:
                            print(f"Error processing frame {frame_count}: {e}")
                            annotated_frame = frame
                            final_safety_status = "ERROR"
                    else:
                        annotated_frame = frame
                        final_safety_status = "UNKNOWN"
                else:
                    # Write original frame without processing
                    annotated_frame = frame
                
                # Write frame to output video
                writer.write(annotated_frame)
                
                # Update progress bar
                if progress_bar:
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                
                frame_count += 1
                
                # Safety limit: truncate long videos to avoid timeout on free Streamlit Cloud
                # Process maximum of 500 frames (approximately 16-17 seconds at 30fps)
                if frame_count >= 500:
                    print(f"Video truncated at frame {frame_count} to avoid timeout on free Streamlit Cloud")
                    break
                
        finally:
            # Clean up
            cap.release()
            writer.release()
        
        return output_path, total_detections, total_violations, total_persons, final_safety_status
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None) -> list:
        """
        Process a video file frame by frame for PPE detection (legacy method)
        
        Args:
            video_path: Path to the video file
            output_path: Optional path to save processed video
            
        Returns:
            List of (annotated_frame, frame_statistics) tuples
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process frame with detector
                if self.detector:
                    try:
                        annotated_pil, detections, safety_status, persons, violations = self.detector.detect_image(frame)
                        annotated_frame = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
                        stats = {
                            "total_persons": persons,
                            "total_violations": violations,
                            "safety_status": safety_status
                        }
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        annotated_frame = frame
                        stats = {"total_persons": 0, "total_violations": 0, "safety_status": "ERROR"}
                else:
                    annotated_frame = frame
                    stats = {"total_persons": 0, "total_violations": 0, "safety_status": "UNKNOWN"}
                
                # Write frame to output video if writer exists
                if writer:
                    writer.write(annotated_frame)
                
                frame_count += 1
                
                # Store result
                results.append((annotated_frame, stats))
                
                # Safety limit for cloud processing
                if frame_count >= 500:
                    print(f"Video truncated at frame {frame_count} to avoid timeout")
                    break
                
        finally:
            # Clean up
            cap.release()
            if writer:
                writer.release()
        
        return results
    
    def extract_frames(self, video_path: str, output_dir: str, frame_interval: int = 30) -> list:
        """
        Extract frames from a video file at specified intervals
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            frame_interval: Extract every Nth frame
            
        Returns:
            List of paths to extracted frames
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        extracted_frames = []
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract frame at specified interval
                if frame_count % frame_interval == 0:
                    frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(str(frame_path))
                
                frame_count += 1
                
        finally:
            cap.release()
        
        return extracted_frames
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get information about a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video properties
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            info = {
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
            }
        finally:
            cap.release()
        
        return info


if __name__ == "__main__":
    # Test video processor
    processor = VideoProcessor()
    print("Video processor module ready")
