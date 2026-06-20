"""
PPE Detection Module using YOLO
Handles detection of safety helmets and head violations
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Dict, Optional, Union
from ultralytics import YOLO


class SafetyDetector:
    """YOLO-based PPE detector for construction site safety monitoring"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the SafetyDetector
        
        Args:
            project_root: Path to project root. If None, uses current file's parent directory
        
        Returns:
            is_demo_model: Boolean flag indicating if using demo model
        """
        self.is_demo_model = False
        self.model = None
        self.class_names = {}
        
        try:
            if project_root is None:
                project_root = Path(__file__).parent.resolve()
            
            # Try to load custom model from models/best.pt
            custom_model_path = project_root / "models" / "best.pt"
            
            if custom_model_path.exists():
                try:
                    self.model = YOLO(str(custom_model_path))
                    self.class_names = self.model.names
                    self.is_demo_model = False
                except Exception as e:
                    print(f"Failed to load custom model: {e}. Falling back to demo model.")
                    self._load_demo_model()
            else:
                self._load_demo_model()
                
        except Exception as e:
            print(f"Error initializing detector: {e}. Using fallback.")
            self._load_demo_model()
    
    def _load_demo_model(self):
        """Load the demo YOLOv8n model"""
        try:
            self.model = YOLO("yolov8n.pt")
            self.class_names = self.model.names
            self.is_demo_model = True
        except Exception as e:
            print(f"Failed to load demo model: {e}")
            self.model = None
            self.class_names = {}
    
    def detect_image(self, image: Union[np.ndarray, Image.Image]) -> Tuple[Image.Image, List[Dict], str, int, int]:
        """
        Run YOLO detection on an image with confidence threshold 0.5
        
        Args:
            image: Input image as PIL Image or numpy array (BGR or RGB format)
            
        Returns:
            Tuple of (annotated_pil_image, detections_list, safety_status, total_person_count, total_violation_count)
        """
        # Handle error cases
        if self.model is None:
            return self._return_error_result(image, "Model not loaded")
        
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                # PIL images are RGB, convert to BGR for OpenCV
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_np = image.copy()
            
            # Run inference with confidence threshold 0.5
            results = self.model(image_np, conf=0.5, verbose=False)
            
            # Process detections
            detections = []
            total_person_count = 0
            total_violation_count = 0
            
            annotated_image = image_np.copy()
            
            # Process each detection
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    # Check if this is a violation
                    is_violation = self._is_violation(class_name)
                    
                    # Count persons (assuming person class or any detection)
                    total_person_count += 1
                    
                    if is_violation:
                        total_violation_count += 1
                    
                    # Store detection info
                    detection_info = {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "is_violation": is_violation
                    }
                    detections.append(detection_info)
                    
                    # Draw bounding box
                    if is_violation:
                        color = (0, 0, 255)  # Red for violations
                        label = f"VIOLATION: {class_name} {confidence:.1%}"
                    elif "person" in class_name.lower():
                        color = (255, 0, 0)  # Blue for person detections
                        label = f"PERSON: {class_name} {confidence:.1%}"
                    else:
                        color = (0, 255, 0)  # Green for safe PPE
                        label = f"SAFE: {class_name} {confidence:.1%}"
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(
                        annotated_image, 
                        (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), 
                        color, 
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_image, 
                        label, 
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        2
                    )
            
            # Determine safety status
            safety_status = "SITE SAFE" if total_violation_count == 0 else "UNSAFE"
            
            # Convert back to PIL Image (RGB)
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            
            return annotated_pil, detections, safety_status, total_person_count, total_violation_count
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return self._return_error_result(image, str(e))
    
    def _is_violation(self, class_name: str) -> bool:
        """
        Check if a class name indicates a safety violation
        
        Args:
            class_name: The class name to check
            
        Returns:
            True if the class indicates a violation, False otherwise
        """
        violation_keywords = ["no_helmet", "no-helmet", "no_vest", "no-vest", "unsafe"]
        class_name_lower = class_name.lower()
        
        for keyword in violation_keywords:
            if keyword in class_name_lower:
                return True
        
        return False
    
    def _return_error_result(self, image: Union[np.ndarray, Image.Image], error_message: str) -> Tuple[Image.Image, List[Dict], str, int, int]:
        """
        Return error result when detection fails
        
        Args:
            image: Original image
            error_message: Error message
            
        Returns:
            Tuple with error state
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR, convert to RGB
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        return pil_image, [], "ERROR", 0, 0
    
    def get_demo_banner_message(self) -> Optional[str]:
        """
        Get the demo banner message if using demo model
        
        Returns:
            Banner message string if using demo model, None otherwise
        """
        if self.is_demo_model:
            return "⚠️ Running demo model. Add your custom trained construction PPE detection model to models/best.pt for production accuracy"
        return None


# Backward compatibility - keep PPEDetector as alias
class PPEDetector(SafetyDetector):
    """Legacy alias for SafetyDetector"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the PPE detector (legacy compatibility)
        
        Args:
            model_path: Path to the YOLO model file (.pt)
        """
        project_root = Path(__file__).parent.resolve()
        
        if model_path and Path(model_path).exists():
            # Use the provided model path
            try:
                self.model = YOLO(model_path)
                self.class_names = self.model.names
                self.is_demo_model = False
            except Exception as e:
                print(f"Failed to load model: {e}. Falling back to demo model.")
                self._load_demo_model()
        else:
            # Use standard initialization
            super().__init__(project_root)
    
    def detect(self, image: np.ndarray) -> list:
        """
        Run YOLO detection on an image (legacy method)
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection results
        """
        if self.model is None:
            return []
        
        try:
            results = self.model(image, conf=0.5, verbose=False)
            return results
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def draw_annotations(self, image: np.ndarray, results: list) -> tuple:
        """
        Draw bounding boxes and labels on the image based on detection results (legacy method)
        
        Args:
            image: Input image as numpy array (BGR format)
            results: YOLO detection results
            
        Returns:
            Tuple of (annotated_image, statistics_dict)
        """
        annotated_image = image.copy()
        
        # Initialize statistics
        stats = {
            "total_persons": 0,
            "helmet_count": 0,
            "head_count": 0,
            "active_violations": 0,
            "compliance_percentage": 0.0
        }
        
        try:
            # Process each detection
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    # Increment total persons
                    stats["total_persons"] += 1
                    
                    # Check if violation
                    is_violation = self._is_violation(class_name)
                    
                    # Draw bounding box and label based on class
                    if is_violation:
                        color = (0, 0, 255)  # Red for violations
                        label = f"VIOLATION - {class_name}: {confidence:.1%}"
                        stats["active_violations"] += 1
                        if "head" in class_name.lower():
                            stats["head_count"] += 1
                    elif "helmet" in class_name.lower():
                        color = (0, 255, 0)  # Green for helmet (SAFE)
                        label = f"SAFE - {class_name}: {confidence:.1%}"
                        stats["helmet_count"] += 1
                    elif "person" in class_name.lower():
                        color = (255, 0, 0)  # Blue for person
                        label = f"PERSON - {class_name}: {confidence:.1%}"
                    else:
                        color = (0, 255, 0)  # Green for other safe items
                        label = f"SAFE - {class_name}: {confidence:.1%}"
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(
                        annotated_image, 
                        (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), 
                        color, 
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_image, 
                        label, 
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        2
                    )
            
            # Calculate compliance percentage
            total_detections = stats["helmet_count"] + stats["head_count"]
            if total_detections > 0:
                stats["compliance_percentage"] = (
                    stats["helmet_count"] / total_detections * 100
                )
        except Exception as e:
            print(f"Error drawing annotations: {e}")
        
        return annotated_image, stats
    
    def get_class_names(self) -> dict:
        """
        Get the class names from the model
        
        Returns:
            Dictionary mapping class IDs to class names
        """
        return self.class_names
