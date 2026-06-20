"""
Database module for Construction Site Safety Monitoring System
Handles SQLite database operations for detection history, violations, and sessions
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd


class DatabaseManager:
    """Manages SQLite database operations for safety monitoring"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize the database manager
        
        Args:
            db_path: Path to the database file. If None, uses default path
        """
        if db_path:
            self.db_path = Path(db_path)
        else:
            # Default to data directory in project root
            self.db_path = Path(__file__).parent / "data" / "safety_monitor.db"
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def initialize_database(self):
        """
        Initialize the SQLite database with required tables
        Creates tables if they don't exist
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Detection_History table - stores all detections
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Detection_History (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                detection_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_type TEXT NOT NULL,
                session_id INTEGER,
                FOREIGN KEY (session_id) REFERENCES Session_History(id)
            )
        ''')
        
        # Violation_History table - stores safety violations with screenshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Violation_History (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                detection_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_type TEXT NOT NULL,
                screenshot_path TEXT,
                session_id INTEGER,
                FOREIGN KEY (session_id) REFERENCES Session_History(id)
            )
        ''')
        
        # Session_History table - stores detection sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Session_History (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_start DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_end DATETIME,
                source_type TEXT NOT NULL,
                total_detections INTEGER DEFAULT 0,
                helmet_count INTEGER DEFAULT 0,
                head_count INTEGER DEFAULT 0,
                compliance_percentage REAL DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_session(self, source_type: str) -> int:
        """
        Start a new detection session
        Returns the session ID
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO Session_History (source_type, session_start)
            VALUES (?, CURRENT_TIMESTAMP)
        ''', (source_type,))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def end_session(self, session_id: int, total_detections: int, helmet_count: int, 
                    head_count: int, compliance_percentage: float):
        """
        End a detection session and update statistics
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE Session_History
            SET session_end = CURRENT_TIMESTAMP,
                total_detections = ?,
                helmet_count = ?,
                head_count = ?,
                compliance_percentage = ?
            WHERE id = ?
        ''', (total_detections, helmet_count, head_count, compliance_percentage, session_id))
        
        conn.commit()
        conn.close()
    
    def add_detection(self, detection_type: str, confidence: float, source_type: str, 
                     session_id: Optional[int] = None):
        """
        Add a detection record to the database
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO Detection_History (detection_type, confidence, source_type, session_id)
            VALUES (?, ?, ?, ?)
        ''', (detection_type, confidence, source_type, session_id))
        
        conn.commit()
        conn.close()
    
    def add_violation(self, detection_type: str, confidence: float, source_type: str,
                      screenshot_path: Optional[str] = None, session_id: Optional[int] = None):
        """
        Add a violation record to the database with optional screenshot path
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO Violation_History (detection_type, confidence, source_type, screenshot_path, session_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (detection_type, confidence, source_type, screenshot_path, session_id))
        
        conn.commit()
        conn.close()
    
    def get_detection_history(self, date_filter: Optional[str] = None, 
                             detection_type_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve detection history with optional filters
        Returns list of detection records
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = "SELECT * FROM Detection_History WHERE 1=1"
        params = []
        
        if date_filter:
            query += " AND DATE(timestamp) = ?"
            params.append(date_filter)
        
        if detection_type_filter:
            query += " AND detection_type = ?"
            params.append(detection_type_filter)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return results
    
    def get_violation_history(self, date_filter: Optional[str] = None,
                              detection_type_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve violation history with optional filters
        Returns list of violation records
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = "SELECT * FROM Violation_History WHERE 1=1"
        params = []
        
        if date_filter:
            query += " AND DATE(timestamp) = ?"
            params.append(date_filter)
        
        if detection_type_filter:
            query += " AND detection_type = ?"
            params.append(detection_type_filter)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return results
    
    def get_session_history(self, date_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve session history with optional date filter
        Returns list of session records
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = "SELECT * FROM Session_History WHERE 1=1"
        params = []
        
        if date_filter:
            query += " AND DATE(session_start) = ?"
            params.append(date_filter)
        
        query += " ORDER BY session_start DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get overall statistics from the database
        Returns dictionary with total counts and percentages
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Total detections
        cursor.execute("SELECT COUNT(*) FROM Detection_History")
        total_detections = cursor.fetchone()[0] or 0
        
        # Helmet count
        cursor.execute("SELECT COUNT(*) FROM Detection_History WHERE detection_type = 'helmet'")
        helmet_count = cursor.fetchone()[0] or 0
        
        # Head count
        cursor.execute("SELECT COUNT(*) FROM Detection_History WHERE detection_type = 'head'")
        head_count = cursor.fetchone()[0] or 0
        
        # Total violations
        cursor.execute("SELECT COUNT(*) FROM Violation_History")
        total_violations = cursor.fetchone()[0] or 0
        
        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM Session_History")
        total_sessions = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_detections": total_detections,
            "total_helmets": helmet_count,
            "total_heads": head_count,
            "total_violations": total_violations,
            "total_sessions": total_sessions
        }


class DetectionDatabase:
    """Simplified database for detection history with cloud-safe operations"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the DetectionDatabase
        
        Args:
            project_root: Path to project root. If None, uses current file's parent directory
        """
        if project_root is None:
            project_root = Path(__file__).parent.resolve()
        
        # Auto-create data folder if it doesn't exist
        self.data_dir = project_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database at data/detections.db
        self.db_path = self.data_dir / "detections.db"
        
        # Initialize database tables
        self._initialize_tables()
    
    def _initialize_tables(self):
        """Initialize the detection_history table"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    input_type TEXT NOT NULL,
                    file_name TEXT,
                    safety_status TEXT NOT NULL,
                    total_persons INTEGER DEFAULT 0,
                    total_detections INTEGER DEFAULT 0,
                    total_violations INTEGER DEFAULT 0,
                    detections_json TEXT,
                    output_path TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error initializing database tables: {e}")
    
    def _safe_db_operation(self, operation_func):
        """
        Wrapper for database operations with error handling for locked/read-only databases
        
        Args:
            operation_func: Function that performs the database operation
            
        Returns:
            Result of the operation or None if it fails
        """
        try:
            return operation_func()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "readonly" in str(e).lower():
                print(f"Database is locked or read-only: {e}")
                return None
            raise
        except Exception as e:
            print(f"Database operation failed: {e}")
            return None
    
    def save_detection(self, record: Dict) -> Optional[int]:
        """
        Insert new detection entry
        
        Args:
            record: Dictionary containing detection data with keys:
                - input_type: Type of input (image, video, etc.)
                - file_name: Name of the input file
                - safety_status: "SITE SAFE" or "UNSAFE"
                - total_persons: Total number of persons detected
                - total_detections: Total number of detections
                - total_violations: Total number of violations
                - detections: List of detection dictionaries (will be JSON serialized)
                - output_path: Path to the output file (optional)
                
        Returns:
            ID of the inserted record, or None if operation failed
        """
        def _save():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Serialize detections list to JSON
            detections_json = json.dumps(record.get('detections', []))
            
            cursor.execute('''
                INSERT INTO detection_history 
                (input_type, file_name, safety_status, total_persons, total_detections, total_violations, detections_json, output_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.get('input_type', 'unknown'),
                record.get('file_name'),
                record.get('safety_status', 'UNKNOWN'),
                record.get('total_persons', 0),
                record.get('total_detections', 0),
                record.get('total_violations', 0),
                detections_json,
                record.get('output_path')
            ))
            
            record_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return record_id
        
        return self._safe_db_operation(_save)
    
    def get_all_detections(self) -> pd.DataFrame:
        """
        Return full history as a pandas DataFrame
        
        Returns:
            DataFrame with all detection records, or empty DataFrame if operation fails
        """
        def _get_all():
            conn = sqlite3.connect(str(self.db_path))
            
            df = pd.read_sql_query("SELECT * FROM detection_history ORDER BY timestamp DESC", conn)
            
            conn.close()
            
            # Parse detections_json back to list
            if 'detections_json' in df.columns and not df.empty:
                df['detections'] = df['detections_json'].apply(lambda x: json.loads(x) if x else [])
                df = df.drop(columns=['detections_json'])
            
            return df
        
        result = self._safe_db_operation(_get_all)
        return result if result is not None else pd.DataFrame()
    
    def get_summary_stats(self) -> Dict:
        """
        Return summary statistics
        
        Returns:
            Dictionary with:
                - total_scans: Total number of scans performed
                - safe_scans: Number of scans with no violations
                - unsafe_scans: Number of scans with violations
                - total_violations: Total number of violations across all scans
                - avg_violations_per_scan: Average violations per scan
        """
        def _get_stats():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Total scans
            cursor.execute("SELECT COUNT(*) FROM detection_history")
            total_scans = cursor.fetchone()[0] or 0
            
            # Safe scans (no violations)
            cursor.execute("SELECT COUNT(*) FROM detection_history WHERE total_violations = 0")
            safe_scans = cursor.fetchone()[0] or 0
            
            # Unsafe scans (has violations)
            cursor.execute("SELECT COUNT(*) FROM detection_history WHERE total_violations > 0")
            unsafe_scans = cursor.fetchone()[0] or 0
            
            # Total violations
            cursor.execute("SELECT SUM(total_violations) FROM detection_history")
            total_violations = cursor.fetchone()[0] or 0
            
            conn.close()
            
            # Average violations per scan
            avg_violations_per_scan = (total_violations / total_scans) if total_scans > 0 else 0.0
            
            return {
                "total_scans": total_scans,
                "safe_scans": safe_scans,
                "unsafe_scans": unsafe_scans,
                "total_violations": total_violations,
                "avg_violations_per_scan": round(avg_violations_per_scan, 2)
            }
        
        result = self._safe_db_operation(_get_stats)
        return result if result is not None else {
            "total_scans": 0,
            "safe_scans": 0,
            "unsafe_scans": 0,
            "total_violations": 0,
            "avg_violations_per_scan": 0.0
        }
    
    def clear_history(self, confirmed: bool = False) -> bool:
        """
        Wipe all records after user confirmation
        
        Args:
            confirmed: Must be True to actually clear the history (safety check)
            
        Returns:
            True if history was cleared successfully, False otherwise
        """
        if not confirmed:
            print("History clear not confirmed. Set confirmed=True to proceed.")
            return False
        
        def _clear():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM detection_history")
            
            conn.commit()
            conn.close()
            
            return True
        
        result = self._safe_db_operation(_clear)
        return result if result is not None else False


# Backward compatibility functions
def initialize_database():
    """Legacy function for backward compatibility"""
    db_manager = DatabaseManager()
    db_manager.initialize_database()

def start_session(source_type: str) -> int:
    """Legacy function for backward compatibility"""
    db_manager = DatabaseManager()
    return db_manager.start_session(source_type)

def end_session(session_id: int, total_detections: int, helmet_count: int, 
                head_count: int, compliance_percentage: float):
    """Legacy function for backward compatibility"""
    db_manager = DatabaseManager()
    db_manager.end_session(session_id, total_detections, helmet_count, head_count, compliance_percentage)

def add_detection(detection_type: str, confidence: float, source_type: str, 
                 session_id: Optional[int] = None):
    """Legacy function for backward compatibility"""
    db_manager = DatabaseManager()
    db_manager.add_detection(detection_type, confidence, source_type, session_id)

def add_violation(detection_type: str, confidence: float, source_type: str,
                  screenshot_path: Optional[str] = None, session_id: Optional[int] = None):
    """Legacy function for backward compatibility"""
    db_manager = DatabaseManager()
    db_manager.add_violation(detection_type, confidence, source_type, screenshot_path, session_id)

def get_detection_history(date_filter: Optional[str] = None, 
                         detection_type_filter: Optional[str] = None) -> List[Dict]:
    """Legacy function for backward compatibility"""
    db_manager = DatabaseManager()
    return db_manager.get_detection_history(date_filter, detection_type_filter)

def get_violation_history(date_filter: Optional[str] = None,
                          detection_type_filter: Optional[str] = None) -> List[Dict]:
    """Legacy function for backward compatibility"""
    db_manager = DatabaseManager()
    return db_manager.get_violation_history(date_filter, detection_type_filter)

def get_session_history(date_filter: Optional[str] = None) -> List[Dict]:
    """Legacy function for backward compatibility"""
    db_manager = DatabaseManager()
    return db_manager.get_session_history(date_filter)

def get_statistics() -> Dict:
    """Legacy function for backward compatibility"""
    db_manager = DatabaseManager()
    return db_manager.get_statistics()

def export_to_csv(table_name: str, output_path: str):
    """Legacy function for backward compatibility"""
    import pandas as pd
    db_manager = DatabaseManager()
    conn = sqlite3.connect(str(db_manager.db_path))
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    df.to_csv(output_path, index=False)
    conn.close()

def export_to_excel(table_name: str, output_path: str):
    """Legacy function for backward compatibility"""
    import pandas as pd
    db_manager = DatabaseManager()
    conn = sqlite3.connect(str(db_manager.db_path))
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    df.to_excel(output_path, index=False)
    conn.close()


if __name__ == "__main__":
    initialize_database()
    print("Database module ready")
