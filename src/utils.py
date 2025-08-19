"""
Utility functions for the document extraction system.
Provides helper functions for text processing, file handling, and data manipulation.
"""

import json
import os
import re
import logging
from typing import Dict, List, Any, Union, Tuple
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text with normalized whitespace and removed artifacts
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common OCR artifacts
    text = re.sub(r'[^\w\s\.,;:!?\-@#$%&*()+=\[\]{}|\\/<>"]', '', text)
    
    # Normalize line breaks
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def bbox_to_dict(bbox: Union[Tuple, List]) -> Dict[str, float]:
    """
    Convert bounding box to dictionary format.
    
    Args:
        bbox: Bounding box as tuple/list (x0, y0, x1, y1)
        
    Returns:
        Dictionary with bbox coordinates
    """
    if not bbox or len(bbox) < 4:
        return {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
    
    return {
        "x0": float(bbox[0]),
        "y0": float(bbox[1]),
        "x1": float(bbox[2]),
        "y1": float(bbox[3])
    }


def save_json(data: Dict[str, Any], filepath: str) -> bool:
    """
    Save structured data as JSON to outputs folder.
    
    Args:
        data: Data to save
        filepath: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure outputs directory exists
        output_dir = Path(filepath).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved extraction results to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {str(e)}")
        return False


def load_file(file_path: str) -> bytes:
    """
    Load file into memory for processing.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content as bytes
    """
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {str(e)}")
        raise


def is_pdf_file(file_path: str) -> bool:
    """
    Check if file is a PDF.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if PDF, False otherwise
    """
    return file_path.lower().endswith('.pdf')


def is_image_file(file_path: str) -> bool:
    """
    Check if file is an image.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if image, False otherwise
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return Path(file_path).suffix.lower() in image_extensions


def extract_numbers(text: str) -> List[float]:
    """
    Extract numeric values from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted numbers
    """
    # Pattern to match numbers (including decimals and currency)
    pattern = r'[\d,]+\.?\d*'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            # Remove commas and convert to float
            number = float(match.replace(',', ''))
            numbers.append(number)
        except ValueError:
            continue
    
    return numbers


def extract_dates(text: str) -> List[str]:
    """
    Extract date patterns from text.
    
    Args:
        text: Input text
        
    Returns:
        List of potential date strings
    """
    # Common date patterns
    date_patterns = [
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # MM/DD/YYYY or MM-DD-YYYY
        r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY/MM/DD or YYYY-MM-DD
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',  # DD Month YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b'  # Month DD, YYYY
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    
    return dates


def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from text.
    
    Args:
        text: Input text
        
    Returns:
        List of email addresses
    """
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)


def extract_phone_numbers(text: str) -> List[str]:
    """
    Extract phone numbers from text.
    
    Args:
        text: Input text
        
    Returns:
        List of phone numbers
    """
    # Pattern for various phone number formats
    patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # XXX-XXX-XXXX
        r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',    # (XXX) XXX-XXXX
        r'\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}'  # International
    ]
    
    phone_numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        phone_numbers.extend(matches)
    
    return phone_numbers


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Simple character-based similarity
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def create_output_filename(doc_type: str, original_filename: str) -> str:
    """
    Create output filename for extracted data.
    
    Args:
        doc_type: Document type
        original_filename: Original file name
        
    Returns:
        Output filename
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(original_filename).stem
    
    return f"{base_name}_{doc_type}_{timestamp}.json"


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get file metadata.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    file_path = Path(file_path)
    
    try:
        stat = file_path.stat()
        return {
            "filename": file_path.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "extension": file_path.suffix.lower(),
            "modified_time": stat.st_mtime,
            "is_pdf": is_pdf_file(str(file_path)),
            "is_image": is_image_file(str(file_path))
        }
    except Exception as e:
        logger.error(f"Failed to get file info for {file_path}: {str(e)}")
        return {}


def validate_extracted_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Basic validation of extracted data structure.
    
    Args:
        data: Extracted data dictionary
        
    Returns:
        Validation results
    """
    issues = []
    
    # Check for required keys
    required_keys = ["doc_type", "fields", "overall_confidence"]
    for key in required_keys:
        if key not in data:
            issues.append(f"Missing required key: {key}")
    
    # Check confidence scores
    if "overall_confidence" in data:
        if not 0 <= data["overall_confidence"] <= 1:
            issues.append("Overall confidence not between 0 and 1")
    
    # Check fields structure
    if "fields" in data and isinstance(data["fields"], list):
        for i, field in enumerate(data["fields"]):
            if not isinstance(field, dict):
                issues.append(f"Field {i} is not a dictionary")
            else:
                required_field_keys = ["name", "value", "confidence"]
                for key in required_field_keys:
                    if key not in field:
                        issues.append(f"Field {i} missing key: {key}")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues
    }
