"""
OCR and document preprocessing module.
Handles text extraction from PDFs and images using pdfplumber, PyMuPDF, and pytesseract.
"""

import logging
import io
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image

# Handle both relative and absolute imports
try:
    from .utils import clean_text, bbox_to_dict
except ImportError:
    from utils import clean_text, bbox_to_dict

# Handle optional imports with better error handling
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not available. PDF text extraction will use PyMuPDF only.")

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    PYTESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available. OCR functionality will be limited.")

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available. Image preprocessing will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """OCR processor for documents."""
    
    def __init__(self):
        """Initialize OCR processor."""
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
    def detect_scanned_pdf(self, file_path: str) -> bool:
        """
        Detect if a PDF is scanned (image-based) or text-based.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            True if scanned (requires OCR), False if text-based
        """
        try:
            doc = fitz.open(file_path)
            
            # Check first few pages for text content
            pages_to_check = min(3, len(doc))
            text_content = ""
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                text_content += page.get_text()
            
            doc.close()
            
            # If very little text found, likely scanned
            cleaned_text = clean_text(text_content).strip()
            
            # Heuristic: if less than 50 characters per page on average, likely scanned
            avg_chars_per_page = len(cleaned_text) / pages_to_check if pages_to_check > 0 else 0
            
            logger.info(f"Average characters per page: {avg_chars_per_page}")
            
            return avg_chars_per_page < 50
            
        except Exception as e:
            logger.error(f"Error detecting PDF type: {str(e)}")
            return True  # Default to assuming it needs OCR
    
    def extract_text_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from text-based PDF using pdfplumber or PyMuPDF fallback.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with extracted text, tables, and metadata
        """
        try:
            extracted_data = {
                'extracted_text': '',
                'table_data': [],
                'page_metadata': [],
                'ocr_quality': 1.0  # High quality for text-based PDFs
            }
            
            if PDFPLUMBER_AVAILABLE:
                # Use pdfplumber for better table extraction
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        # Extract text
                        page_text = page.extract_text() or ""
                        extracted_data['extracted_text'] += f"\n--- Page {page_num} ---\n{page_text}\n"
                        
                        # Extract tables
                        tables = page.extract_tables()
                        for table_idx, table in enumerate(tables):
                            if table:
                                table_dict = {
                                    'page': page_num,
                                    'table_index': table_idx,
                                    'headers': table[0] if table else [],
                                    'rows': table[1:] if len(table) > 1 else [],
                                    'raw_table': table
                                }
                                extracted_data['table_data'].append(table_dict)
                        
                        # Page metadata
                        page_meta = {
                            'page_num': page_num,
                            'width': page.width,
                            'height': page.height,
                            'text_length': len(page_text)
                        }
                        extracted_data['page_metadata'].append(page_meta)
            else:
                # Fallback to PyMuPDF
                logger.info("Using PyMuPDF fallback for PDF text extraction")
                doc = fitz.open(file_path)
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_text = page.get_text()
                    extracted_data['extracted_text'] += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    
                    # Basic page metadata
                    rect = page.rect
                    page_meta = {
                        'page_num': page_num + 1,
                        'width': rect.width,
                        'height': rect.height,
                        'text_length': len(page_text)
                    }
                    extracted_data['page_metadata'].append(page_meta)
                
                doc.close()
            
            extracted_data['extracted_text'] = clean_text(extracted_data['extracted_text'])
            logger.info(f"Extracted {len(extracted_data['extracted_text'])} characters from PDF")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def extract_text_with_ocr(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from scanned PDF or image using OCR.
        
        Args:
            file_path: Path to PDF or image file
            
        Returns:
            Dictionary with extracted text, confidence scores, and metadata
        """
        try:
            extracted_data = {
                'extracted_text': '',
                'table_data': [],
                'page_metadata': [],
                'ocr_quality': 0.0
            }
            
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                # Convert PDF pages to images and OCR
                doc = fitz.open(file_path)
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Convert page to image
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Process with OCR
                    image = Image.open(io.BytesIO(img_data))
                    page_result = self._ocr_image(image, page_num + 1)
                    
                    extracted_data['extracted_text'] += page_result['text']
                    extracted_data['page_metadata'].extend(page_result['metadata'])
                    
                    # Accumulate OCR quality
                    if page_result['confidence'] > 0:
                        extracted_data['ocr_quality'] += page_result['confidence']
                
                doc.close()
                
                # Average OCR quality
                if len(doc) > 0:
                    extracted_data['ocr_quality'] /= len(doc)
                
            elif file_ext in self.supported_image_formats:
                # Direct image OCR
                image = Image.open(file_path)
                result = self._ocr_image(image, 1)
                
                extracted_data['extracted_text'] = result['text']
                extracted_data['page_metadata'] = result['metadata']
                extracted_data['ocr_quality'] = result['confidence']
            
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            extracted_data['extracted_text'] = clean_text(extracted_data['extracted_text'])
            logger.info(f"OCR extracted {len(extracted_data['extracted_text'])} characters")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error performing OCR: {str(e)}")
            raise
    
    def _ocr_image(self, image: Image.Image, page_num: int) -> Dict[str, Any]:
        """
        Perform OCR on a single image.
        
        Args:
            image: PIL Image object
            page_num: Page number
            
        Returns:
            Dictionary with OCR results
        """
        try:
            if not PYTESSERACT_AVAILABLE:
                logger.error("pytesseract is not available. Cannot perform OCR.")
                return {
                    'text': f"\n--- Page {page_num} ---\n[OCR not available - pytesseract not installed]\n",
                    'confidence': 0.0,
                    'metadata': [{
                        'page_num': page_num,
                        'width': image.width,
                        'height': image.height,
                        'text_length': 0,
                        'word_count': 0,
                        'avg_confidence': 0,
                        'error': 'pytesseract not available'
                    }]
                }
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(image)
            
            # Get detailed OCR data with confidence scores
            ocr_data = pytesseract.image_to_data(
                processed_image, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'  # Assume a single uniform block of text
            )
            
            # Extract text and calculate average confidence
            text_parts = []
            confidences = []
            
            for i, word in enumerate(ocr_data['text']):
                if word.strip():
                    text_parts.append(word)
                    conf = int(ocr_data['conf'][i])
                    if conf > 0:  # Ignore negative confidence scores
                        confidences.append(conf)
            
            text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Convert confidence to 0-1 scale
            normalized_confidence = avg_confidence / 100.0
            
            # Page metadata
            metadata = [{
                'page_num': page_num,
                'width': image.width,
                'height': image.height,
                'text_length': len(text),
                'word_count': len(text_parts),
                'avg_confidence': avg_confidence
            }]
            
            return {
                'text': f"\n--- Page {page_num} ---\n{text}\n",
                'confidence': normalized_confidence,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            return {
                'text': f"\n--- Page {page_num} ---\n[OCR failed: {str(e)}]\n",
                'confidence': 0.0,
                'metadata': [{
                    'page_num': page_num,
                    'width': image.width if hasattr(image, 'width') else 0,
                    'height': image.height if hasattr(image, 'height') else 0,
                    'text_length': 0,
                    'word_count': 0,
                    'avg_confidence': 0,
                    'error': str(e)
                }]
            }
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            if not OPENCV_AVAILABLE:
                # Basic preprocessing using PIL only
                logger.info("OpenCV not available, using basic PIL preprocessing")
                
                # Convert to grayscale
                if image.mode != 'L':
                    image = image.convert('L')
                
                # Return preprocessed image
                return image
            
            # Advanced preprocessing with OpenCV
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            img_array = cv2.GaussianBlur(img_array, (1, 1), 0)
            
            # Apply threshold to get binary image
            _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL Image
            return Image.fromarray(img_array)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return image  # Return original if preprocessing fails
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Main document processing function.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with extracted text, tables, and metadata
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                # Check if PDF needs OCR
                if self.detect_scanned_pdf(file_path):
                    logger.info("PDF appears to be scanned, using OCR")
                    return self.extract_text_with_ocr(file_path)
                else:
                    logger.info("PDF appears to be text-based, extracting text directly")
                    return self.extract_text_from_pdf(file_path)
            
            elif file_ext in self.supported_image_formats:
                logger.info("Processing image file with OCR")
                return self.extract_text_with_ocr(file_path)
            
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise


def extract_document_text(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to extract text from a document.
    
    Args:
        file_path: Path to document file
        
    Returns:
        Dictionary with extracted text and metadata
    """
    processor = OCRProcessor()
    return processor.process_document(file_path)
