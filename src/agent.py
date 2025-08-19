"""
Main orchestrator pipeline for the Agentic Document Extraction system.
Coordinates OCR, routing, extraction, validation, and confidence scoring using LangChain.
"""

import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Handle both relative and absolute imports
try:
    from .ocr import extract_document_text
    from .routing import detect_document_type
    from .extraction import extract_document_data
    from .validation import validate_extracted_data
    from .scoring import score_extraction_confidence
    from .schemas import ExtractionResult, FieldExtraction, ValidationResult
    from .utils import save_json, create_output_filename, get_file_info
except ImportError:
    from ocr import extract_document_text
    from routing import detect_document_type
    from extraction import extract_document_data
    from validation import validate_extracted_data
    from scoring import score_extraction_confidence
    from schemas import ExtractionResult, FieldExtraction, ValidationResult
    from utils import save_json, create_output_filename, get_file_info

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentExtractionAgent:
    """Main orchestrator for document extraction pipeline."""
    
    def __init__(self, api_key: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize the document extraction agent.
        
        Args:
            api_key: OpenAI API key (if not provided, will use environment variable)
            output_dir: Directory to save extraction results (default: from env or outputs/)
        """
        self.api_key = api_key
        
        # Set output directory from parameter, environment, or default
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.getenv("DEFAULT_OUTPUT_DIR", "outputs")
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Document extraction agent initialized with output directory: {self.output_dir}")
    
    def process_document(self, file_path: str, save_results: bool = True) -> Dict[str, Any]:
        """
        Main document processing pipeline.
        
        Args:
            file_path: Path to the document file (PDF or image)
            save_results: Whether to save results to JSON file
            
        Returns:
            Complete extraction results
        """
        try:
            start_time = datetime.now()
            logger.info(f"Starting document processing for: {file_path}")
            
            # Step 1: Get file information
            file_info = get_file_info(file_path)
            logger.info(f"Processing file: {file_info.get('filename', 'unknown')} ({file_info.get('size_mb', 0):.2f} MB)")
            
            # Step 2: OCR and text extraction
            logger.info("Step 1/5: Extracting text and tables from document...")
            ocr_results = extract_document_text(file_path)
            
            extracted_text = ocr_results.get("extracted_text", "")
            table_data = ocr_results.get("table_data", [])
            ocr_quality = ocr_results.get("ocr_quality", 0.8)
            
            if not extracted_text.strip():
                raise ValueError("No text could be extracted from the document")
            
            logger.info(f"Extracted {len(extracted_text)} characters, {len(table_data)} tables")
            
            # Step 3: Document type detection
            logger.info("Step 2/5: Detecting document type...")
            routing_results = detect_document_type(extracted_text, self.api_key)
            
            doc_type = routing_results.get("doc_type", "unknown")
            routing_confidence = routing_results.get("confidence", 0.0)
            
            logger.info(f"Detected document type: {doc_type} (confidence: {routing_confidence:.3f})")
            
            if doc_type == "unknown":
                logger.warning("Document type could not be determined")
            
            # Step 4: Structured extraction
            logger.info("Step 3/5: Extracting structured data...")
            extraction_results = extract_document_data(
                doc_type=doc_type,
                extracted_text=extracted_text,
                table_data=table_data,
                api_key=self.api_key
            )
            
            if not extraction_results.get("success", False):
                logger.error(f"Structured extraction failed: {extraction_results.get('error', 'Unknown error')}")
                structured_data = {}
            else:
                structured_data = extraction_results.get("structured_data", {})
                logger.info(f"Successfully extracted {len(structured_data)} fields")
            
            # Step 5: Validation
            logger.info("Step 4/5: Validating extracted data...")
            validation_results = validate_extracted_data(doc_type, structured_data)
            
            passed_rules = len(validation_results.get("passed_rules", []))
            failed_rules = len(validation_results.get("failed_rules", []))
            
            logger.info(f"Validation completed: {passed_rules} passed, {failed_rules} failed")
            
            # Step 6: Confidence scoring
            logger.info("Step 5/5: Calculating confidence scores...")
            
            extraction_metadata = {
                "ocr_quality": ocr_quality,
                "extraction_method": extraction_results.get("extraction_method", "unknown"),
                "routing_confidence": routing_confidence,
                "file_info": file_info
            }
            
            confidence_results = score_extraction_confidence(
                structured_data=structured_data,
                extraction_metadata=extraction_metadata,
                validation_results=validation_results
            )
            
            overall_confidence = confidence_results.get("overall_confidence", 0.0)
            field_confidences = confidence_results.get("field_confidences", {})
            
            logger.info(f"Overall confidence: {overall_confidence:.3f}")
            
            # Build final results
            processing_time = (datetime.now() - start_time).total_seconds()
            
            final_results = self._build_final_results(
                doc_type=doc_type,
                structured_data=structured_data,
                field_confidences=field_confidences,
                overall_confidence=overall_confidence,
                validation_results=validation_results,
                extraction_metadata=extraction_metadata,
                processing_time=processing_time,
                source_file=file_path
            )
            
            # Save results if requested
            if save_results:
                output_filename = create_output_filename(doc_type, Path(file_path).name)
                output_path = Path(self.output_dir) / output_filename
                
                if save_json(final_results, str(output_path)):
                    final_results["output_file"] = str(output_path)
                    logger.info(f"Results saved to: {output_path}")
            
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return self._build_error_results(str(e), file_path)
    
    def _build_final_results(self, 
                             doc_type: str,
                             structured_data: Dict[str, Any],
                             field_confidences: Dict[str, float],
                             overall_confidence: float,
                             validation_results: Dict[str, Any],
                             extraction_metadata: Dict[str, Any],
                             processing_time: float,
                             source_file: str) -> Dict[str, Any]:
        """Build the final results dictionary."""
        
        # Convert structured data to field extractions
        fields = []
        
        def extract_fields(data, prefix="", source_page=1):
            """Recursively extract fields from structured data."""
            if isinstance(data, dict):
                for key, value in data.items():
                    if key.endswith("_confidence"):
                        continue
                    
                    field_path = f"{prefix}.{key}" if prefix else key
                    
                    if isinstance(value, (dict, list)):
                        extract_fields(value, field_path, source_page)
                    elif value is not None and str(value).strip():
                        confidence = field_confidences.get(field_path, 0.5)
                        
                        field = FieldExtraction(
                            name=field_path,
                            value=str(value),
                            confidence=confidence,
                            source={
                                "page": source_page,
                                "bbox": {"x0": 0, "y0": 0, "x1": 0, "y1": 0}  # Placeholder
                            }
                        )
                        fields.append(field)
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    item_path = f"{prefix}[{i}]" if prefix else f"item_{i}"
                    extract_fields(item, item_path, source_page)
        
        extract_fields(structured_data)
        
        # Build validation result object
        validation = ValidationResult(
            passed_rules=validation_results.get("passed_rules", []),
            failed_rules=validation_results.get("failed_rules", []),
            notes=validation_results.get("notes", "")
        )
        
        # Build final extraction result
        result = ExtractionResult(
            doc_type=doc_type,
            fields=fields,
            structured_data=structured_data,
            overall_confidence=overall_confidence,
            validation=validation,
            processing_metadata={
                "processing_time_seconds": processing_time,
                "source_file": source_file,
                "extraction_timestamp": datetime.now().isoformat(),
                "extraction_metadata": extraction_metadata,
                "field_count": len(fields),
                "confidence_distribution": extraction_metadata.get("confidence_distribution", {})
            }
        )
        
        # Convert to dictionary for JSON serialization
        return result.dict()
    
    def _build_error_results(self, error_message: str, source_file: str) -> Dict[str, Any]:
        """Build error results dictionary."""
        return {
            "doc_type": "unknown",
            "fields": [],
            "structured_data": {},
            "overall_confidence": 0.0,
            "validation": {
                "passed_rules": [],
                "failed_rules": ["processing_error"],
                "notes": f"Processing failed: {error_message}"
            },
            "processing_metadata": {
                "processing_time_seconds": 0.0,
                "source_file": source_file,
                "extraction_timestamp": datetime.now().isoformat(),
                "error": error_message,
                "field_count": 0
            }
        }
    
    def batch_process_documents(self, file_paths: list, save_results: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            save_results: Whether to save individual results
            
        Returns:
            Dictionary mapping file paths to results
        """
        logger.info(f"Starting batch processing of {len(file_paths)} documents")
        
        results = {}
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"Processing file {i}/{len(file_paths)}: {file_path}")
                result = self.process_document(file_path, save_results)
                results[file_path] = result
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results[file_path] = self._build_error_results(str(e), file_path)
        
        # Save batch summary
        if save_results:
            batch_summary = self._create_batch_summary(results)
            summary_path = Path(self.output_dir) / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_json(batch_summary, str(summary_path))
            logger.info(f"Batch summary saved to: {summary_path}")
        
        logger.info(f"Batch processing completed: {len(results)} files processed")
        
        return results
    
    def _create_batch_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of batch processing results."""
        total_files = len(results)
        successful = sum(1 for r in results.values() if r.get("overall_confidence", 0) > 0)
        failed = total_files - successful
        
        doc_type_counts = {}
        confidence_scores = []
        
        for result in results.values():
            doc_type = result.get("doc_type", "unknown")
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
            
            confidence = result.get("overall_confidence", 0.0)
            if confidence > 0:
                confidence_scores.append(confidence)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            "batch_summary": {
                "total_files": total_files,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total_files if total_files > 0 else 0.0,
                "average_confidence": avg_confidence,
                "document_types": doc_type_counts
            },
            "individual_results": {
                file_path: {
                    "doc_type": result.get("doc_type"),
                    "confidence": result.get("overall_confidence"),
                    "field_count": result.get("processing_metadata", {}).get("field_count", 0),
                    "processing_time": result.get("processing_metadata", {}).get("processing_time_seconds", 0)
                }
                for file_path, result in results.items()
            },
            "generation_timestamp": datetime.now().isoformat()
        }


def process_document_pipeline(file_path: str, 
                              api_key: Optional[str] = None, 
                              output_dir: Optional[str] = None,
                              save_results: bool = True) -> Dict[str, Any]:
    """
    Convenience function to process a single document through the complete pipeline.
    
    Args:
        file_path: Path to document file
        api_key: OpenAI API key
        output_dir: Output directory for results
        save_results: Whether to save results to file
        
    Returns:
        Complete extraction results
    """
    agent = DocumentExtractionAgent(api_key=api_key, output_dir=output_dir)
    return agent.process_document(file_path, save_results=save_results)
