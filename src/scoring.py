"""
Confidence scoring system for document extraction.
Combines LLM confidence, OCR quality, and validation results into overall scores.
"""

import logging
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Confidence scoring system for extracted document data."""
    
    def __init__(self, 
                 llm_weight: float = 0.6, 
                 ocr_weight: float = 0.2, 
                 validation_weight: float = 0.2):
        """
        Initialize confidence scorer with configurable weights.
        
        Args:
            llm_weight: Weight for LLM confidence scores (default 0.6)
            ocr_weight: Weight for OCR quality scores (default 0.2)
            validation_weight: Weight for validation results (default 0.2)
        """
        # Normalize weights to sum to 1.0
        total_weight = llm_weight + ocr_weight + validation_weight
        self.llm_weight = llm_weight / total_weight
        self.ocr_weight = ocr_weight / total_weight
        self.validation_weight = validation_weight / total_weight
        
        logger.info(f"Confidence scorer initialized with weights: LLM={self.llm_weight:.2f}, OCR={self.ocr_weight:.2f}, Validation={self.validation_weight:.2f}")
    
    def calculate_field_confidence(self, 
                                   field_name: str,
                                   field_value: Any,
                                   llm_confidence: Optional[float] = None,
                                   ocr_quality: Optional[float] = None,
                                   validation_passed: Optional[bool] = None,
                                   text_match_score: Optional[float] = None) -> float:
        """
        Calculate confidence score for a single field.
        
        Args:
            field_name: Name of the field
            field_value: Value of the field
            llm_confidence: LLM-reported confidence (0-1)
            ocr_quality: OCR quality score (0-1)
            validation_passed: Whether validation passed for this field
            text_match_score: How well the value matches the source text (0-1)
            
        Returns:
            Combined confidence score (0-1)
        """
        try:
            # Default values if not provided
            llm_conf = llm_confidence if llm_confidence is not None else self._estimate_llm_confidence(field_value)
            ocr_qual = ocr_quality if ocr_quality is not None else 0.8  # Default to good OCR quality
            val_score = 1.0 if validation_passed else 0.0 if validation_passed is False else 0.7  # Neutral if not validated
            
            # Apply text matching bonus/penalty
            if text_match_score is not None:
                llm_conf = min(1.0, llm_conf * (0.5 + 0.5 * text_match_score))
            
            # Calculate weighted confidence
            confidence = (
                self.llm_weight * llm_conf +
                self.ocr_weight * ocr_qual +
                self.validation_weight * val_score
            )
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            logger.debug(f"Field '{field_name}' confidence: {confidence:.3f} (LLM:{llm_conf:.3f}, OCR:{ocr_qual:.3f}, Val:{val_score:.3f})")
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence for field '{field_name}': {str(e)}")
            return 0.5  # Default moderate confidence
    
    def calculate_document_confidence(self, 
                                      structured_data: Dict[str, Any],
                                      extraction_metadata: Optional[Dict[str, Any]] = None,
                                      validation_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate confidence scores for all fields and overall document.
        
        Args:
            structured_data: Extracted structured data
            extraction_metadata: Metadata from extraction process
            validation_results: Results from validation process
            
        Returns:
            Dictionary with field-level and overall confidence scores
        """
        try:
            # Extract metadata
            ocr_quality = extraction_metadata.get("ocr_quality", 0.8) if extraction_metadata else 0.8
            extraction_method = extraction_metadata.get("extraction_method", "unknown") if extraction_metadata else "unknown"
            
            # Extract validation info
            validation_passed = validation_results.get("overall_valid", True) if validation_results else True
            passed_rules = validation_results.get("passed_rules", []) if validation_results else []
            failed_rules = validation_results.get("failed_rules", []) if validation_results else []
            
            # Calculate field-level confidences
            field_confidences = {}
            total_confidence = 0.0
            field_count = 0
            
            def process_fields(data, prefix=""):
                nonlocal total_confidence, field_count
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        if key.endswith("_confidence"):
                            continue  # Skip existing confidence fields
                        
                        field_path = f"{prefix}.{key}" if prefix else key
                        
                        if isinstance(value, (dict, list)):
                            # Recursively process nested structures
                            process_fields(value, field_path)
                        elif value is not None and str(value).strip():
                            # Calculate confidence for this field
                            llm_conf = self._extract_llm_confidence(data, key)
                            field_validation = self._get_field_validation_status(key, passed_rules, failed_rules)
                            
                            confidence = self.calculate_field_confidence(
                                field_name=field_path,
                                field_value=value,
                                llm_confidence=llm_conf,
                                ocr_quality=ocr_quality,
                                validation_passed=field_validation
                            )
                            
                            field_confidences[field_path] = confidence
                            total_confidence += confidence
                            field_count += 1
                
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        item_prefix = f"{prefix}[{i}]" if prefix else f"item_{i}"
                        process_fields(item, item_prefix)
            
            # Process all fields
            process_fields(structured_data)
            
            # Calculate overall confidence
            if field_count > 0:
                base_confidence = total_confidence / field_count
            else:
                base_confidence = 0.5  # Default if no fields found
            
            # Apply document-level adjustments
            overall_confidence = self._apply_document_adjustments(
                base_confidence, 
                validation_results, 
                extraction_metadata
            )
            
            # Calculate confidence distribution
            confidence_distribution = self._calculate_confidence_distribution(field_confidences)
            
            result = {
                "field_confidences": field_confidences,
                "overall_confidence": overall_confidence,
                "field_count": field_count,
                "avg_field_confidence": base_confidence,
                "confidence_distribution": confidence_distribution,
                "scoring_metadata": {
                    "ocr_quality": ocr_quality,
                    "extraction_method": extraction_method,
                    "validation_passed": validation_passed,
                    "weights_used": {
                        "llm_weight": self.llm_weight,
                        "ocr_weight": self.ocr_weight,
                        "validation_weight": self.validation_weight
                    }
                }
            }
            
            logger.info(f"Document confidence calculated: {overall_confidence:.3f} (based on {field_count} fields)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating document confidence: {str(e)}")
            return {
                "field_confidences": {},
                "overall_confidence": 0.5,
                "field_count": 0,
                "avg_field_confidence": 0.5,
                "confidence_distribution": {},
                "scoring_metadata": {"error": str(e)}
            }
    
    def _estimate_llm_confidence(self, field_value: Any) -> float:
        """Estimate LLM confidence based on field value characteristics."""
        if not field_value:
            return 0.0
        
        value_str = str(field_value).strip()
        
        # Length-based confidence (longer values often more confident)
        length_conf = min(1.0, len(value_str) / 20.0)
        
        # Pattern-based confidence
        pattern_conf = 0.5
        
        # Numbers (usually high confidence)
        if value_str.replace('.', '').replace(',', '').isdigit():
            pattern_conf = 0.9
        
        # Dates (moderate to high confidence)
        elif any(char.isdigit() for char in value_str) and any(sep in value_str for sep in ['/', '-', ' ']):
            pattern_conf = 0.8
        
        # Names (moderate confidence)
        elif value_str.replace(' ', '').isalpha() and len(value_str) > 3:
            pattern_conf = 0.7
        
        return min(1.0, (length_conf + pattern_conf) / 2.0)
    
    def _extract_llm_confidence(self, data: Dict[str, Any], field_name: str) -> Optional[float]:
        """Extract LLM confidence score for a field if available."""
        # Look for confidence field with various naming patterns
        confidence_keys = [
            f"{field_name}_confidence",
            f"{field_name}Confidence",
            "confidence"
        ]
        
        for key in confidence_keys:
            if key in data:
                try:
                    return float(data[key])
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _get_field_validation_status(self, field_name: str, passed_rules: List[str], failed_rules: List[str]) -> Optional[bool]:
        """Determine if validation passed for a specific field."""
        field_lower = field_name.lower()
        
        # Check if any validation rule specifically mentions this field
        for rule in failed_rules:
            if field_lower in rule.lower():
                return False
        
        for rule in passed_rules:
            if field_lower in rule.lower():
                return True
        
        return None  # No specific validation for this field
    
    def _apply_document_adjustments(self, 
                                    base_confidence: float, 
                                    validation_results: Optional[Dict[str, Any]], 
                                    extraction_metadata: Optional[Dict[str, Any]]) -> float:
        """Apply document-level adjustments to confidence score."""
        adjusted_confidence = base_confidence
        
        # Validation adjustments
        if validation_results:
            failed_rules = validation_results.get("failed_rules", [])
            passed_rules = validation_results.get("passed_rules", [])
            
            # Penalty for failed critical validations
            critical_failures = [rule for rule in failed_rules if any(critical in rule for critical in 
                                ["total", "amount", "balance", "required", "present"])]
            
            if critical_failures:
                penalty = min(0.3, len(critical_failures) * 0.1)
                adjusted_confidence *= (1.0 - penalty)
                logger.debug(f"Applied validation penalty: -{penalty:.3f} for {len(critical_failures)} critical failures")
        
        # Extraction method adjustments
        if extraction_metadata:
            extraction_method = extraction_metadata.get("extraction_method", "")
            
            if "fallback" in extraction_method:
                adjusted_confidence *= 0.8  # Reduce confidence for fallback methods
                logger.debug("Applied fallback method penalty: -20%")
            elif "structured" in extraction_method:
                adjusted_confidence *= 1.05  # Small boost for structured extraction
                adjusted_confidence = min(1.0, adjusted_confidence)
        
        # Ensure valid range
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _calculate_confidence_distribution(self, field_confidences: Dict[str, float]) -> Dict[str, Any]:
        """Calculate confidence distribution statistics."""
        if not field_confidences:
            return {}
        
        confidences = list(field_confidences.values())
        
        # Count fields by confidence ranges
        ranges = {
            "high (0.8-1.0)": sum(1 for c in confidences if c >= 0.8),
            "medium (0.5-0.8)": sum(1 for c in confidences if 0.5 <= c < 0.8),
            "low (0.0-0.5)": sum(1 for c in confidences if c < 0.5)
        }
        
        return {
            "ranges": ranges,
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "std_deviation": self._calculate_std(confidences)
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of confidence scores."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


def score_extraction_confidence(structured_data: Dict[str, Any],
                                extraction_metadata: Optional[Dict[str, Any]] = None,
                                validation_results: Optional[Dict[str, Any]] = None,
                                weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Convenience function to score extraction confidence.
    
    Args:
        structured_data: Extracted structured data
        extraction_metadata: Metadata from extraction process
        validation_results: Results from validation process
        weights: Custom weights for scoring components
        
    Returns:
        Dictionary with confidence scores and metadata
    """
    # Apply custom weights if provided
    if weights:
        scorer = ConfidenceScorer(
            llm_weight=weights.get("llm_weight", 0.6),
            ocr_weight=weights.get("ocr_weight", 0.2),
            validation_weight=weights.get("validation_weight", 0.2)
        )
    else:
        scorer = ConfidenceScorer()
    
    return scorer.calculate_document_confidence(
        structured_data=structured_data,
        extraction_metadata=extraction_metadata,
        validation_results=validation_results
    )
