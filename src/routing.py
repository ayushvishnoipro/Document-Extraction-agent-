"""
Document type detection using LangChain and OpenAI GPT.
Classifies documents into invoice, medical_bill, or prescription categories.
"""

import logging
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentTypeResult(BaseModel):
    """Model for document type classification result."""
    doc_type: str = Field(description="Document type: invoice, medical_bill, or prescription")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation for the classification")


class DocumentTypeClassifier:
    """Document type classifier using LangChain and OpenAI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the document classifier.
        
        Args:
            api_key: OpenAI API key (if not provided, will use environment variable)
        """
        # Set up OpenAI API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            logger.warning("OpenAI API key not found. Please set OPENAI_API_KEY in .env file or environment variable.")
        
        # Get model from environment or use default
        model_name = os.getenv("OPENAI_MODEL_GPT4_MINI", "gpt-4o-mini")
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=200
        )
        
        # Set up output parser
        self.output_parser = JsonOutputParser(pydantic_object=DocumentTypeResult)
        
        # Create the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "Please classify the following document text:\n\n{text}")
        ])
        
        # Create the chain
        self.chain = self.prompt_template | self.llm | self.output_parser
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for document classification."""
        return """You are an expert document classifier. Your task is to classify documents into one of three categories:

1. **invoice** - Business invoices, bills for goods/services, purchase orders
   - Keywords: Invoice, Bill, Purchase Order, Vendor, Customer, Line Items, Total, Due Date, Tax
   - Contains: Business addresses, item descriptions, quantities, prices, totals

2. **medical_bill** - Hospital bills, medical service charges, healthcare invoices
   - Keywords: Hospital, Clinic, Patient, Medical, Healthcare, Insurance, Diagnosis, Treatment
   - Contains: Patient information, medical services, insurance details, healthcare provider info

3. **prescription** - Medical prescriptions, medication orders
   - Keywords: Prescription, Doctor, Patient, Medicine, Medication, Dosage, Pharmacy, Dr.
   - Contains: Doctor information, patient details, medication names, dosages, instructions

Classification Guidelines:
- Look for specific keywords and document structure
- Consider the context and purpose of the document
- Invoice: Focus on commercial transactions
- Medical Bill: Focus on healthcare services and charges
- Prescription: Focus on medication orders and instructions

Provide your response as JSON with:
- doc_type: One of "invoice", "medical_bill", or "prescription"
- confidence: Float between 0.0 and 1.0
- reasoning: Brief explanation for your classification

Examples:

Text containing "Invoice #12345, Vendor: ABC Corp, Total: $1,500"
→ {{"doc_type": "invoice", "confidence": 0.95, "reasoning": "Contains invoice number, vendor information, and commercial total"}}

Text containing "Patient: John Doe, Hospital: City Medical Center, Services: Surgery"
→ {{"doc_type": "medical_bill", "confidence": 0.90, "reasoning": "Contains patient information, hospital name, and medical services"}}

Text containing "Dr. Smith, Prescription for: Amoxicillin 500mg, Take twice daily"
→ {{"doc_type": "prescription", "confidence": 0.92, "reasoning": "Contains doctor name, medication, and dosage instructions"}}"""
    
    def classify_document(self, text: str) -> Dict[str, Any]:
        """
        Classify document type based on extracted text.
        
        Args:
            text: Extracted text from the document
            
        Returns:
            Dictionary with classification results
        """
        try:
            if not text or not text.strip():
                return {
                    "doc_type": "unknown",
                    "confidence": 0.0,
                    "reasoning": "No text content found"
                }
            
            # Truncate text if too long (keep first 2000 characters for efficiency)
            if len(text) > 2000:
                text = text[:2000] + "..."
                logger.info("Truncated text for classification")
            
            # Run the classification chain
            result = self.chain.invoke({"text": text})
            
            # Validate the result
            if not isinstance(result, dict):
                raise ValueError("Invalid response format from LLM")
            
            # Ensure doc_type is valid
            valid_types = {"invoice", "medical_bill", "prescription"}
            if result.get("doc_type") not in valid_types:
                logger.warning(f"Invalid doc_type: {result.get('doc_type')}, defaulting to 'unknown'")
                result["doc_type"] = "unknown"
                result["confidence"] = 0.0
            
            # Ensure confidence is in valid range
            confidence = result.get("confidence", 0.0)
            if not 0.0 <= confidence <= 1.0:
                logger.warning(f"Invalid confidence score: {confidence}, clamping to range [0,1]")
                result["confidence"] = max(0.0, min(1.0, confidence))
            
            logger.info(f"Classified document as '{result['doc_type']}' with confidence {result['confidence']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            return {
                "doc_type": "unknown",
                "confidence": 0.0,
                "reasoning": f"Classification failed: {str(e)}"
            }
    
    def classify_with_keywords(self, text: str) -> Dict[str, Any]:
        """
        Fallback classification method using keyword matching.
        
        Args:
            text: Extracted text from the document
            
        Returns:
            Dictionary with classification results
        """
        text_lower = text.lower()
        
        # Define keyword sets for each document type
        invoice_keywords = {
            'invoice', 'bill', 'purchase order', 'vendor', 'customer', 'total', 'amount due',
            'line item', 'quantity', 'price', 'subtotal', 'tax', 'payment terms'
        }
        
        medical_keywords = {
            'hospital', 'clinic', 'patient', 'medical', 'healthcare', 'insurance',
            'diagnosis', 'treatment', 'service date', 'provider', 'physician'
        }
        
        prescription_keywords = {
            'prescription', 'doctor', 'dr.', 'medicine', 'medication', 'dosage',
            'pharmacy', 'rx', 'take', 'tablet', 'capsule', 'mg', 'ml'
        }
        
        # Count keyword matches
        invoice_score = sum(1 for keyword in invoice_keywords if keyword in text_lower)
        medical_score = sum(1 for keyword in medical_keywords if keyword in text_lower)
        prescription_score = sum(1 for keyword in prescription_keywords if keyword in text_lower)
        
        # Determine classification
        max_score = max(invoice_score, medical_score, prescription_score)
        
        if max_score == 0:
            return {
                "doc_type": "unknown",
                "confidence": 0.0,
                "reasoning": "No matching keywords found"
            }
        
        if invoice_score == max_score:
            doc_type = "invoice"
        elif medical_score == max_score:
            doc_type = "medical_bill"
        else:
            doc_type = "prescription"
        
        # Calculate confidence based on keyword density
        total_words = len(text_lower.split())
        confidence = min(0.8, max_score / max(1, total_words / 50))  # Cap at 0.8 for keyword-based classification
        
        return {
            "doc_type": doc_type,
            "confidence": confidence,
            "reasoning": f"Keyword-based classification: {max_score} matching keywords"
        }


def detect_document_type(text: str, api_key: Optional[str] = None, use_fallback: bool = True) -> Dict[str, Any]:
    """
    Convenience function to detect document type.
    
    Args:
        text: Extracted text from document
        api_key: OpenAI API key
        use_fallback: Whether to use keyword-based fallback if LLM fails
        
    Returns:
        Dictionary with document type and confidence
    """
    classifier = DocumentTypeClassifier(api_key)
    
    try:
        result = classifier.classify_document(text)
        
        # If LLM classification failed and fallback is enabled
        if result["doc_type"] == "unknown" and use_fallback:
            logger.info("LLM classification failed, trying keyword-based fallback")
            result = classifier.classify_with_keywords(text)
        
        return result
        
    except Exception as e:
        logger.error(f"Document classification failed: {str(e)}")
        
        if use_fallback:
            logger.info("Attempting keyword-based fallback classification")
            try:
                return classifier.classify_with_keywords(text)
            except Exception as fallback_error:
                logger.error(f"Fallback classification also failed: {str(fallback_error)}")
        
        return {
            "doc_type": "unknown",
            "confidence": 0.0,
            "reasoning": f"All classification methods failed: {str(e)}"
        }
