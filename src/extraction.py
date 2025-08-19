"""
LLM-based structured extraction using LangChain and OpenAI GPT.
Extracts structured data from documents based on document type and schema.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List, Type
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

# Handle both relative and absolute imports
try:
    from .schemas import InvoiceSchema, MedicalBillSchema, PrescriptionSchema, SCHEMA_MAPPING
except ImportError:
    from schemas import InvoiceSchema, MedicalBillSchema, PrescriptionSchema, SCHEMA_MAPPING

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StructuredExtractor:
    """LLM-based structured data extractor."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the structured extractor.
        
        Args:
            api_key: OpenAI API key (if not provided, will use environment variable)
        """
        # Set up OpenAI API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            logger.warning("OpenAI API key not found. Please set OPENAI_API_KEY in .env file or environment variable.")
        
        # Get model from environment or use default
        model_name = os.getenv("OPENAI_MODEL_GPT4", "gpt-4o")
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=2000
        )
    
    def _get_extraction_prompt(self, doc_type: str) -> str:
        """
        Get extraction prompt template for specific document type.
        
        Args:
            doc_type: Document type (invoice, medical_bill, prescription)
            
        Returns:
            Prompt template string
        """
        base_prompt = """You are an expert document data extractor. Your task is to extract structured information from the provided document text.

IMPORTANT INSTRUCTIONS:
1. Extract information EXACTLY as it appears in the document
2. Do not invent or hallucinate any information
3. If a field is not found, set it to null
4. For confidence scores: use 0.9+ for clearly visible text, 0.7-0.8 for partially visible, 0.5-0.6 for uncertain
5. For dates: preserve the original format when possible
6. For amounts: extract numeric values without currency symbols
7. For line items: extract all items you can identify

"""
        
        type_specific_prompts = {
            "invoice": """
This is an INVOICE document. Focus on extracting:
- Invoice identification (number, date, due date)
- Vendor/supplier information (name, address)
- Customer/buyer information (name, address)
- Line items (description, quantity, unit price, total)
- Financial totals (subtotal, tax, total amount)
- Currency information

Pay special attention to:
- Tables with itemized charges
- Total amounts and calculations
- Payment terms and due dates
""",
            
            "medical_bill": """
This is a MEDICAL BILL document. Focus on extracting:
- Patient information (name, ID)
- Healthcare provider information (hospital/clinic name, address)
- Service information (dates, attending doctor)
- Medical services and charges (line items)
- Insurance and payment information
- Financial totals (total amount, paid amount, balance due)

Pay special attention to:
- Patient demographics
- Service dates vs billing dates
- Insurance coverage details
- Medical service descriptions
""",
            
            "prescription": """
This is a PRESCRIPTION document. Focus on extracting:
- Patient information (name, age, address)
- Doctor information (name, license, clinic)
- Prescription details (date, diagnosis)
- Medications (name, dosage, frequency, duration)
- Instructions and notes

Pay special attention to:
- Medication names and dosages
- Frequency and duration instructions
- Doctor credentials and signatures
- Pharmacy information if present
"""
        }
        
        return base_prompt + type_specific_prompts.get(doc_type, "")
    
    def extract_structured_data(self, doc_type: str, extracted_text: str, table_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Extract structured data using LLM with schema enforcement.
        
        Args:
            doc_type: Document type (invoice, medical_bill, prescription)
            extracted_text: Raw extracted text from document
            table_data: Optional table data from OCR
            
        Returns:
            Dictionary with structured extraction results
        """
        try:
            # Get the appropriate schema
            schema_class = SCHEMA_MAPPING.get(doc_type)
            if not schema_class:
                raise ValueError(f"Unknown document type: {doc_type}")
            
            # Set up output parser with schema
            output_parser = PydanticOutputParser(pydantic_object=schema_class)
            
            # Create prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", self._get_extraction_prompt(doc_type)),
                ("human", self._create_human_prompt(extracted_text, table_data, output_parser))
            ])
            
            # Create extraction chain
            chain = prompt_template | self.llm | output_parser
            
            # Run extraction with template variables
            result = chain.invoke({
                "text": extracted_text[:4000],  # Limit text length for efficiency
                "tables": self._format_table_data(table_data) if table_data else "No tables found"
            })
            
            # Convert Pydantic model to dict
            structured_data = result.dict() if hasattr(result, 'dict') else dict(result)
            
            # Add confidence scores to each field if not present
            structured_data = self._add_confidence_scores(structured_data, extracted_text)
            
            logger.info(f"Successfully extracted structured data for {doc_type}")
            
            return {
                "success": True,
                "structured_data": structured_data,
                "doc_type": doc_type,
                "extraction_method": "llm_structured"
            }
            
        except OutputParserException as e:
            logger.error(f"Output parsing failed: {str(e)}")
            # Fallback to JSON extraction
            return self._fallback_json_extraction(doc_type, extracted_text, table_data)
            
        except Exception as e:
            logger.error(f"Structured extraction failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "doc_type": doc_type,
                "extraction_method": "failed"
            }
    
    def _create_human_prompt(self, text: str, table_data: Optional[List[Dict]], output_parser) -> str:
        """Create the human prompt for extraction."""
        # Get format instructions and escape curly braces for LangChain template compatibility
        format_instructions = output_parser.get_format_instructions()
        # Escape curly braces to prevent LangChain from treating them as template variables
        escaped_format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
        
        prompt = f"""
Document Text:
{{text}}

Table Data:
{{tables}}

{escaped_format_instructions}

Extract the information and return it in the specified JSON format. Be thorough but accurate.
"""
        return prompt
    
    def _format_table_data(self, table_data: List[Dict]) -> str:
        """Format table data for inclusion in prompt."""
        if not table_data:
            return "No tables found"
        
        formatted_tables = []
        for i, table in enumerate(table_data):
            table_str = f"Table {i+1} (Page {table.get('page', 'unknown')}):\n"
            
            if table.get('headers'):
                table_str += f"Headers: {', '.join(table['headers'])}\n"
            
            if table.get('rows'):
                for j, row in enumerate(table['rows'][:5]):  # Limit to first 5 rows
                    table_str += f"Row {j+1}: {', '.join(str(cell) for cell in row)}\n"
                
                if len(table['rows']) > 5:
                    table_str += f"... and {len(table['rows']) - 5} more rows\n"
            
            formatted_tables.append(table_str)
        
        return "\n".join(formatted_tables)
    
    def _add_confidence_scores(self, data: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Add confidence scores to extracted fields based on text analysis."""
        def add_confidence_recursive(obj, parent_key=""):
            if isinstance(obj, dict):
                # Create a list of items to avoid modifying dict during iteration
                items_to_process = list(obj.items())
                for key, value in items_to_process:
                    if isinstance(value, (dict, list)):
                        add_confidence_recursive(value, key)
                    elif value is not None and str(value).strip():
                        # Calculate confidence based on text matching
                        confidence = self._calculate_field_confidence(str(value), original_text)
                        if key != 'confidence' and not key.endswith('_confidence'):
                            # Add confidence field if not already present
                            confidence_key = f"{key}_confidence"
                            if confidence_key not in obj:
                                obj[confidence_key] = confidence
            elif isinstance(obj, list):
                for item in obj:
                    add_confidence_recursive(item, parent_key)
        
        add_confidence_recursive(data)
        return data
    
    def _calculate_field_confidence(self, value: str, text: str) -> float:
        """Calculate confidence score for a field based on text matching."""
        if not value or not text:
            return 0.0
        
        # Simple confidence calculation based on exact match
        if value.lower() in text.lower():
            return 0.9
        
        # Partial match confidence
        words = value.lower().split()
        matching_words = sum(1 for word in words if word in text.lower())
        
        if matching_words > 0:
            return 0.7 * (matching_words / len(words))
        
        return 0.3  # Default low confidence
    
    def _fallback_json_extraction(self, doc_type: str, text: str, table_data: Optional[List[Dict]]) -> Dict[str, Any]:
        """
        Fallback extraction method using JSON output.
        
        Args:
            doc_type: Document type
            text: Extracted text
            table_data: Table data
            
        Returns:
            Extraction results
        """
        try:
            logger.info("Attempting fallback JSON extraction")
            
            # Create a simplified prompt for JSON extraction
            prompt = f"""
Extract information from this {doc_type} document and return it as valid JSON.

Document Text:
{text[:2000]}

Return a JSON object with the relevant fields for a {doc_type}. 
Include confidence scores (0.0 to 1.0) for each field.
Use null for missing information.
"""
            
            # Simple LLM call for JSON
            response = self.llm.invoke(prompt)
            
            # Try to parse JSON from response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                structured_data = json.loads(json_str)
                
                return {
                    "success": True,
                    "structured_data": structured_data,
                    "doc_type": doc_type,
                    "extraction_method": "fallback_json"
                }
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            logger.error(f"Fallback extraction also failed: {str(e)}")
            return {
                "success": False,
                "error": f"All extraction methods failed: {str(e)}",
                "doc_type": doc_type,
                "extraction_method": "failed"
            }


def extract_document_data(doc_type: str, extracted_text: str, table_data: Optional[List[Dict]] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to extract structured data from document.
    
    Args:
        doc_type: Document type (invoice, medical_bill, prescription)
        extracted_text: Extracted text from document
        table_data: Optional table data
        api_key: OpenAI API key
        
    Returns:
        Dictionary with extraction results
    """
    extractor = StructuredExtractor(api_key)
    return extractor.extract_structured_data(doc_type, extracted_text, table_data)
