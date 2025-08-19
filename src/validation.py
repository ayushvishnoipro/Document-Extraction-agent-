"""
Validation layer for extracted document data.
Performs regex checks, cross-field validation, and business rule validation.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Handle both relative and absolute imports
try:
    from .utils import extract_numbers, extract_dates, extract_emails, extract_phone_numbers
except ImportError:
    from utils import extract_numbers, extract_dates, extract_emails, extract_phone_numbers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentValidator:
    """Validator for extracted document data."""
    
    def __init__(self):
        """Initialize the document validator."""
        self.validation_rules = {
            "invoice": self._get_invoice_rules(),
            "medical_bill": self._get_medical_bill_rules(),
            "prescription": self._get_prescription_rules()
        }
    
    def validate_document(self, doc_type: str, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted document data.
        
        Args:
            doc_type: Document type (invoice, medical_bill, prescription)
            structured_data: Extracted structured data
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_result = {
                "passed_rules": [],
                "failed_rules": [],
                "warnings": [],
                "notes": "",
                "overall_valid": True
            }
            
            # Get validation rules for document type
            rules = self.validation_rules.get(doc_type, [])
            
            # Run each validation rule
            for rule in rules:
                try:
                    result = rule["function"](structured_data)
                    if result["passed"]:
                        validation_result["passed_rules"].append(rule["name"])
                    else:
                        validation_result["failed_rules"].append(rule["name"])
                        validation_result["overall_valid"] = False
                    
                    # Add warnings if any
                    if result.get("warnings"):
                        validation_result["warnings"].extend(result["warnings"])
                        
                except Exception as e:
                    logger.error(f"Error running validation rule '{rule['name']}': {str(e)}")
                    validation_result["failed_rules"].append(f"{rule['name']} (error)")
                    validation_result["overall_valid"] = False
            
            # Run common validation rules
            common_results = self._validate_common_fields(structured_data)
            validation_result["passed_rules"].extend(common_results["passed_rules"])
            validation_result["failed_rules"].extend(common_results["failed_rules"])
            validation_result["warnings"].extend(common_results["warnings"])
            
            if common_results["failed_rules"]:
                validation_result["overall_valid"] = False
            
            # Generate summary notes
            validation_result["notes"] = self._generate_validation_notes(validation_result)
            
            logger.info(f"Validation completed for {doc_type}: {len(validation_result['passed_rules'])} passed, {len(validation_result['failed_rules'])} failed")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                "passed_rules": [],
                "failed_rules": ["validation_system_error"],
                "warnings": [f"Validation system error: {str(e)}"],
                "notes": "Validation could not be completed due to system error",
                "overall_valid": False
            }
    
    def _get_invoice_rules(self) -> List[Dict[str, Any]]:
        """Get validation rules for invoice documents."""
        return [
            {
                "name": "invoice_number_present",
                "function": lambda data: self._validate_field_present(data, "invoice_number", "Invoice number")
            },
            {
                "name": "total_amount_valid",
                "function": lambda data: self._validate_amount_field(data, "total_amount", "Total amount")
            },
            {
                "name": "line_items_total_match",
                "function": self._validate_invoice_totals
            },
            {
                "name": "vendor_name_present",
                "function": lambda data: self._validate_field_present(data, "vendor_name", "Vendor name")
            },
            {
                "name": "date_format_valid",
                "function": lambda data: self._validate_date_field(data, "date", "Invoice date")
            }
        ]
    
    def _get_medical_bill_rules(self) -> List[Dict[str, Any]]:
        """Get validation rules for medical bill documents."""
        return [
            {
                "name": "patient_name_present",
                "function": lambda data: self._validate_field_present(data, "patient_name", "Patient name")
            },
            {
                "name": "hospital_name_present",
                "function": lambda data: self._validate_field_present(data, "hospital_name", "Hospital name")
            },
            {
                "name": "total_amount_valid",
                "function": lambda data: self._validate_amount_field(data, "total_amount", "Total amount")
            },
            {
                "name": "bill_date_valid",
                "function": lambda data: self._validate_date_field(data, "bill_date", "Bill date")
            },
            {
                "name": "balance_calculation_valid",
                "function": self._validate_medical_bill_balance
            }
        ]
    
    def _get_prescription_rules(self) -> List[Dict[str, Any]]:
        """Get validation rules for prescription documents."""
        return [
            {
                "name": "patient_name_present",
                "function": lambda data: self._validate_field_present(data, "patient_name", "Patient name")
            },
            {
                "name": "doctor_name_present",
                "function": lambda data: self._validate_field_present(data, "doctor_name", "Doctor name")
            },
            {
                "name": "medicines_present",
                "function": self._validate_medicines_list
            },
            {
                "name": "prescription_date_valid",
                "function": lambda data: self._validate_date_field(data, "prescription_date", "Prescription date")
            },
            {
                "name": "medicine_dosages_valid",
                "function": self._validate_medicine_dosages
            }
        ]
    
    def _validate_field_present(self, data: Dict[str, Any], field_name: str, field_description: str) -> Dict[str, Any]:
        """Validate that a required field is present and not empty."""
        value = data.get(field_name)
        
        if value is None or (isinstance(value, str) and not value.strip()):
            return {
                "passed": False,
                "message": f"{field_description} is missing or empty"
            }
        
        return {
            "passed": True,
            "message": f"{field_description} is present"
        }
    
    def _validate_amount_field(self, data: Dict[str, Any], field_name: str, field_description: str) -> Dict[str, Any]:
        """Validate that an amount field contains a valid positive number."""
        value = data.get(field_name)
        
        if value is None:
            return {
                "passed": False,
                "message": f"{field_description} is missing"
            }
        
        try:
            amount = float(value)
            if amount < 0:
                return {
                    "passed": False,
                    "message": f"{field_description} cannot be negative"
                }
            
            return {
                "passed": True,
                "message": f"{field_description} is valid"
            }
            
        except (ValueError, TypeError):
            return {
                "passed": False,
                "message": f"{field_description} is not a valid number"
            }
    
    def _validate_date_field(self, data: Dict[str, Any], field_name: str, field_description: str) -> Dict[str, Any]:
        """Validate date field format."""
        value = data.get(field_name)
        
        if value is None or (isinstance(value, str) and not value.strip()):
            return {
                "passed": False,
                "message": f"{field_description} is missing"
            }
        
        # Common date patterns
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}',
            r'\d{1,2}\s+\w+\s+\d{2,4}',
            r'\w+\s+\d{1,2},?\s+\d{2,4}'
        ]
        
        value_str = str(value)
        for pattern in date_patterns:
            if re.search(pattern, value_str):
                return {
                    "passed": True,
                    "message": f"{field_description} format appears valid"
                }
        
        return {
            "passed": False,
            "message": f"{field_description} format does not match expected patterns",
            "warnings": [f"Unusual date format: {value_str}"]
        }
    
    def _validate_invoice_totals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that invoice line items sum to total."""
        try:
            line_items = data.get("line_items", [])
            total_amount = data.get("total_amount")
            subtotal = data.get("subtotal")
            tax_amount = data.get("tax_amount")
            
            if not line_items:
                return {
                    "passed": False,
                    "message": "No line items found to validate totals"
                }
            
            # Calculate sum of line items
            line_item_total = 0
            for item in line_items:
                if isinstance(item, dict) and "total_price" in item:
                    try:
                        line_item_total += float(item["total_price"])
                    except (ValueError, TypeError):
                        continue
            
            warnings = []
            
            # Check subtotal if present
            if subtotal is not None:
                try:
                    subtotal_float = float(subtotal)
                    if abs(line_item_total - subtotal_float) > 0.01:  # Allow for rounding
                        warnings.append(f"Line items total ({line_item_total:.2f}) doesn't match subtotal ({subtotal_float:.2f})")
                except (ValueError, TypeError):
                    warnings.append("Subtotal is not a valid number")
            
            # Check total amount
            if total_amount is not None:
                try:
                    total_float = float(total_amount)
                    expected_total = line_item_total
                    
                    # Add tax if present
                    if tax_amount is not None:
                        try:
                            expected_total += float(tax_amount)
                        except (ValueError, TypeError):
                            pass
                    
                    if abs(expected_total - total_float) > 0.01:
                        return {
                            "passed": False,
                            "message": f"Total amount ({total_float:.2f}) doesn't match calculated total ({expected_total:.2f})",
                            "warnings": warnings
                        }
                    
                except (ValueError, TypeError):
                    return {
                        "passed": False,
                        "message": "Total amount is not a valid number",
                        "warnings": warnings
                    }
            
            return {
                "passed": True,
                "message": "Invoice totals are consistent",
                "warnings": warnings
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Error validating invoice totals: {str(e)}"
            }
    
    def _validate_medical_bill_balance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medical bill balance calculation."""
        try:
            total_amount = data.get("total_amount")
            amount_paid = data.get("amount_paid")
            balance_due = data.get("balance_due")
            
            if total_amount is None:
                return {
                    "passed": False,
                    "message": "Total amount required for balance validation"
                }
            
            try:
                total_float = float(total_amount)
                paid_float = float(amount_paid) if amount_paid is not None else 0
                balance_float = float(balance_due) if balance_due is not None else None
                
                calculated_balance = total_float - paid_float
                
                if balance_float is not None:
                    if abs(calculated_balance - balance_float) > 0.01:
                        return {
                            "passed": False,
                            "message": f"Balance due ({balance_float:.2f}) doesn't match calculated balance ({calculated_balance:.2f})"
                        }
                
                return {
                    "passed": True,
                    "message": "Medical bill balance calculation is correct"
                }
                
            except (ValueError, TypeError):
                return {
                    "passed": False,
                    "message": "Invalid numeric values in balance calculation"
                }
                
        except Exception as e:
            return {
                "passed": False,
                "message": f"Error validating balance: {str(e)}"
            }
    
    def _validate_medicines_list(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that medicines list is present and not empty."""
        medicines = data.get("medicines", [])
        
        if not medicines or len(medicines) == 0:
            return {
                "passed": False,
                "message": "No medicines found in prescription"
            }
        
        # Check that medicines have required fields
        valid_medicines = 0
        for medicine in medicines:
            if isinstance(medicine, dict) and medicine.get("name"):
                valid_medicines += 1
        
        if valid_medicines == 0:
            return {
                "passed": False,
                "message": "No valid medicines with names found"
            }
        
        return {
            "passed": True,
            "message": f"Found {valid_medicines} valid medicines"
        }
    
    def _validate_medicine_dosages(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medicine dosage formats."""
        medicines = data.get("medicines", [])
        
        if not medicines:
            return {
                "passed": True,
                "message": "No medicines to validate dosages"
            }
        
        dosage_pattern = r'\d+\s*(mg|ml|g|mcg|units?|tablets?|capsules?)'
        valid_dosages = 0
        warnings = []
        
        for i, medicine in enumerate(medicines):
            if isinstance(medicine, dict):
                dosage = medicine.get("dosage", "")
                if dosage:
                    if re.search(dosage_pattern, dosage, re.IGNORECASE):
                        valid_dosages += 1
                    else:
                        warnings.append(f"Medicine {i+1} has unusual dosage format: {dosage}")
        
        return {
            "passed": True,  # Don't fail on dosage format, just warn
            "message": f"Validated dosages for {valid_dosages} medicines",
            "warnings": warnings
        }
    
    def _validate_common_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate common fields across all document types."""
        passed_rules = []
        failed_rules = []
        warnings = []
        
        # Validate email addresses if present
        for field_name in ["email", "contact_email", "vendor_email"]:
            if field_name in data and data[field_name]:
                email = str(data[field_name])
                if self._is_valid_email(email):
                    passed_rules.append(f"{field_name}_format_valid")
                else:
                    failed_rules.append(f"{field_name}_format_invalid")
        
        # Validate phone numbers if present
        for field_name in ["phone", "contact_phone", "vendor_phone"]:
            if field_name in data and data[field_name]:
                phone = str(data[field_name])
                if self._is_valid_phone(phone):
                    passed_rules.append(f"{field_name}_format_valid")
                else:
                    warnings.append(f"Unusual phone number format: {phone}")
        
        return {
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "warnings": warnings
        }
    
    def _is_valid_email(self, email: str) -> bool:
        """Check if email format is valid."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _is_valid_phone(self, phone: str) -> bool:
        """Check if phone number format is reasonable."""
        # Remove common formatting characters
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # Check if it looks like a phone number (7-15 digits)
        return 7 <= len(cleaned.replace('+', '')) <= 15
    
    def _generate_validation_notes(self, validation_result: Dict[str, Any]) -> str:
        """Generate summary notes for validation results."""
        passed_count = len(validation_result["passed_rules"])
        failed_count = len(validation_result["failed_rules"])
        warning_count = len(validation_result["warnings"])
        
        notes = f"{passed_count} validation rules passed"
        
        if failed_count > 0:
            notes += f", {failed_count} failed"
        
        if warning_count > 0:
            notes += f", {warning_count} warnings"
        
        if failed_count == 0 and warning_count == 0:
            notes += " - Document appears valid"
        elif failed_count > 0:
            notes += " - Document has validation issues"
        else:
            notes += " - Document valid with minor warnings"
        
        return notes


def validate_extracted_data(doc_type: str, structured_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to validate extracted document data.
    
    Args:
        doc_type: Document type (invoice, medical_bill, prescription)
        structured_data: Extracted structured data
        
    Returns:
        Dictionary with validation results
    """
    validator = DocumentValidator()
    return validator.validate_document(doc_type, structured_data)
