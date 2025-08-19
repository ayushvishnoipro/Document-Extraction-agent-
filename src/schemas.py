"""
Pydantic models and JSON schemas for structured document extraction.
Defines data models for different document types: invoices, medical bills, and prescriptions.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class FieldExtraction(BaseModel):
    """Base model for extracted field with metadata."""
    name: str = Field(description="Field name")
    value: str = Field(description="Extracted value")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    source: Optional[Dict[str, Any]] = Field(default=None, description="Source location metadata")


class LineItem(BaseModel):
    """Model for invoice/bill line items."""
    description: str = Field(description="Item description")
    quantity: Optional[float] = Field(default=None, description="Item quantity")
    unit_price: Optional[float] = Field(default=None, description="Unit price")
    total_price: Optional[float] = Field(default=None, description="Total price for this item")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")


class Medicine(BaseModel):
    """Model for prescription medicines."""
    name: str = Field(description="Medicine name")
    dosage: Optional[str] = Field(default=None, description="Dosage information")
    frequency: Optional[str] = Field(default=None, description="Frequency of intake")
    duration: Optional[str] = Field(default=None, description="Duration of treatment")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")


class InvoiceSchema(BaseModel):
    """Schema for invoice documents."""
    invoice_number: Optional[str] = Field(default=None, description="Invoice number")
    date: Optional[str] = Field(default=None, description="Invoice date")
    due_date: Optional[str] = Field(default=None, description="Due date")
    vendor_name: Optional[str] = Field(default=None, description="Vendor/supplier name")
    vendor_address: Optional[str] = Field(default=None, description="Vendor address")
    customer_name: Optional[str] = Field(default=None, description="Customer name")
    customer_address: Optional[str] = Field(default=None, description="Customer address")
    line_items: List[LineItem] = Field(default_factory=list, description="Invoice line items")
    subtotal: Optional[float] = Field(default=None, description="Subtotal amount")
    tax_amount: Optional[float] = Field(default=None, description="Tax amount")
    total_amount: Optional[float] = Field(default=None, description="Total amount")
    currency: Optional[str] = Field(default="USD", description="Currency")


class MedicalBillSchema(BaseModel):
    """Schema for medical bill documents."""
    patient_name: Optional[str] = Field(default=None, description="Patient name")
    patient_id: Optional[str] = Field(default=None, description="Patient ID")
    hospital_name: Optional[str] = Field(default=None, description="Hospital/clinic name")
    hospital_address: Optional[str] = Field(default=None, description="Hospital address")
    bill_date: Optional[str] = Field(default=None, description="Bill date")
    service_date: Optional[str] = Field(default=None, description="Service date")
    doctor_name: Optional[str] = Field(default=None, description="Attending doctor")
    line_items: List[LineItem] = Field(default_factory=list, description="Medical services")
    insurance_info: Optional[str] = Field(default=None, description="Insurance information")
    total_amount: Optional[float] = Field(default=None, description="Total bill amount")
    amount_paid: Optional[float] = Field(default=None, description="Amount already paid")
    balance_due: Optional[float] = Field(default=None, description="Balance due")
    currency: Optional[str] = Field(default="USD", description="Currency")


class PrescriptionSchema(BaseModel):
    """Schema for prescription documents."""
    patient_name: Optional[str] = Field(default=None, description="Patient name")
    patient_age: Optional[str] = Field(default=None, description="Patient age")
    patient_address: Optional[str] = Field(default=None, description="Patient address")
    doctor_name: Optional[str] = Field(default=None, description="Doctor name")
    doctor_license: Optional[str] = Field(default=None, description="Doctor license number")
    clinic_name: Optional[str] = Field(default=None, description="Clinic/hospital name")
    clinic_address: Optional[str] = Field(default=None, description="Clinic address")
    prescription_date: Optional[str] = Field(default=None, description="Prescription date")
    medicines: List[Medicine] = Field(default_factory=list, description="Prescribed medicines")
    diagnosis: Optional[str] = Field(default=None, description="Diagnosis")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class ValidationResult(BaseModel):
    """Model for validation results."""
    passed_rules: List[str] = Field(default_factory=list, description="Validation rules that passed")
    failed_rules: List[str] = Field(default_factory=list, description="Validation rules that failed")
    notes: str = Field(default="", description="Additional validation notes")


class ExtractionResult(BaseModel):
    """Complete extraction result model."""
    doc_type: str = Field(description="Document type")
    fields: List[FieldExtraction] = Field(default_factory=list, description="Extracted fields")
    structured_data: Optional[Dict[str, Any]] = Field(default=None, description="Structured extracted data")
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence score")
    validation: ValidationResult = Field(default_factory=ValidationResult, description="Validation results")
    processing_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Processing metadata")


# Schema mapping for document types
SCHEMA_MAPPING = {
    "invoice": InvoiceSchema,
    "medical_bill": MedicalBillSchema,
    "prescription": PrescriptionSchema
}
