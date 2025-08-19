# Agentic Document Extraction System

A modular AI-powered document extraction system built with LangChain, OpenAI GPT, and Streamlit. The system automatically detects document types (invoices, medical bills, prescriptions) and extracts structured data with confidence scoring.

## 🚀 Features

- **Document Type Detection**: Automatically classifies documents into invoice, medical_bill, or prescription
- **OCR Support**: Handles both text-based PDFs and scanned images using pdfplumber, PyMuPDF, and pytesseract
- **Structured Extraction**: Uses OpenAI GPT with Pydantic schemas for reliable data extraction
- **Validation Layer**: Comprehensive validation with regex checks and business rules
- **Confidence Scoring**: Multi-factor confidence scoring combining LLM, OCR, and validation results
- **Interactive UI**: Clean Streamlit interface with real-time processing and visualization
- **Batch Processing**: Support for processing multiple documents

## 🛠 Tech Stack

- **AI/ML**: LangChain, OpenAI GPT-4
- **OCR**: pdfplumber, PyMuPDF, pytesseract
- **UI**: Streamlit
- **Data Models**: Pydantic
- **Image Processing**: OpenCV, Pillow
- **Language**: Python 3.8+

## 📂 Project Structure

```
agentic_doc_extraction/
│── data/                     # Sample documents
│── outputs/                  # Extracted results
│── src/
│   │── __init__.py
│   │── agent.py               # Main orchestrator using LangChain
│   │── routing.py             # Document type detection
│   │── ocr.py                 # OCR + preprocessing
│   │── extraction.py          # LLM extraction chain
│   │── validation.py          # Regex + business rules
│   │── scoring.py             # Confidence scoring logic
│   │── ui_streamlit.py        # Streamlit frontend
│   │── schemas.py             # Pydantic models & JSON schemas
│   │── utils.py               # Helper functions
│── requirements.txt
│── README.md
```

## 🔧 Setup Instructions

### 1. Clone and Install Dependencies

```bash
# Clone the repository
cd agentic_doc_extraction

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

**Option 1: Using .env file (Recommended)**

1. Copy the example environment file:
```bash
copy .env.example .env
```

2. Edit `.env` file and add your OpenAI API key:
```env
OPENAI_API_KEY=your_actual_openai_api_key_here
OPENAI_MODEL_GPT4=gpt-4o
OPENAI_MODEL_GPT4_MINI=gpt-4o-mini
DEFAULT_OUTPUT_DIR=outputs
```

**Option 2: Environment Variables**

Set the environment variable directly:

```bash
# Windows
set OPENAI_API_KEY=your_openai_api_key_here

# macOS/Linux
export OPENAI_API_KEY=your_openai_api_key_here
```

**Get your OpenAI API Key:**
1. Visit https://platform.openai.com/api-keys
2. Create a new API key
3. Copy and paste it into your `.env` file

### 3. Install OCR Dependencies

For pytesseract (OCR engine):

**Windows:**
1. Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Add Tesseract to your PATH or set the path in your code

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

## 🚀 Run Instructions

**Note**: Make sure you have configured your OpenAI API key in the `.env` file before running the application.

### Option 1: Streamlit UI (Recommended)

**Method 1: Using the run script (Recommended)**
```bash
python run_app.py
```

**Method 2: From src directory**
```bash
cd src
streamlit run ui_streamlit.py
```

**Method 3: Direct streamlit command**
```bash
streamlit run src/ui_streamlit.py
```

Then open your browser to `http://localhost:8501`

The Streamlit UI will automatically load your API key from the `.env` file.

### Option 2: Python API

```python
from src.agent import process_document_pipeline

# Process a single document (API key loaded from .env automatically)
results = process_document_pipeline(
    file_path="path/to/your/document.pdf",
    save_results=True
)

print(f"Document type: {results['doc_type']}")
print(f"Confidence: {results['overall_confidence']:.2f}")

# Or pass API key explicitly if needed
results = process_document_pipeline(
    file_path="path/to/your/document.pdf",
    api_key="your_openai_api_key",  # Optional if set in .env
    save_results=True
)
```

### Option 3: Batch Processing

```python
from src.agent import DocumentExtractionAgent

# API key loaded from .env automatically
agent = DocumentExtractionAgent()

# Process multiple documents
file_paths = ["doc1.pdf", "doc2.png", "doc3.pdf"]
results = agent.batch_process_documents(file_paths)
```

## 📄 Supported Document Types

### 1. Invoices
- Invoice number, date, due date
- Vendor and customer information
- Line items with quantities, prices
- Subtotal, tax, total amounts

### 2. Medical Bills
- Patient information
- Hospital/clinic details
- Service dates and descriptions
- Insurance information
- Billing amounts and balances

### 3. Prescriptions
- Patient and doctor information
- Prescription date
- Medications with dosages
- Instructions and notes

## 📊 Example Output

```json
{
  "doc_type": "invoice",
  "fields": [
    {
      "name": "invoice_number",
      "value": "INV-2024-001",
      "confidence": 0.95,
      "source": {"page": 1, "bbox": [100, 150, 250, 180]}
    },
    {
      "name": "vendor_name",
      "value": "ABC Corporation", 
      "confidence": 0.92,
      "source": {"page": 1, "bbox": [100, 200, 300, 230]}
    },
    {
      "name": "total_amount",
      "value": "1250.00",
      "confidence": 0.88,
      "source": {"page": 1, "bbox": [400, 500, 500, 530]}
    }
  ],
  "overall_confidence": 0.91,
  "validation": {
    "passed_rules": ["totals_match", "date_format_valid"],
    "failed_rules": [],
    "notes": "Document appears valid"
  },
  "structured_data": {
    "invoice_number": "INV-2024-001",
    "vendor_name": "ABC Corporation",
    "total_amount": 1250.00,
    "line_items": [
      {
        "description": "Software License",
        "quantity": 1,
        "unit_price": 1000.00,
        "total_price": 1000.00
      }
    ]
  }
}
```

## 🎯 Confidence Scoring

The system uses a weighted confidence scoring approach:

```
confidence = (llm_score * 0.6) + (ocr_quality * 0.2) + (validation_pass * 0.2)
```

- **LLM Score**: Self-reported confidence from the language model
- **OCR Quality**: Quality of text extraction (1.0 for text PDFs, variable for scanned images)
- **Validation Pass**: Whether extracted data passes validation rules

## 🔍 Validation Rules

### Invoice Validation
- Invoice number present
- Valid total amounts
- Line items sum matches total
- Valid date formats

### Medical Bill Validation
- Patient name present
- Hospital information available
- Balance calculations correct
- Valid service dates

### Prescription Validation
- Patient and doctor information
- Medication list with dosages
- Valid prescription date
- Proper dosage formats

## 🛡️ Error Handling

The system includes comprehensive error handling:

- **OCR Failures**: Graceful degradation with multiple OCR approaches
- **LLM Failures**: Fallback to keyword-based extraction
- **Validation Errors**: Non-blocking warnings and confidence adjustments
- **File Format Issues**: Clear error messages and supported format guidance

## 🔧 Customization

### Adding New Document Types

1. Add schema to `schemas.py`:
```python
class NewDocumentSchema(BaseModel):
    field1: Optional[str] = None
    field2: Optional[float] = None
```

2. Update routing in `routing.py`
3. Add validation rules in `validation.py`
4. Update UI in `ui_streamlit.py`

### Adjusting Confidence Weights

```python
from src.scoring import ConfidenceScorer

scorer = ConfidenceScorer(
    llm_weight=0.7,      # Increase LLM importance
    ocr_weight=0.1,      # Decrease OCR importance  
    validation_weight=0.2
)
```

## 📝 Troubleshooting

### Import Errors

If you encounter import errors when running the application:

1. **Test your setup first:**
   ```bash
   python test_setup.py
   ```

2. **Use the recommended run script:**
   ```bash
   python run_app.py
   ```

3. **If running Streamlit directly, run from src directory:**
   ```bash
   cd src
   streamlit run ui_streamlit.py
   ```

### Common Issues

1. **OpenAI API Key**: Ensure your API key is valid and has sufficient credits
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Tesseract OCR**: Make sure Tesseract is properly installed and in PATH
4. **File Formats**: Check that uploaded files are in supported formats
5. **Memory**: Large PDF files may require sufficient system memory

### Optional Dependencies

Some features require optional dependencies:

- **pdfplumber**: For advanced PDF table extraction
- **pytesseract**: For OCR functionality on scanned documents  
- **opencv-python**: For advanced image preprocessing

Install them with:
```bash
pip install pdfplumber pytesseract opencv-python
```

The system will work with reduced functionality if these are not available.

### Performance Tips

- Use text-based PDFs when possible (faster than OCR)
- Optimize image resolution for OCR (300 DPI recommended)
- Process documents in batches for efficiency
- Consider using smaller LLM models for faster processing

## 📈 Future Enhancements

- Support for additional document types
- Multi-language document support
- Advanced table extraction
- Integration with cloud storage
- API endpoints for integration
- Real-time processing monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain for the orchestration framework
- OpenAI for the language models
- Streamlit for the web interface
- The open-source OCR community
