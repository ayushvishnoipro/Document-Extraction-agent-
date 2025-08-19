"""
Streamlit UI for the Agentic Document Extraction System.
Provides a clean, interactive interface for document upload, processing, and results visualization.
"""

import streamlit as st
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our modules with better error handling
try:
    # Try relative imports first (when run as module)
    from .agent import DocumentExtractionAgent
    from .utils import get_file_info
except ImportError:
    try:
        # Try absolute imports (when run directly)
        from agent import DocumentExtractionAgent
        from utils import get_file_info
    except ImportError:
        # Add current directory to path and try again
        import sys
        from pathlib import Path
        
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        try:
            from agent import DocumentExtractionAgent
            from utils import get_file_info
        except ImportError as e:
            st.error(f"""
            **Import Error**: {str(e)}
            
            **Possible Solutions:**
            1. Run the app using: `python run_app.py` from the project root
            2. Or run from src directory: `cd src && streamlit run ui_streamlit.py`
            3. Install missing dependencies: `pip install -r requirements.txt`
            """)
            st.stop()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Agentic Document Extraction",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .confidence-high {
        color: #27ae60;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .confidence-low {
        color: #e74c3c;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">📄 Agentic Document Extraction System</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    setup_sidebar()
    
    # Main content area
    uploaded_file = st.file_uploader(
        "Upload a document (PDF or Image)",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
        help="Supported formats: PDF, PNG, JPG, JPEG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        process_uploaded_document(uploaded_file)
    else:
        show_welcome_message()


def setup_sidebar():
    """Setup the sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")
    
    # Check if API key is already in environment
    env_api_key = os.getenv("OPENAI_API_KEY")
    
    # OpenAI API Key input
    if env_api_key and env_api_key != "your_openai_api_key_here":
        st.sidebar.success("✅ API Key loaded from .env file")
        api_key = env_api_key
        # Removed the masked key display for security
    else:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key for document processing"
        )
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.sidebar.success("✅ API Key configured")
        else:
            st.sidebar.warning("⚠️ Please enter your OpenAI API key or configure it in .env file")
    
    # Processing options
    st.sidebar.subheader("Processing Options")
    
    save_results = st.sidebar.checkbox(
        "Save results to file",
        value=True,
        help="Save extraction results as JSON file"
    )
    
    # Get default output directory from environment
    default_output_dir = os.getenv("DEFAULT_OUTPUT_DIR", "outputs")
    output_dir = st.sidebar.text_input(
        "Output directory",
        value=default_output_dir,
        help="Directory to save extraction results"
    )
    
    # Store settings in session state
    st.session_state.save_results = save_results
    st.session_state.output_dir = output_dir
    st.session_state.api_key = api_key
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This system uses AI to extract structured data from documents:
    - **Invoice** extraction
    - **Medical Bill** processing  
    - **Prescription** analysis
    
    Powered by LangChain and OpenAI GPT.
    
    💡 **Tip**: Configure your API key in the `.env` file for automatic loading.
    """)


def show_welcome_message():
    """Show welcome message and instructions."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 1rem; margin: 2rem 0;">
        <h3>Welcome to the Document Extraction System</h3>
        <p>Upload a document to get started with AI-powered data extraction.</p>
        <p><strong>Supported document types:</strong></p>
        <ul style="list-style: none; padding: 0;">
            <li>📄 <strong>Invoices</strong> - Extract vendor info, line items, totals</li>
            <li>🏥 <strong>Medical Bills</strong> - Extract patient info, services, charges</li>
            <li>💊 <strong>Prescriptions</strong> - Extract patient, doctor, medications</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Example results section
    with st.expander("📊 See Example Output", expanded=False):
        example_json = {
            "doc_type": "invoice",
            "fields": [
                {
                    "name": "invoice_number",
                    "value": "INV-2024-001",
                    "confidence": 0.95
                },
                {
                    "name": "vendor_name", 
                    "value": "ABC Corporation",
                    "confidence": 0.92
                },
                {
                    "name": "total_amount",
                    "value": "1250.00",
                    "confidence": 0.88
                }
            ],
            "overall_confidence": 0.91,
            "validation": {
                "passed_rules": ["totals_match", "date_format_valid"],
                "failed_rules": [],
                "notes": "Document appears valid"
            }
        }
        
        st.json(example_json)


def process_uploaded_document(uploaded_file):
    """Process the uploaded document."""
    # Check if API key is configured
    api_key = st.session_state.get("api_key") or os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your_openai_api_key_here":
        st.error("⚠️ Please configure your OpenAI API key in the sidebar or in the .env file before processing.")
        st.info("💡 **Setup Instructions:**\n1. Create a `.env` file in the project root\n2. Add: `OPENAI_API_KEY=your_actual_api_key`\n3. Restart the application")
        return
    
    # File information
    file_info = {
        "name": uploaded_file.name,
        "size": uploaded_file.size,
        "type": uploaded_file.type
    }
    
    st.markdown('<div class="section-header">📋 Document Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Filename", file_info["name"])
    with col2:
        st.metric("📊 Size", f"{file_info['size'] / 1024:.1f} KB")
    with col3:
        st.metric("🏷️ Type", file_info["type"])
    
    # Process button
    if st.button("🚀 Process Document", type="primary", use_container_width=True):
        process_document_workflow(uploaded_file, file_info)


def process_document_workflow(uploaded_file, file_info: Dict[str, Any]):
    """Execute the document processing workflow."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Initialize the extraction agent
        agent = DocumentExtractionAgent(
            api_key=st.session_state.get("api_key"),
            output_dir=st.session_state.get("output_dir", "outputs")
        )
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: OCR and text extraction
        status_text.text("🔍 Step 1/5: Extracting text from document...")
        progress_bar.progress(20)
        
        # Step 2: Document type detection
        status_text.text("🏷️ Step 2/5: Detecting document type...")
        progress_bar.progress(40)
        
        # Step 3: Structured extraction
        status_text.text("🧠 Step 3/5: Extracting structured data with AI...")
        progress_bar.progress(60)
        
        # Step 4: Validation
        status_text.text("✅ Step 4/5: Validating extracted data...")
        progress_bar.progress(80)
        
        # Step 5: Confidence scoring
        status_text.text("📊 Step 5/5: Calculating confidence scores...")
        progress_bar.progress(90)
        
        # Process the document
        results = agent.process_document(
            file_path=tmp_file_path,
            save_results=st.session_state.get("save_results", True)
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Processing completed!")
        
        # Display results
        display_extraction_results(results)
        
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.exception(e)
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except:
            pass


def display_extraction_results(results: Dict[str, Any]):
    """Display the extraction results in an organized manner."""
    st.markdown('<div class="section-header">🎯 Extraction Results</div>', unsafe_allow_html=True)
    
    # Overview metrics
    display_overview_metrics(results)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📋 Extracted Data", 
        "✅ Validation", 
        "📈 Confidence Analysis",
        "💾 Raw JSON"
    ])
    
    with tab1:
        display_overview_tab(results)
    
    with tab2:
        display_extracted_data_tab(results)
    
    with tab3:
        display_validation_tab(results)
    
    with tab4:
        display_confidence_tab(results)
    
    with tab5:
        display_raw_json_tab(results)


def display_overview_metrics(results: Dict[str, Any]):
    """Display overview metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    doc_type = results.get("doc_type", "unknown").replace("_", " ").title()
    confidence = results.get("overall_confidence", 0.0)
    field_count = len(results.get("fields", []))
    validation_status = "✅ Valid" if results.get("validation", {}).get("passed_rules", []) else "⚠️ Issues"
    
    with col1:
        st.metric("📄 Document Type", doc_type)
    
    with col2:
        confidence_color = get_confidence_color(confidence)
        st.markdown(f'<div class="metric-card"><strong>🎯 Overall Confidence</strong><br><span class="{confidence_color}">{confidence:.1%}</span></div>', unsafe_allow_html=True)
    
    with col3:
        st.metric("📋 Fields Extracted", field_count)
    
    with col4:
        st.metric("✅ Validation", validation_status)


def display_overview_tab(results: Dict[str, Any]):
    """Display overview information."""
    st.subheader("📊 Processing Summary")
    
    processing_metadata = results.get("processing_metadata", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Processing Details:**")
        st.write(f"- Processing Time: {processing_metadata.get('processing_time_seconds', 0):.2f} seconds")
        st.write(f"- Extraction Method: {processing_metadata.get('extraction_metadata', {}).get('extraction_method', 'unknown')}")
        st.write(f"- OCR Quality: {processing_metadata.get('extraction_metadata', {}).get('ocr_quality', 0):.1%}")
        st.write(f"- Source File: {Path(processing_metadata.get('source_file', '')).name}")
    
    with col2:
        st.write("**Document Analysis:**")
        doc_type = results.get("doc_type", "unknown")
        st.write(f"- Document Type: {doc_type.replace('_', ' ').title()}")
        st.write(f"- Fields Detected: {len(results.get('fields', []))}")
        
        validation = results.get("validation", {})
        passed = len(validation.get("passed_rules", []))
        failed = len(validation.get("failed_rules", []))
        st.write(f"- Validation: {passed} passed, {failed} failed")


def display_extracted_data_tab(results: Dict[str, Any]):
    """Display extracted data in a structured format."""
    st.subheader("📋 Extracted Fields")
    
    fields = results.get("fields", [])
    structured_data = results.get("structured_data", {})
    
    if structured_data:
        # Display structured data by category
        display_structured_data(structured_data)
    
    if fields:
        st.subheader("📊 All Extracted Fields")
        
        # Create DataFrame for field display
        field_data = []
        for field in fields:
            field_data.append({
                "Field": field.get("name", ""),
                "Value": field.get("value", ""),
                "Confidence": field.get("confidence", 0.0)
            })
        
        df = pd.DataFrame(field_data)
        
        # Apply confidence color coding
        def color_confidence(val):
            if val >= 0.8:
                return 'background-color: #d4edda; color: #155724;'
            elif val >= 0.5:
                return 'background-color: #fff3cd; color: #856404;'
            else:
                return 'background-color: #f8d7da; color: #721c24;'
        
        styled_df = df.style.applymap(color_confidence, subset=['Confidence'])
        st.dataframe(styled_df, use_container_width=True)


def display_structured_data(data: Dict[str, Any], prefix: str = ""):
    """Display structured data in an organized way."""
    for key, value in data.items():
        if key.endswith("_confidence"):
            continue
            
        display_key = f"{prefix}.{key}" if prefix else key
        display_key = display_key.replace("_", " ").title()
        
        if isinstance(value, dict):
            st.subheader(f"📁 {display_key}")
            display_structured_data(value, key)
        elif isinstance(value, list) and value:
            st.subheader(f"📋 {display_key}")
            
            if isinstance(value[0], dict):
                # Display as table if list of dictionaries
                df = pd.DataFrame(value)
                if "confidence" in df.columns:
                    df["confidence"] = df["confidence"].apply(lambda x: f"{x:.1%}")
                st.dataframe(df, use_container_width=True)
            else:
                # Display as bullet points
                for item in value:
                    st.write(f"• {item}")
        elif value is not None and str(value).strip():
            # Get confidence if available
            confidence_key = f"{key}_confidence"
            confidence = data.get(confidence_key, 0.5)
            confidence_color = get_confidence_color(confidence)
            
            st.markdown(f"**{display_key}:** {value} <span class='{confidence_color}'>({confidence:.1%})</span>", unsafe_allow_html=True)


def display_validation_tab(results: Dict[str, Any]):
    """Display validation results."""
    st.subheader("✅ Validation Results")
    
    validation = results.get("validation", {})
    passed_rules = validation.get("passed_rules", [])
    failed_rules = validation.get("failed_rules", [])
    notes = validation.get("notes", "")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if passed_rules:
            st.success(f"✅ Passed Rules ({len(passed_rules)})")
            for rule in passed_rules:
                st.write(f"• {rule.replace('_', ' ').title()}")
        else:
            st.info("No validation rules passed")
    
    with col2:
        if failed_rules:
            st.error(f"❌ Failed Rules ({len(failed_rules)})")
            for rule in failed_rules:
                st.write(f"• {rule.replace('_', ' ').title()}")
        else:
            st.success("No validation failures")
    
    if notes:
        st.info(f"📝 Notes: {notes}")


def display_confidence_tab(results: Dict[str, Any]):
    """Display confidence analysis."""
    st.subheader("📈 Confidence Analysis")
    
    overall_confidence = results.get("overall_confidence", 0.0)
    fields = results.get("fields", [])
    
    # Overall confidence display
    confidence_color = get_confidence_color(overall_confidence)
    st.markdown(f'<div style="text-align: center; font-size: 2rem; margin: 1rem 0;"><span class="{confidence_color}">Overall Confidence: {overall_confidence:.1%}</span></div>', unsafe_allow_html=True)
    
    # Confidence distribution
    if fields:
        confidences = [field.get("confidence", 0.0) for field in fields]
        
        high_conf = sum(1 for c in confidences if c >= 0.8)
        med_conf = sum(1 for c in confidences if 0.5 <= c < 0.8)
        low_conf = sum(1 for c in confidences if c < 0.5)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🟢 High Confidence", f"{high_conf} fields", delta=f"{high_conf/len(confidences):.1%}")
        
        with col2:
            st.metric("🟡 Medium Confidence", f"{med_conf} fields", delta=f"{med_conf/len(confidences):.1%}")
        
        with col3:
            st.metric("🔴 Low Confidence", f"{low_conf} fields", delta=f"{low_conf/len(confidences):.1%}")
        
        # Confidence histogram
        st.subheader("📊 Confidence Distribution")
        
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist_data = pd.cut(confidences, bins=bins, labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]).value_counts()
        
        st.bar_chart(hist_data)


def display_raw_json_tab(results: Dict[str, Any]):
    """Display raw JSON results."""
    st.subheader("💾 Raw JSON Output")
    
    # Download button
    json_str = json.dumps(results, indent=2, default=str)
    
    st.download_button(
        label="📥 Download JSON",
        data=json_str,
        file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )
    
    # Display JSON
    st.json(results)


def get_confidence_color(confidence: float) -> str:
    """Get CSS class for confidence color coding."""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"


if __name__ == "__main__":
    main()

