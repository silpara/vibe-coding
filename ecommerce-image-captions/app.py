import streamlit as st
import tempfile
import os
import json
from image_captioner import ImageCaptioner

st.set_page_config(
    page_title="E-commerce Image Captioner",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Keep sidebar collapsed by default
)

# Initialize session state for caption data
if 'caption_data' not in st.session_state:
    st.session_state.caption_data = None

st.title("E-commerce Image Captioner")
st.write("Upload a product image to generate detailed captions with cross-sell suggestions.")

# Initialize the image captioner
captioner = ImageCaptioner()

# Get available models
available_models = captioner.get_available_models()
if not available_models:
    st.error("Could not connect to Ollama service. Please ensure Ollama is running.")
    st.stop()

# Model selection in sidebar
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox(
        "Select Model",
        available_models,
        index=available_models.index("gemma3") if "gemma3" in available_models else 0
    )
    captioner.set_model(selected_model)

# Create two columns for layout
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# Only show the right column content if an image is uploaded
if uploaded_file is not None:
    # Process the image
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Show spinner in the right column at the top
        with right_col:
            with st.spinner("Analyzing image..."):
                caption = captioner.generate_caption(tmp_file_path)
                # Store caption data in session state
                st.session_state.caption_data = caption.to_dict()
            
            st.subheader("Product Details")
            st.write(f"**Product Name:** {caption.product_name}")
            st.write(f"**Description:** {caption.description}")
            
            st.subheader("Attributes")
            for attr in caption.attributes:
                st.write(f"- **{attr.name}:** {attr.value} (Confidence: {attr.confidence:.2f})")
            
            st.subheader("User Needs")
            st.write("Key benefits this product provides:")
            for need in caption.user_needs:
                st.write(f"- **{need}**")
            
            st.subheader("Cross-sell Suggestions")
            for product in caption.cross_sell_products:
                with st.expander(f"{product.category} - {product.product_name}"):
                    st.write(f"**Search Query:** `{product.search_query}`")
                    st.write(f"**Description:** {product.description}")
                    st.write("**Attributes:**")
                    for attr in product.attributes:
                        st.write(f"- {attr.name}: {attr.value} (Confidence: {attr.confidence:.2f})")
        
        # Show JSON viewer and download button in left column after caption generation
        with left_col:
            st.markdown("---")  # Add a visual separator
            with st.expander("View Raw JSON"):
                st.code(json.dumps(st.session_state.caption_data, indent=2), language="json")
            
            # Download button below JSON viewer
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                json_str = json.dumps(st.session_state.caption_data, indent=2)
                st.download_button(
                    label="📥 Download Caption Data (JSON)",
                    data=json_str,
                    file_name="caption_data.json",
                    mime="application/json",
                    help="Download the caption data in JSON format"
                )
    
    except Exception as e:
        with right_col:
            st.error(f"Error generating caption: {str(e)}")
    
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path) 