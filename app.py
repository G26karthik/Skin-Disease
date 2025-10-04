"""
Streamlit Demo Application for AI-Powered Skin Lesion Classifier with XAI.

This interactive web application allows users to upload skin lesion images
and get instant classification with Grad-CAM explanations.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import time

from src.config import CLASS_NAMES, MODEL_PATH, CONFIDENCE_THRESHOLD
from src.inference import predict_from_array, load_model, is_confident
from src.utils import denormalize_image


# Page configuration
st.set_page_config(
    page_title="AI Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_model():
    """Load model and cache it."""
    try:
        model = load_model()
        return model, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def create_probability_chart(probabilities: dict):
    """Create horizontal bar chart for class probabilities."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Dynamic colors for all classes using tab10 colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(probabilities)))
    
    bars = ax.barh(classes, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1%}', 
                ha='left', va='center', fontweight='bold', fontsize=12)
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability', fontweight='bold', fontsize=12)
    ax.set_title('Classification Probabilities', fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">🔬 AI-Powered Skin Lesion Classifier</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explainable AI for Skin Cancer Screening</div>', 
                unsafe_allow_html=True)
    
    # Medical disclaimer
    st.markdown("""
        <div class="warning-box">
            <strong>⚠️ MEDICAL DISCLAIMER</strong><br>
            This tool is for <strong>educational and research purposes only</strong>. 
            It is NOT a substitute for professional medical diagnosis. Always consult 
            a qualified dermatologist or healthcare provider for proper evaluation and 
            treatment of skin lesions.
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, error = load_cached_model()
    
    if error:
        st.error(f"❌ {error}")
        st.info("💡 Please train a model first using: `python -m src.train --data_dir data`")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Minimum confidence level for prediction"
        )
        
        show_gradcam = st.checkbox(
            "Show Grad-CAM Visualization",
            value=True,
            help="Display heatmap showing which regions influenced the prediction"
        )
        
        gradcam_alpha = st.slider(
            "Grad-CAM Overlay Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
            help="Transparency of the heatmap overlay"
        ) if show_gradcam else 0.4
        
        st.markdown("---")
        
        # Class descriptions
        with st.expander("📊 Class Descriptions (HAM10000)"):
            st.markdown("""
                **Melanocytic nevi (nv)**: Common benign moles, ~67% of dataset
                
                **Melanoma (mel)**: Malignant skin cancer, requires immediate treatment
                
                **Benign keratosis (bkl)**: Non-cancerous growths, common in older adults
                
                **Basal cell carcinoma (bcc)**: Malignant but rarely metastasizes
                
                **Actinic keratoses (akiec)**: Pre-cancerous lesions, requires monitoring
                
                **Vascular lesions (vasc)**: Benign blood vessel abnormalities
                
                **Dermatofibroma (df)**: Benign fibrous nodules, typically harmless
            """)
        
        # About section
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
                This application uses a deep learning model (EfficientNet_B0) 
                trained to classify skin lesions. The Grad-CAM visualization 
                highlights the regions the AI model focused on when making 
                its prediction.
                
                **Technology Stack:**
                - PyTorch for deep learning
                - EfficientNet_B0 architecture
                - Grad-CAM for explainability
                - Streamlit for the interface
                
                **Developer:** G. Karthik Koundinya
            """)
        
        # Data & Ethics
        with st.expander("🔒 Data & Ethics"):
            st.markdown("""
                **Privacy:** Images are processed locally and not stored.
                
                **Dataset:** Model trained on HAM10000 dataset (10,015 dermatoscopic images, 7 classes)
                
                **Ethical Considerations:**
                - This tool should augment, not replace, clinical judgment
                - Model performance may vary across different skin types
                - Always seek professional medical advice
                
                **Citation:** Tschandl et al., "The HAM10000 dataset, a large collection of 
                multi-source dermatoscopic images of common pigmented skin lesions" (2018)
                
                **Source:** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
            """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of the skin lesion"
        )
        
        # Sample images (if available)
        st.markdown("---")
        st.subheader("Or try a sample image:")
        
        sample_dir = Path("data/sample")
        if sample_dir.exists():
            sample_images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
            if sample_images:
                sample_cols = st.columns(min(3, len(sample_images)))
                for idx, sample_path in enumerate(sample_images[:3]):
                    with sample_cols[idx]:
                        if st.button(f"Sample {idx+1}", key=f"sample_{idx}"):
                            uploaded_file = sample_path
    
    with col2:
        st.header("🔍 Results")
        
        if uploaded_file is not None:
            # Load image
            if isinstance(uploaded_file, Path):
                image = Image.open(uploaded_file).convert('RGB')
            else:
                image = Image.open(uploaded_file).convert('RGB')
            
            image_array = np.array(image)
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
            
            # Classify button
            if st.button("🚀 Classify", type="primary"):
                with st.spinner("Analyzing image..."):
                    start_time = time.time()
                    
                    # Make prediction
                    result = predict_from_array(
                        image_array,
                        model=model,
                        generate_gradcam=show_gradcam
                    )
                    
                    inference_time = time.time() - start_time
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Predicted class
                predicted_label = result['label']
                confidence = result['confidence']
                
                # Color-code by class (HAM10000 categories)
                label_colors = {
                    'Melanocytic_nevi': '#1f77b4',      # Blue (benign, common)
                    'Melanoma': '#d62728',               # Red (malignant)
                    'Benign_keratosis': '#2ca02c',       # Green (benign)
                    'Basal_cell_carcinoma': '#ff7f0e',   # Orange (malignant but less aggressive)
                    'Actinic_keratoses': '#ffbb00',      # Yellow (pre-cancerous)
                    'Vascular_lesions': '#9467bd',       # Purple (benign)
                    'Dermatofibroma': '#8c564b'          # Brown (benign)
                }
                color = label_colors.get(predicted_label, '#17a2b8')
                
                # Display prediction
                if confidence >= confidence_threshold:
                    st.markdown(f"""
                        <div class="success-box">
                            <h3 style="color: {color}; margin: 0;">Prediction: {predicted_label}</h3>
                            <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                                Confidence: <strong>{confidence:.1%}</strong>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="warning-box">
                            <h3 style="margin: 0;">Prediction: {predicted_label}</h3>
                            <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                                Confidence: <strong>{confidence:.1%}</strong>
                            </p>
                            <p style="margin: 0;">
                                ⚠️ Low confidence - please consult a healthcare professional
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Probability chart
                st.pyplot(create_probability_chart(result['class_probabilities']))
                
                # Grad-CAM visualization
                if show_gradcam and 'gradcam_overlay' in result:
                    st.markdown("---")
                    st.subheader("🎯 Explainability (Grad-CAM)")
                    
                    st.info("""
                        The heatmap below shows which parts of the image the AI model 
                        focused on when making its prediction. Red/yellow areas indicate 
                        regions of high importance.
                    """)
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Overlay", "Heatmap", "Original"])
                    
                    with tab1:
                        st.image(result['gradcam_overlay'], 
                                caption="Grad-CAM Overlay",
                                use_column_width=True)
                    
                    with tab2:
                        # Show just the heatmap
                        from src.gradcam import create_gradcam_visualization
                        _, heatmap_colored = create_gradcam_visualization(
                            image_array, 
                            result['gradcam_heatmap']
                        )
                        st.image(heatmap_colored,
                                caption="Grad-CAM Heatmap",
                                use_column_width=True)
                    
                    with tab3:
                        st.image(image_array,
                                caption="Original Image",
                                use_column_width=True)
                
                # Inference time
                st.markdown("---")
                st.caption(f"⏱️ Inference time: {inference_time:.3f} seconds")
                
        else:
            st.info("👆 Please upload an image or select a sample to begin")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #888; padding: 2rem;">
            <p>Developed by <strong>G. Karthik Koundinya</strong></p>
            <p>Built with PyTorch, EfficientNet, and Streamlit | 
            <a href="https://github.com" target="_blank">GitHub</a></p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
