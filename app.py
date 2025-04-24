import streamlit as st
import os
import tempfile
from PIL import Image
from deepface import DeepFace
from fpdf import FPDF
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import atexit
import numpy as np

# Configure environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# App configuration
st.set_page_config(
    page_title="AI Face Recognition & Insight App",
    page_icon=":camera:",
    layout="wide"
)

# Initialize session state
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'reference_img' not in st.session_state:
    st.session_state.reference_img = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Pre-load models at startup
@st.cache_resource
def load_models():
    try:
        models = {
            "Age": DeepFace.build_model("Age"),
            "Gender": DeepFace.build_model("Gender"),
            "Emotion": DeepFace.build_model("Emotion"),
            "Race": DeepFace.build_model("Race")
        }
        st.session_state.models_loaded = True
        return models
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return {}

models = load_models()

# File uploader - multiple images
uploaded_files = st.file_uploader(
    "Upload Images (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files and len(uploaded_files) > 0:
    st.session_state.uploaded_images = []
    
    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            max_size = (1024, 1024)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
            st.session_state.uploaded_images.append({
                'name': uploaded_file.name,
                'image': img,
                'analysis': None,
                'comparison': None
            })
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

# Display image gallery
if st.session_state.uploaded_images:
    st.header("Image Gallery")
    cols = st.columns(min(4, len(st.session_state.uploaded_images)))
    
    for idx, img_data in enumerate(st.session_state.uploaded_images):
        with cols[idx % len(cols)]:
            st.image(img_data['image'], use_column_width=True, caption=img_data['name'])
            if st.button(f"Set as Reference", key=f"ref_{idx}"):
                st.session_state.reference_img = idx
                st.success(f"Set {img_data['name']} as reference image")

# Analysis functions
def analyze_image(img, attributes):
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name, format="JPEG", quality=90)
            analysis = DeepFace.analyze(
                img_path=tmp.name,
                actions=[attr.lower() for attr in attributes],
                enforce_detection=False,
                detector_backend="opencv"
            )
        os.unlink(tmp.name)
        return analysis[0] if isinstance(analysis, list) else analysis
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

def compare_faces(img1, img2):
    try:
        with tempfile.NamedTemporaryFile(suffix="1.jpg", delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix="2.jpg", delete=False) as tmp2:
            
            img1.save(tmp1.name, format="JPEG", quality=90)
            img2.save(tmp2.name, format="JPEG", quality=90)
            
            result = DeepFace.verify(
                img1_path=tmp1.name,
                img2_path=tmp2.name,
                detector_backend="opencv",
                model_name="VGG-Face",
                enforce_detection=False
            )
        os.unlink(tmp1.name)
        os.unlink(tmp2.name)
        return result
    except Exception as e:
        st.error(f"Comparison failed: {str(e)}")
        return None

# Run analysis
if st.session_state.uploaded_images and st.button("Run Analysis"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(st.session_state.uploaded_images)
    
    try:
        for idx, img_data in enumerate(st.session_state.uploaded_images):
            status_text.text(f"Analyzing {img_data['name']}...")
            analysis = analyze_image(img_data['image'], ["Age", "Gender", "Emotion", "Race"])
            if analysis:
                st.session_state.uploaded_images[idx]['analysis'] = analysis
            progress_bar.progress((idx + 1) / total)
        
        if st.session_state.reference_img is not None:
            ref_img = st.session_state.uploaded_images[st.session_state.reference_img]['image']
            for idx, img_data in enumerate(st.session_state.uploaded_images):
                if idx == st.session_state.reference_img:
                    continue
                status_text.text(f"Comparing with {img_data['name']}...")
                comparison = compare_faces(ref_img, img_data['image'])
                if comparison:
                    st.session_state.uploaded_images[idx]['comparison'] = comparison
                progress_bar.progress((idx + 1) / total)
        
        st.success("Analysis completed!")
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

# Display results
if st.session_state.uploaded_images and any(img.get('analysis') for img in st.session_state.uploaded_images):
    st.header("Analysis Results")
    
    for idx, img_data in enumerate(st.session_state.uploaded_images):
        if not img_data['analysis']:
            continue
            
        with st.expander(f"Results for {img_data['name']}"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(img_data['image'], use_column_width=True)
                
            with col2:
                analysis = img_data['analysis']
                
                # Age
                if 'age' in analysis:
                    st.metric("Age", f"{analysis['age']} years")
                
                # Gender
                if 'gender' in analysis:
                    gender, confidence = max(analysis['gender'].items(), key=lambda x: x[1])
                    st.metric("Gender", f"{gender} ({confidence:.1f}%)")
                
                # Emotion
                if 'emotion' in analysis:
                    emotion, confidence = max(analysis['emotion'].items(), key=lambda x: x[1])
                    st.metric("Dominant Emotion", f"{emotion} ({confidence:.1f}%)")
                    
                    fig, ax = plt.subplots()
                    ax.bar(analysis['emotion'].keys(), analysis['emotion'].values())
                    st.pyplot(fig)
                    plt.close()
                
                # Comparison
                if idx != st.session_state.reference_img and img_data.get('comparison'):
                    similarity = 1 - img_data['comparison']['distance']
                    st.metric("Similarity Score", f"{similarity:.1%}")
