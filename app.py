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
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Better memory management

# App configuration
st.set_page_config(
    page_title="AI Face Recognition & Insight App",
    page_icon=":camera:",
    layout="wide"
)

# Cleanup function for temp files
def cleanup():
    if 'uploaded_images' in st.session_state:
        for img_data in st.session_state.uploaded_images:
            if img_data.get('temp_path') and os.path.exists(img_data['temp_path']):
                try:
                    os.unlink(img_data['temp_path'])
                except:
                    pass

# Register cleanup function
atexit.register(cleanup)

# Title and description
st.title("AI Face Recognition & Insight App")
st.markdown("""
    Upload multiple images to analyze facial attributes and compare faces.
    Select one as reference to compare against others.
""")

# Initialize session state
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'reference_img' not in st.session_state:
    st.session_state.reference_img = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    analysis_options = st.multiselect(
        "Select attributes to analyze",
        ["Age", "Gender", "Emotion", "Race"],
        default=["Age", "Gender"]
    )
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.6, 0.01)
    enforce_detection = st.checkbox("Enforce Face Detection", value=True)
    detector_backend = st.selectbox(
        "Detection Backend",
        ["opencv", "ssd", "dlib", "mtcnn", "retinaface"],
        index=0
    )
    
    # Pre-load models button
    if st.button("Pre-load Models (Recommended)"):
        with st.spinner("Loading models..."):
            try:
                for model in analysis_options:
                    DeepFace.build_model(model)
                st.session_state.models_loaded = True
                st.success("Models loaded successfully!")
            except Exception as e:
                st.error(f"Model loading failed: {str(e)}")

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
            # Convert to RGB if necessary and resize if too large
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if image is too large to prevent memory issues
            max_size = (1024, 1024)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
            st.session_state.uploaded_images.append({
                'name': uploaded_file.name,
                'image': img,
                'analysis': None,
                'comparison': None,
                'temp_path': None
            })
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

# Display image gallery if images are uploaded
if st.session_state.uploaded_images:
    st.header("Image Gallery")
    cols = st.columns(min(4, len(st.session_state.uploaded_images)))
    
    for idx, img_data in enumerate(st.session_state.uploaded_images):
        with cols[idx % len(cols)]:
            st.image(img_data['image'], use_column_width=True, caption=img_data['name'])
            if st.button(f"Set as Reference", key=f"ref_{idx}"):
                st.session_state.reference_img = idx
                st.success(f"Set {img_data['name']} as reference image")

# Show current reference image
if st.session_state.reference_img is not None:
    ref_data = st.session_state.uploaded_images[st.session_state.reference_img]
    st.header("Reference Image")
    st.image(ref_data['image'], width=300, caption=f"Reference: {ref_data['name']}")

# Analysis and comparison functions
def analyze_image(img_path, attributes):
    try:
        analysis = DeepFace.analyze(
            img_path=img_path,
            actions=[attr.lower() for attr in attributes],
            enforce_detection=enforce_detection,
            detector_backend=detector_backend,
            silent=True  # Remove progress bar parameter
        )
        return analysis[0] if isinstance(analysis, list) else analysis
    except Exception as e:
        st.error(f"Analysis failed for {img_path}: {str(e)}")
        return None

def compare_faces(img1_path, img2_path):
    try:
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            detector_backend=detector_backend,
            model_name='VGG-Face',
            distance_metric='cosine',
            enforce_detection=enforce_detection,
            align=True
        )
        return result
    except Exception as e:
        st.error(f"Comparison failed: {str(e)}")
        return None

def create_pdf_report(images_data, reference_idx):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Face Recognition Analysis Report", 0, 1, 'C')
    pdf.ln(10)
    
    # Reference image
    if reference_idx is not None:
        ref_data = images_data[reference_idx]
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Reference Image: {ref_data['name']}", 0, 1)
        
        # Add reference image to PDF
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            ref_data['image'].save(tmp.name, format="JPEG", quality=85)
            pdf.image(tmp.name, x=10, w=60)
        
        # Add reference analysis
        if ref_data['analysis']:
            pdf.ln(5)
            pdf.set_font("Arial", '', 10)
            for key, value in ref_data['analysis'].items():
                if key.lower() in ['region', 'face_confidence']:
                    continue
                if isinstance(value, dict):
                    pdf.cell(0, 6, f"{key}: {max(value.items(), key=lambda x: x[1])[0]}", 0, 1)
                else:
                    pdf.cell(0, 6, f"{key}: {value}", 0, 1)
    
    # Comparisons
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Image Comparisons", 0, 1)
    
    for idx, img_data in enumerate(images_data):
        if idx == reference_idx:
            continue
        
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Image: {img_data['name']}", 0, 1)
        
        # Add image to PDF
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            img_data['image'].save(tmp.name, format="JPEG", quality=85)
            pdf.image(tmp.name, x=10, w=60)
        
        # Add comparison results
        if img_data.get('comparison'):
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 6, f"Match: {'Yes' if img_data['comparison']['verified'] else 'No'}", 0, 1)
            pdf.cell(0, 6, f"Similarity Score: {1 - img_data['comparison']['distance']:.2f}", 0, 1)
            pdf.cell(0, 6, f"Threshold: {img_data['comparison']['threshold']:.2f}", 0, 1)
        
        # Add analysis
        if img_data['analysis']:
            pdf.ln(5)
            for key, value in img_data['analysis'].items():
                if key.lower() in ['region', 'face_confidence']:
                    continue
                if isinstance(value, dict):
                    pdf.cell(0, 6, f"{key}: {max(value.items(), key=lambda x: x[1])[0]}", 0, 1)
                else:
                    pdf.cell(0, 6, f"{key}: {value}", 0, 1)
    
    # Save to bytes buffer
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# Run analysis and comparisons
if st.session_state.uploaded_images and st.button("Run Analysis"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = len(st.session_state.uploaded_images)
    if st.session_state.reference_img is not None:
        total_steps += len(st.session_state.uploaded_images) - 1
    
    current_step = 0
    
    try:
        # Analyze all images
        for idx, img_data in enumerate(st.session_state.uploaded_images):
            status_text.text(f"Processing {img_data['name']}...")
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img_data['image'].save(tmp.name, format="JPEG", quality=90)
                img_data['temp_path'] = tmp.name
            
            analysis = analyze_image(img_data['temp_path'], analysis_options)
            if analysis:
                st.session_state.uploaded_images[idx]['analysis'] = analysis
            
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        # Compare with reference if set
        if st.session_state.reference_img is not None:
            ref_path = st.session_state.uploaded_images[st.session_state.reference_img]['temp_path']
            
            for idx, img_data in enumerate(st.session_state.uploaded_images):
                if idx == st.session_state.reference_img:
                    continue
                
                status_text.text(f"Comparing with {img_data['name']}...")
                comparison = compare_faces(ref_path, img_data['temp_path'])
                if comparison:
                    st.session_state.uploaded_images[idx]['comparison'] = comparison
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
        
        st.success("Analysis completed successfully!")
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
    finally:
        cleanup()
        progress_bar.empty()
        status_text.empty()

# Display results in an organized way
if st.session_state.uploaded_images and any(img.get('analysis') for img in st.session_state.uploaded_images):
    st.header("Analysis Results")
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Individual Results", "Comparison Summary"])
    
    with tab1:
        # Individual results for each image
        for idx, img_data in enumerate(st.session_state.uploaded_images):
            if not img_data['analysis']:
                continue
            
            with st.expander(f"Results for {img_data['name']}"):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.image(img_data['image'], use_column_width=True)
                    
                    # Reverse image search links
                    st.markdown("**Reverse Image Search:**")
                    google_url = f"https://www.google.com/searchbyimage?image_url={img_data['name']}"
                    bing_url = f"https://www.bing.com/images/search?q=imgurl:{img_data['name']}"
                    st.markdown(f"[Google Search]({google_url}) | [Bing Search]({bing_url})")
                
                with col2:
                    # Display analysis results in a clean format
                    analysis = img_data['analysis']
                    
                    # Create metrics columns
                    cols = st.columns(2)
                    
                    # Age
                    if 'age' in analysis:
                        with cols[0]:
                            st.metric(label="Age", value=f"{analysis['age']} years")
                    
                    # Gender
                    if 'gender' in analysis:
                        with cols[1]:
                            gender = max(analysis['gender'].items(), key=lambda x: x[1])
                            st.metric(label="Gender", value=f"{gender[0]} ({gender[1]:.1f}%)")
                    
                    # Emotion
                    if 'emotion' in analysis:
                        st.subheader("Emotion Analysis")
                        emotion = max(analysis['emotion'].items(), key=lambda x: x[1])
                        st.metric(label="Dominant Emotion", value=f"{emotion[0]} ({emotion[1]:.1f}%)")
                        
                        # Emotion chart with better styling
                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors = plt.cm.viridis(np.linspace(0, 1, len(analysis['emotion'])))
                        ax.bar(analysis['emotion'].keys(), analysis['emotion'].values(), color=colors)
                        ax.set_title("Emotion Distribution", pad=20)
                        ax.tick_params(axis='x', rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    # Race
                    if 'race' in analysis:
                        st.subheader("Race Analysis")
                        race = max(analysis['race'].items(), key=lambda x: x[1])
                        st.metric(label="Predominant Race", value=f"{race[0]} ({race[1]:.1f}%)")
                    
                    # Comparison results if available
                    if idx != st.session_state.reference_img and img_data.get('comparison'):
                        st.subheader("Comparison with Reference")
                        comparison = img_data['comparison']
                        similarity = 1 - comparison['distance']
                        st.metric(
                            label="Similarity Score",
                            value=f"{similarity:.1%}",
                            help=f"Threshold: {comparison['threshold']:.2f}"
                        )
                        if similarity > threshold:
                            st.success("✅ Faces match (above threshold)")
                        else:
                            st.warning("❌ Faces don't match (below threshold)")
    
    with tab2:
        # Summary comparison table
        if st.session_state.reference_img is not None:
            st.subheader("Comparison Summary")
            
            # Create summary data
            summary_data = []
            ref_name = st.session_state.uploaded_images[st.session_state.reference_img]['name']
            
            for idx, img_data in enumerate(st.session_state.uploaded_images):
                if idx == st.session_state.reference_img:
                    continue
                
                if img_data.get('comparison'):
                    similarity = 1 - img_data['comparison']['distance']
                    match = "Yes" if similarity > threshold else "No"
                    
                    summary_data.append({
                        "Image": img_data['name'],
                        "Similarity": f"{similarity:.1%}",
                        "Match": match,
                        "Age": img_data['analysis'].get('age', 'N/A'),
                        "Gender": max(img_data['analysis'].get('gender', {}).items(), key=lambda x: x[1])[0] if 'gender' in img_data['analysis'] else 'N/A'
                    })
            
            if summary_data:
                st.table(summary_data)
            else:
                st.warning("No comparison data available")
        else:
            st.info("Set a reference image to see comparison summary")

# Generate PDF report
if st.session_state.uploaded_images and any(img.get('analysis') for img in st.session_state.uploaded_images):
    st.header("Report Generation")
    if st.button("Generate PDF Report"):
        with st.spinner("Generating PDF..."):
            try:
                pdf_buffer = create_pdf_report(st.session_state.uploaded_images, st.session_state.reference_img)
                
                # Create download link
                b64 = base64.b64encode(pdf_buffer.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="face_analysis_report.pdf">Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("PDF report generated successfully!")
            except Exception as e:
                st.error(f"Failed to generate PDF: {str(e)}")
