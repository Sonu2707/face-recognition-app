import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import base64
import requests
import plotly.graph_objects as go
from fpdf import FPDF
import traceback
import os

# Critical imports with error handling
try:
    from deepface import DeepFace
except ImportError as e:
    st.error("Failed to import DeepFace. Please check your installation.")
    st.error(f"Error message: {str(e)}")
    st.stop()

# Initialize debug information list
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []

# Initialize session state for uploaded files, analysis results, and ImgBB URLs
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'imgbb_urls' not in st.session_state:
    st.session_state.imgbb_urls = {}

# Custom CSS for modern styling
st.markdown("""
<style>
    .stFileUploader, .stImage, .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stExpander {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .sidebar .sidebar-content {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    img {
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Face Recognition Web Application")
st.markdown("Upload images to analyze facial features including age, gender, emotion, and race. View results in interactive charts and download a detailed report.")

# Sidebar for navigation and image history
with st.sidebar:
    st.header("Image History")
    if st.session_state.uploaded_files:
        for file in st.session_state.uploaded_files:
            st.image(file.getvalue(), width=100, caption=file.name)
        if st.button("Clear Gallery"):
            st.session_state.uploaded_files = []
            st.session_state.analysis_results = {}
            st.session_state.imgbb_urls = {}
            st.experimental_rerun()
    else:
        st.write("No images uploaded yet.")

# File uploader
new_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

# Process new uploads
if new_files:
    existing_names = [f.name for f in st.session_state.uploaded_files]
    for file in new_files:
        if file.name not in existing_names:
            st.session_state.uploaded_files.append(file)
            # Analyze the image
            with st.spinner(f"Analyzing {file.name}..."):
                try:
                    img_bytes = file.getvalue()
                    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    result = DeepFace.analyze(img, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
                    st.session_state.analysis_results[file.name] = result
                except Exception as e:
                    st.error(f"Error analyzing {file.name}: {str(e)}")
                    st.session_state.debug_info.append(str(e))
                    st.session_state.debug_info.append(traceback.format_exc())
            # Upload to ImgBB for reverse search
            try:
                api_key = st.secrets.get("IMGBB_API_KEY", "your_default_api_key")  # Set in secrets.toml
                url = "https://api.imgbb.com/1/upload"
                payload = {
                    "key": api_key,
                    "image": base64.b64encode(img_bytes).decode()
                }
                response = requests.post(url, payload)
                if response.status_code == 200:
                    st.session_state.imgbb_urls[file.name] = response.json()['data']['url']
                else:
                    st.error(f"Failed to upload {file.name} to ImgBB")
            except Exception as e:
                st.error(f"Error uploading to ImgBB: {str(e)}")
                st.session_state.debug_info.append(str(e))
                st.session_state.debug_info.append(traceback.format_exc())

# Display gallery and analysis in tabs
if st.session_state.uploaded_files:
    tab_names = [file.name for file in st.session_state.uploaded_files]
    tabs = st.tabs(tab_names)
    for i, tab in enumerate(tabs):
        with tab:
            file = st.session_state.uploaded_files[i]
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(file.getvalue(), caption=file.name)
                # Reverse image search buttons
                if file.name in st.session_state.imgbb_urls:
                    img_url = st.session_state.imgbb_urls[file.name]
                    st.subheader("Reverse Image Search")
                    st.markdown(f"[Google](https://www.google.com/searchbyimage?image_url={img_url})", unsafe_allow_html=True)
                    st.markdown(f"[Bing](https://www.bing.com/images/search?view=detailv2&iss=sbi&FORM=IRSBIQ&sbisrc=UrlPaste&q=imgurl:{img_url})", unsafe_allow_html=True)
                    # Placeholder URLs for Yahoo and DuckDuckGo (adjust as needed)
                    st.markdown(f"[Yahoo](https://images.search.yahoo.com/search/images?p={img_url})", unsafe_allow_html=True)
                    st.markdown(f"[DuckDuckGo](https://duckduckgo.com/?q={img_url}&iax=images&ia=images)", unsafe_allow_html=True)
            with col2:
                if file.name in st.session_state.analysis_results:
                    results = st.session_state.analysis_results[file.name]
                    if isinstance(results, list):
                        for j, result in enumerate(results):
                            st.subheader(f"Face {j+1}")
                            # Age
                            st.write(f"Age: {result['age']}")
                            # Gender
                            gender_probs = result['gender']
                            fig_gender = go.Figure(data=[go.Pie(labels=list(gender_probs.keys()), values=list(gender_probs.values()), hole=0.3)])
                            fig_gender.update_layout(title="Gender Probability", margin=dict(t=40, b=0, l=0, r=0))
                            st.plotly_chart(fig_gender, use_container_width=True)
                            # Emotion
                            emotion_probs = result['emotion']
                            fig_emotion = go.Figure(data=[go.Bar(x=list(emotion_probs.keys()), y=list(emotion_probs.values()))])
                            fig_emotion.update_layout(title="Emotion Probability", margin=dict(t=40, b=0, l=0, r=0))
                            st.plotly_chart(fig_emotion, use_container_width=True)
                            # Race
                            race_probs = result['race']
                            fig_race = go.Figure(data=[go.Pie(labels=list(race_probs.keys()), values=list(race_probs.values()), hole=0.3)])
                            fig_race.update_layout(title="Race Probability", margin=dict(t=40, b=0, l=0, r=0))
                            st.plotly_chart(fig_race, use_container_width=True)
                else:
                    st.write("Analysis not available yet.")

# PDF report generation function
def generate_pdf():
    pdf = FPDF()
    for file in st.session_state.uploaded_files:
        pdf.add_page()
        # Add image
        img = Image.open(file)
        img_path = f"temp_{file.name}"
        img.save(img_path)
        pdf.image(img_path, x=10, y=10, w=90)
        os.remove(img_path)
        # Add analysis
        if file.name in st.session_state.analysis_results:
            results = st.session_state.analysis_results[file.name]
            pdf.set_xy(10, 110)
            pdf.set_font("Arial", size=12)
            for j, result in enumerate(results):
                pdf.cell(0, 10, f"Face {j+1}: Age {result['age']}, Gender {result['dominant_gender']}, Emotion {result['dominant_emotion']}, Race {result['dominant_race']}", ln=True)
                # Add charts
                for chart_type, probs in [('Gender', result['gender']), ('Race', result['race'])]:
                    fig = go.Figure(data=[go.Pie(labels=list(probs.keys()), values=list(probs.values()))])
                    img_bytes = fig.to_image(format="png", engine="kaleido")
                    img = Image.open(io.BytesIO(img_bytes))
                    img_path = f"temp_chart_{file.name}_{chart_type}_{j}.png"
                    img.save(img_path)
                    pdf.image(img_path, x=10, y=pdf.get_y() + 10, w=50)
                    os.remove(img_path)
                    pdf.set_y(pdf.get_y() + 60)
    return pdf.output(dest='S').encode('latin1')

# Download report button
if st.session_state.uploaded_files and st.button("Download Full Report"):
    with st.spinner("Generating PDF report..."):
        pdf_data = generate_pdf()
        st.download_button(label="Download PDF", data=pdf_data, file_name="face_analysis_report.pdf", mime="application/pdf")

# Debug section
with st.expander("Debug Information"):
    if st.session_state.debug_info:
        for info in st.session_state.debug_info:
            st.text(info)
    else:
        st.write("No errors logged.")
