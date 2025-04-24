import streamlit as st
import numpy as np
import cv2
from PIL import Image, ExifTags
import io
import base64
import requests
import plotly.graph_objects as go
from fpdf import FPDF
import traceback
import os
import json
from google.cloud import vision  # Google Cloud Vision API

# Critical imports with error handling
try:
    from deepface import DeepFace
except ImportError as e:
    st.error("Failed to import DeepFace. Please check your installation.")
    st.error(f"Error message: {str(e)}")
    st.stop()

# Initialize Google Cloud Vision client
try:
    vision_client = vision.ImageAnnotatorClient()
except Exception as e:
    st.warning("Google Cloud Vision not initialized. Location detection will be limited.")
    st.session_state.debug_info.append(f"Google Vision error: {str(e)}")
    vision_client = None

# Initialize session state
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'imgbb_urls' not in st.session_state:
    st.session_state.imgbb_urls = {}
if 'location_info' not in st.session_state:
    st.session_state.location_info = {}

# Custom CSS for styling
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
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Image Analysis Web Application (Test Mode)")
st.markdown("Upload images to analyze facial features, detect locations, and explore online presence using Google Cloud Vision and basic reverse image search. Ideal for testing purposes.")

# Sidebar for image history
with st.sidebar:
    st.header("Image History")
    if st.session_state.uploaded_files:
        for file in st.session_state.uploaded_files:
            st.image(file.getvalue(), width=100, caption=file.name)
        if st.button("Clear Gallery"):
            st.session_state.uploaded_files = []
            st.session_state.analysis_results = {}
            st.session_state.imgbb_urls = {}
            st.session_state.location_info = {}
            st.experimental_rerun()

# Test Plotly chart to verify rendering
st.subheader("Test Chart")
test_fig = go.Figure(data=[go.Pie(labels=['A', 'B'], values=[30, 70])])
test_fig.update_layout(title="Test Pie Chart", margin=dict(t=40, b=0, l=0, r=0))
st.plotly_chart(test_fig, use_container_width=True)

# File uploader
new_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

# Function to extract EXIF data
def extract_exif_data(image):
    try:
        img = Image.open(image)
        exif_data = img._getexif()
        if exif_data:
            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
            gps_info = exif.get('GPSInfo')
            if gps_info:
                lat = gps_info.get(2)
                lon = gps_info.get(4)
                if lat and lon:
                    lat = float(lat[0]) + float(lat[1])/60 + float(lat[2])/3600
                    lon = float(lon[0]) + float(lon[1])/60 + float(lon[2])/3600
                    if gps_info.get(1) == 'S':
                        lat = -lat
                    if gps_info.get(3) == 'W':
                        lon = -lon
                    return {"latitude": lat, "longitude": lon, "other": exif}
        return {"latitude": None, "longitude": None, "other": exif_data}
    except Exception as e:
        st.session_state.debug_info.append(f"EXIF extraction error: {str(e)}")
        return {"latitude": None, "longitude": None, "other": None}

# Function for Google Cloud Vision analysis
def google_vision_analysis(img_bytes):
    if not vision_client:
        return {"landmarks": [], "objects": [], "labels": []}
    try:
        image = vision.Image(content=img_bytes)
        # Landmark detection
        response = vision_client.landmark_detection(image=image)
        landmarks = [landmark.description for landmark in response.landmark_annotations]
        # Object detection
        response = vision_client.object_localization(image=image)
        objects = [obj.name for obj in response.localized_object_annotations]
        # Label detection
        response = vision_client.label_detection(image=image)
        labels = [label.description for label in response.label_annotations]
        return {"landmarks": landmarks, "objects": objects, "labels": labels}
    except Exception as e:
        st.session_state.debug_info.append(f"Google Vision analysis error: {str(e)}")
        return {"landmarks": [], "objects": [], "labels": []}

# Process new uploads
if new_files:
    existing_names = [f.name for f in st.session_state.uploaded_files]
    for file in new_files:
        if file.type not in ['image/jpeg', 'image/png']:
            st.error(f"Invalid file type for {file.name}. Please upload a jpg or png image.")
            st.session_state.debug_info.append(f"Invalid file type: {file.type} for {file.name}")
            continue
        if file.name not in existing_names:
            st.session_state.uploaded_files.append(file)
            img_bytes = file.getvalue()
            
            # DeepFace analysis
            with st.spinner(f"Analyzing facial features for {file.name}..."):
                try:
                    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    result = DeepFace.analyze(img, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
                    st.session_state.analysis_results[file.name] = result
                    st.session_state.debug_info.append(f"DeepFace result for {file.name}: {result}")
                except Exception as e:
                    st.error(f"Error analyzing {file.name}: {str(e)}")
                    st.session_state.debug_info.append(str(e))

            # EXIF data extraction
            with st.spinner(f"Extracting metadata for {file.name}..."):
                exif_data = extract_exif_data(file)
                st.session_state.location_info[file.name] = exif_data

            # Google Cloud Vision analysis
            with st.spinner(f"Detecting location and objects for {file.name}..."):
                vision_results = google_vision_analysis(img_bytes)
                st.session_state.location_info[file.name].update(vision_results)

            # ImgBB upload for reverse search
            try:
                api_key = st.secrets.get("IMGBB_API_KEY", None)
                if not api_key:
                    st.error("ImgBB API key not set. Reverse image search will be limited.")
                    st.session_state.debug_info.append("ImgBB API key not found in secrets.")
                else:
                    url = "https://api.imgbb.com/1/upload"
                    payload = {"key": api_key, "image": base64.b64encode(img_bytes).decode()}
                    response = requests.post(url, payload)
                    if response.status_code == 200:
                        st.session_state.imgbb_urls[file.name] = response.json()['data']['url']
                    else:
                        st.session_state.debug_info.append(f"ImgBB upload failed: {response.text}")
            except Exception as e:
                st.session_state.debug_info.append(f"ImgBB error: {str(e)}")

# Display results
if st.session_state.uploaded_files:
    tab_names = [file.name for file in st.session_state.uploaded_files]
    tabs = st.tabs(tab_names)
    for i, tab in enumerate(tabs):
        with tab:
            file = st.session_state.uploaded_files[i]
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(file.getvalue(), caption=file.name)
                if file.name in st.session_state.imgbb_urls:
                    img_url = st.session_state.imgbb_urls[file.name]
                    st.subheader("Reverse Image Search")
                    st.markdown(f"[Google](https://www.google.com/searchbyimage?image_url={img_url})")
                    st.markdown(f"[Bing](https://www.bing.com/images/search?view=detailv2&iss=sbi&FORM=IRSBIQ&sbisrc=UrlPaste&q=imgurl:{img_url})")
                    st.markdown(f"[Yandex](https://yandex.com/images/search?rpt=imageview&url={img_url})")
                    st.markdown("**Tip**: Click these links to manually search for social media profiles or websites where this image appears.")
            with col2:
                # Facial analysis
                if file.name in st.session_state.analysis_results:
                    results = st.session_state.analysis_results[file.name]
                    if isinstance(results, list) and len(results) > 0:
                        for j, result in enumerate(results):
                            st.subheader(f"Face {j+1}")
                            st.write(f"Age: {result['age']}")
                            fig_gender = go.Figure(data=[go.Pie(labels=list(result['gender'].keys()), values=list(result['gender'].values()), hole=0.3)])
                            fig_gender.update_layout(title="Gender Probability")
                            st.plotly_chart(fig_gender, use_container_width=True)
                            fig_emotion = go.Figure(data=[go.Bar(x=list(result['emotion'].keys()), y=list(result['emotion'].values()))])
                            fig_emotion.update_layout(title="Emotion Probability")
                            st.plotly_chart(fig_emotion, use_container_width=True)
                            fig_race = go.Figure(data=[go.Pie(labels=list(result['race'].keys()), values=list(result['race'].values()), hole=0.3)])
                            fig_race.update_layout(title="Race Probability")
                            st.plotly_chart(fig_race, use_container_width=True)
                    else:
                        st.warning(f"No faces detected in {file.name}.")

                # Location and metadata
                st.subheader("Location and Metadata")
                location_info = st.session_state.location_info.get(file.name, {})
                if location_info.get("latitude") and location_info.get("longitude"):
                    st.write(f"GPS Coordinates: ({location_info['latitude']}, {location_info['longitude']})")
                    st.markdown(f"[View on Google Maps](https://www.google.com/maps?q={location_info['latitude']},{location_info['longitude']})")
                else:
                    st.write("No GPS data found in EXIF. (Note: Social media images often lack EXIF data.)")
                if location_info.get("landmarks"):
                    st.write(f"Landmarks Detected: {', '.join(location_info['landmarks'])}")
                if location_info.get("objects"):
                    st.write(f"Objects Detected: {', '.join(location_info['objects'])}")
                if location_info.get("labels"):
                    st.write(f"Image Labels: {', '.join(location_info['labels'])}")
                if location_info.get("other"):
                    st.write(f"Other Metadata: {json.dumps(location_info['other'], indent=2)}")

# PDF report generation
def generate_pdf():
    pdf = FPDF()
    for file in st.session_state.uploaded_files:
        pdf.add_page()
        img = Image.open(file)
        img_path = f"temp_{file.name}"
        img.save(img_path)
        pdf.image(img_path, x=10, y=10, w=90)
        os.remove(img_path)
        pdf.set_xy(10, 110)
        pdf.set_font("Arial", size=12)
        
        # Facial analysis
        if file.name in st.session_state.analysis_results:
            results = st.session_state.analysis_results[file.name]
            if isinstance(results, list) and len(results) > 0:
                for j, result in enumerate(results):
                    pdf.cell(0, 10, f"Face {j+1}: Age {result['age']}, Gender {result['dominant_gender']}, Emotion {result['dominant_emotion']}, Race {result['dominant_race']}", ln=True)
            else:
                pdf.cell(0, 10, f"No faces detected in {file.name}", ln=True)
        
        # Location and metadata
        location_info = st.session_state.location_info.get(file.name, {})
        if location_info.get("latitude") and location_info.get("longitude"):
            pdf.cell(0, 10, f"GPS: ({location_info['latitude']}, {location_info['longitude']})", ln=True)
        if location_info.get("landmarks"):
            pdf.cell(0, 10, f"Landmarks: {', '.join(location_info['landmarks'])}", ln=True)
        if location_info.get("objects"):
            pdf.cell(0, 10, f"Objects: {', '.join(location_info['objects'])}", ln=True)
        if location_info.get("labels"):
            pdf.cell(0, 10, f"Labels: {', '.join(location_info['labels'])}", ln=True)
    
    return pdf.output(dest='S').encode('latin1')

# Download report button
if st.session_state.uploaded_files and st.button("Download Full Report"):
    with st.spinner("Generating PDF report..."):
        pdf_data = generate_pdf()
        st.download_button(label="Download PDF", data=pdf_data, file_name="image_analysis_report.pdf", mime="application/pdf")

# Debug section
with st.expander("Debug Information"):
    if st.session_state.debug_info:
        for info in st.session_state.debug_info:
            st.text(info)
    else:
        st.write("No errors logged.")
