import streamlit as st
import cv2
import numpy as np
import face_recognition
import sqlite3
import os
import logging
import pandas as pd
from datetime import datetime
from utils import initialize_database, load_known_faces, add_known_face, recognize_face

# Configure logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set page config
st.set_page_config(page_title="Enhanced Face Recognition App", layout="wide")

# Initialize session state
if "known_faces_loaded" not in st.session_state:
    st.session_state.known_faces_loaded = False
    st.session_state.known_face_encodings = []
    st.session_state.known_face_names = []

# Title
st.title("Enhanced Face Recognition App")

# Initialize database
initialize_database()

# Load known faces from database
if not st.session_state.known_faces_loaded:
    st.session_state.known_face_encodings, st.session_state.known_face_names = load_known_faces()
    st.session_state.known_faces_loaded = True

# Sidebar for adding known faces
st.sidebar.header("Add Known Face")
with st.sidebar.form("upload_known_face"):
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    person_name = st.text_input("Enter person's name")
    submit_button = st.form_submit_button("Add Face")
    if submit_button and uploaded_file and person_name:
        try:
            file_path = os.path.join("known_faces", f"{person_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            add_known_face(file_path, person_name)
            st.session_state.known_faces_loaded = False  # Trigger reload
            st.success(f"Added {person_name} to known faces!")
            logging.info(f"Added known face: {person_name}")
        except Exception as e:
            st.error(f"Error adding face: {str(e)}")
            logging.error(f"Error adding face: {str(e)}")

# Main app: Choose input method
option = st.selectbox("Choose input method", ["Upload Image", "Upload Video", "Use Webcam"])

# Recognition history
st.subheader("Recognition History")
history_conn = sqlite3.connect("database.db")
history_df = pd.read_sql_query("SELECT timestamp, name, confidence FROM recognition_history", history_conn)
history_conn.close()
st.dataframe(history_df, use_container_width=True)
if st.button("Download History as CSV"):
    history_df.to_csv("recognition_history.csv", index=False)
    st.success("History downloaded as recognition_history.csv")

if option == "Upload Image":
    st.subheader("Upload an Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        with st.spinner("Processing image..."):
            try:
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                result_image, names, confidences = recognize_face(
                    image, st.session_state.known_face_encodings, st.session_state.known_face_names
                )
                st.image(result_image, channels="BGR", caption="Processed Image")
                st.write("Recognized faces:", ", ".join([f"{n} ({c:.2f})" for n, c in zip(names, confidences)]))
                logging.info(f"Image processed. Recognized: {names}")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                logging.error(f"Error processing image: {str(e)}")

elif option == "Upload Video":
    st.subheader("Upload a Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        with st.spinner("Processing video..."):
            try:
                # Save video temporarily
                temp_video = "temp_video.mp4"
                with open(temp_video, "wb") as f:
                    f.write(uploaded_video.read())
                cap = cv2.VideoCapture(temp_video)
                FRAME_WINDOW = st.image([])
                recognized_names = []
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % 5 == 0:  # Process every 5th frame
                        result_image, names, confidences = recognize_face(
                            frame, st.session_state.known_face_encodings, st.session_state.known_face_names
                        )
                        FRAME_WINDOW.image(result_image, channels="BGR")
                        recognized_names.extend(names)
                    frame_count += 1
                cap.release()
                os.remove(temp_video)
                st.write("Recognized faces:", ", ".join(set(recognized_names)))
                logging.info(f"Video processed. Recognized: {recognized_names}")
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                logging.error(f"Error processing video: {str(e)}")

elif option == "Use Webcam":
    st.subheader("Real-Time Webcam Face Recognition")
    run = st.checkbox("Run Webcam")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            logging.error("Failed to capture webcam video")
            break
        if frame_count % 3 == 0:  # Process every 3rd frame for performance
            result_image, names, confidences = recognize_face(
                frame, st.session_state.known_face_encodings, st.session_state.known_face_names
            )
            FRAME_WINDOW.image(result_image, channels="BGR")
            st.write("Recognized faces:", ", ".join([f"{n} ({c:.2f})" for n, c in zip(names, confidences)]))
        frame_count += 1
    cap.release()

# Instructions
st.markdown("""
### Instructions
1. **Add Known Faces**: Upload an image and name in the sidebar to add to the face database.
2. **Upload Image/Video**: Upload an image or video to detect and recognize faces.
3. **Webcam**: Enable the webcam for real-time face recognition.
4. **History**: View and download the recognition history as a CSV file.
""")
