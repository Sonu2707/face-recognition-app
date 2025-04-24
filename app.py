import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from utils import load_known_faces, recognize_face

# Set page config
st.set_page_config(page_title="Face Recognition App", layout="wide")

# Title
st.title("Face Recognition App")

# Create directory for known faces if it doesn't exist
if not os.path.exists("known_faces"):
    os.makedirs("known_faces")

# Load known faces
known_face_encodings, known_face_names = load_known_faces("known_faces")

# Sidebar for uploading known faces
st.sidebar.header("Add Known Face")
with st.sidebar.form("upload_known_face"):
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    person_name = st.text_input("Enter person's name")
    submit_button = st.form_submit_button("Add Face")
    if submit_button and uploaded_file and person_name:
        # Save the uploaded image
        file_path = os.path.join("known_faces", f"{person_name}.jpg")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Added {person_name} to known faces!")
        # Reload known faces
        known_face_encodings, known_face_names = load_known_faces("known_faces")

# Main app: Choose input method
option = st.selectbox("Choose input method", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    st.subheader("Upload an Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        # Recognize faces
        result_image, names = recognize_face(image, known_face_encodings, known_face_names)
        # Display result
        st.image(result_image, channels="BGR", caption="Processed Image")
        st.write("Recognized faces:", ", ".join(names) if names else "No faces recognized")

elif option == "Use Webcam":
    st.subheader("Webcam Face Recognition")
    run = st.checkbox("Run Webcam")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
        # Recognize faces
        result_image, names = recognize_face(frame, known_face_encodings, known_face_names)
        # Display result
        FRAME_WINDOW.image(result_image, channels="BGR")
        st.write("Recognized faces:", ", ".join(names) if names else "No faces recognized")
    cap.release()

# Instructions
st.markdown("""
### Instructions
1. **Add Known Faces**: Use the sidebar to upload images of known individuals and provide their names.
2. **Upload Image**: Upload an image to detect and recognize faces.
3. **Use Webcam**: Enable the webcam to perform real-time face recognition.
""")
