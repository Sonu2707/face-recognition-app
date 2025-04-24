import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Face Recognition & Analysis App", layout="centered")
st.title("Face Recognition & Analysis App")
st.markdown("Upload images or use your webcam to analyze and compare faces.")

# Function to save uploaded or captured image
def save_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

# Function to display reverse image search links
def reverse_image_search_links(image_path):
    st.subheader("Reverse Image Search")
    st.markdown("You can use the following links to perform a manual reverse image search:")
    st.markdown("[Google Images](https://images.google.com)")
    st.markdown("[Bing Visual Search](https://www.bing.com/visualsearch)")

# Image Upload or Capture
st.sidebar.header("Image Inputs")
input_method = st.sidebar.radio("Select Image Input Method:", ("Upload Images", "Use Webcam"))

if input_method == "Upload Images":
    uploaded_image1 = st.sidebar.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
    uploaded_image2 = st.sidebar.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])
    if uploaded_image1 and uploaded_image2:
        image1_path = save_image(uploaded_image1)
        image2_path = save_image(uploaded_image2)
        st.image([uploaded_image1, uploaded_image2], caption=["Image 1", "Image 2"], width=250)
elif input_method == "Use Webcam":
    captured_image1 = st.sidebar.camera_input("Capture First Image")
    captured_image2 = st.sidebar.camera_input("Capture Second Image")
    if captured_image1 and captured_image2:
        image1_path = save_image(captured_image1)
        image2_path = save_image(captured_image2)
        st.image([captured_image1, captured_image2], caption=["Image 1", "Image 2"], width=250)

# Analyze and Compare Faces
if st.button("Analyze & Compare"):
    if 'image1_path' in locals() and 'image2_path' in locals():
        with st.spinner("Analyzing..."):
            try:
                analysis = DeepFace.analyze(img_path=image1_path, actions=['age', 'gender', 'emotion', 'race'])
                verification = DeepFace.verify(img1_path=image1_path, img2_path=image2_path)

                st.subheader("Image 1 Analysis")
                st.json(analysis[0])

                st.subheader("Face Comparison Result")
                st.write("Are they the same person?:", verification['verified'])
                st.write("Similarity Score (lower means more similar):", verification['distance'])

                reverse_image_search_links(image1_path)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide both images for analysis.")
