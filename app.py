# face_app.py
import streamlit as st
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Face Recognition & Analysis App", layout="centered")
st.title("Face Recognition & Analysis App")
st.markdown("Upload two face images to analyze facial attributes and compare them.")

# Save uploaded image
def save_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

# Display reverse image search links
def reverse_image_search_links():
    st.subheader("Reverse Image Search")
    st.markdown("[Google Images](https://images.google.com)")
    st.markdown("[Bing Visual Search](https://www.bing.com/visualsearch)")

# Image Upload Section
st.sidebar.header("Upload Images")
uploaded_image1 = st.sidebar.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
uploaded_image2 = st.sidebar.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

image1_path = image2_path = None

if uploaded_image1:
    st.image(uploaded_image1, caption="Image 1", width=250)
    image1_path = save_image(uploaded_image1)
if uploaded_image2:
    st.image(uploaded_image2, caption="Image 2", width=250)
    image2_path = save_image(uploaded_image2)

# Analyze and Compare Faces
if st.button("Analyze & Compare"):
    if image1_path and image2_path:
        with st.spinner("Analyzing..."):
            try:
                from deepface import DeepFace
                analysis = DeepFace.analyze(img_path=image1_path, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
                verification = DeepFace.verify(img1_path=image1_path, img2_path=image2_path, enforce_detection=False)

                st.subheader("Image 1 Analysis")
                st.json(analysis[0])

                st.subheader("Comparison Result")
                st.write("Same person?", verification['verified'])
                st.write("Similarity Score:", verification['distance'])

                reverse_image_search_links()

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload both images before running analysis.")
