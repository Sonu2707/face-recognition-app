import streamlit as st
import os
from PIL import Image

# Attempt to import DeepFace
try:
    from deepface import DeepFace
except Exception as e:
    st.error(f"Failed to import DeepFace: {e}")
    st.stop()

# Title
st.set_page_config(page_title="Face Recognition App")
st.title("Face Recognition App")

# File uploader
uploaded_file1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

# Run comparison
if uploaded_file1 and uploaded_file2:
    img1_path = os.path.join("temp1.jpg")
    img2_path = os.path.join("temp2.jpg")

    with open(img1_path, "wb") as f:
        f.write(uploaded_file1.getbuffer())

    with open(img2_path, "wb") as f:
        f.write(uploaded_file2.getbuffer())

    # Show images
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open(img1_path), caption="First Image", use_column_width=True)
    with col2:
        st.image(Image.open(img2_path), caption="Second Image", use_column_width=True)

    # Face verification
    with st.spinner("Verifying faces..."):
        try:
            result = DeepFace.verify(img1_path, img2_path)
            verified = result.get("verified", False)
            distance = result.get("distance", "N/A")
            threshold = result.get("threshold", "N/A")

            if verified:
                st.success(f"Faces match! Distance: {distance:.4f} | Threshold: {threshold}")
            else:
                st.warning(f"Faces do not match. Distance: {distance:.4f} | Threshold: {threshold}")
        except Exception as e:
            st.error(f"Verification failed: {e}")
