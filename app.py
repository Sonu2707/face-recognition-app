import streamlit as st
from PIL import Image
import tempfile
import os
from fpdf import FPDF

st.set_page_config(page_title="AI Face Recognition & Insight App", layout="wide")
st.title("AI Face Recognition & Insight App")
st.markdown("Upload multiple face images to analyze attributes, compare identities, and download a full report.")

# Safe import for DeepFace (runtime only)
def get_deepface():
    from deepface import DeepFace
    return DeepFace

# Save uploaded image
def save_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

# PDF report generation
def generate_pdf_report(analysis_data, comparisons):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI Face Recognition Report", ln=1, align="C")

    for idx, entry in enumerate(analysis_data):
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=f"Image {idx+1} Analysis", ln=1)
        for key in ['age', 'dominant_gender', 'dominant_emotion', 'dominant_race']:
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"{key.capitalize().replace('_', ' ')}: {entry.get(key, 'N/A')}", ln=1)

    if comparisons:
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Comparison Results", ln=1)
        for comp in comparisons:
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Image {comp['index']} Match: {comp['match']}, Score: {comp['score']:.4f}", ln=1)

    report_path = os.path.join(tempfile.gettempdir(), "face_report.pdf")
    pdf.output(report_path)
    return report_path

# Sidebar upload and reference selection
st.sidebar.header("Upload Multiple Images")
uploaded_images = st.sidebar.file_uploader("Upload Images (2 to 10)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_images and len(uploaded_images) >= 2:
    st.sidebar.markdown("### Select Reference Image")
    reference_index = st.sidebar.selectbox("Choose Reference Image to Compare Others Against", range(len(uploaded_images)))

    image_paths = []
    analysis_data = []
    st.subheader("Uploaded Image Gallery")
    gallery_cols = st.columns(min(5, len(uploaded_images)))

    for i, img in enumerate(uploaded_images):
        path = save_image(img)
        image_paths.append(path)
        with gallery_cols[i % len(gallery_cols)]:
            st.image(img, caption=f"Image {i+1}", width=150)

    if st.button("Analyze & Compare All"):
        with st.spinner("Analyzing images and comparing faces..."):
            try:
                DeepFace = get_deepface()
                for path in image_paths:
                    analysis = DeepFace.analyze(img_path=path, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
                    analysis_data.append(analysis[0])

                comparisons = []
                for i, path in enumerate(image_paths):
                    if i == reference_index:
                        continue
                    result = DeepFace.verify(img1_path=image_paths[reference_index], img2_path=path, enforce_detection=False)
                    comparisons.append({
                        "index": i + 1,
                        "match": result['verified'],
                        "score": result['distance']
                    })

                st.success("Analysis Complete")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"### Reference Image: Image {reference_index + 1}")
                    ref = analysis_data[reference_index]
                    st.write("**Age:**", ref['age'])
                    st.write("**Gender:**", ref['dominant_gender'])
                    st.write("**Emotion:**", ref['dominant_emotion'])
                    st.write("**Ethnicity:**", ref['dominant_race'])

                with col2:
                    st.markdown("### Comparison Results")
                    for comp in comparisons:
                        st.write(f"Image {comp['index']} Match: {comp['match']}, Score: {comp['score']:.4f}")

                # Generate and offer PDF download
                report_path = generate_pdf_report(analysis_data, comparisons)
                with open(report_path, "rb") as file:
                    st.download_button(
                        label="Download Report as PDF",
                        data=file,
                        file_name="face_report.pdf",
                        mime="application/pdf"
                    )

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
else:
    st.warning("Please upload at least 2 images to begin analysis.")
