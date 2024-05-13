import streamlit as st
import numpy as np
from PIL import Image
from detection.object_detection import detect_object


def main():
    st.title("Deteksi mata sapi dengan Yolov4-Tiny")
    img_array = upload_image_ui()

    if isinstance(img_array, np.ndarray):
        image = detect_object(img_array)
        st.image(image)

def upload_image_ui():
    uploaded_image = st.file_uploader("silahkan masukkan gambar disini", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
        except Exception:
            st.error("Error: Invalid image")
        else:
            img_array = np.array(image)
            return img_array

if __name__ == '__main__':
    main()
