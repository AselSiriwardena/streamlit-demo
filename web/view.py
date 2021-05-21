import numpy as np
import streamlit as st
from PIL import Image
import imutils

from utils.face_detection import detect
from web.generate_image import lowlight
from utils.transforms_util import trans_tensor_to_pil, trans_tensor_to_b64
import torchvision.transforms.functional as TF


def image_input():
    content_file = st.sidebar.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg", "bmp"])
    if content_file is not None:
        content = Image.open(content_file)
        width, height = content.size
        content = np.array(content)  # pil to cv
    else:
        st.info("Upload an Image to Enhance")
        st.stop()

    if width > 600:
        WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(500, width + 1, 1)), value=600)
    else:
        WIDTH = st.sidebar.select_slider('QUALITY (Maximum resolution applied)', list(range(0, width + 1, 1)),
                                         value=width)

    content = imutils.resize(content, width=WIDTH)
    try:
        generated = lowlight(content)
        generated = trans_tensor_to_pil(generated)

        image_container = st.beta_container()
        container_col1, container_col2 = image_container.beta_columns([1, 1])
        btn_container = st.beta_container()

        with container_col1:
            st.header("Original Image")

        with container_col2:
            st.header("Enhanced Image")

        container_col1.image(content, clamp=True, output_format='PNG')

        b64_image_gen = trans_tensor_to_b64(generated)
        href = f'<a download="enhanced.JPEG" href="{b64_image_gen}">Download Image</a>'
        btn_container.markdown(href, unsafe_allow_html=True)
        container_col2.image(generated, clamp=True, output_format='PNG')

    except RuntimeError as e:
        if str(e).startswith('CUDA out of memory.'):
            st.warning("Image is loo large to handle, Reduce size")
        else:
            st.warning(e)


def face_detect():
    content_file_face = st.sidebar.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg", "bmp"])
    if content_file_face is not None:
        face_low = Image.open(content_file_face)
        width, height = face_low.size
        face_low = np.array(face_low)  # pil to cv
    else:
        st.info("Upload an Image for Face Detection")
        st.success('Many, many thanks to Davis King (https://github.com/davisking) for creating dlib and for providing '
                   'the trained facial feature detection and face encoding models used in this library.')
        st.stop()

    if width > 600:
        WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(500, width + 1, 50)), value=600)
        content_face_low1 = imutils.resize(face_low, width=WIDTH)
        content_face_low2 = imutils.resize(face_low, width=WIDTH)
    else:
        WIDTH = st.sidebar.select_slider('QUALITY (Maximum resolution applied)', list(range(0, width + 1, 50)),
                                         value=width)
        content_face_low1 = imutils.resize(face_low, width=WIDTH)
        content_face_low2 = imutils.resize(face_low, width=WIDTH)

    try:
        org = detect(content_face_low1)

        generated = lowlight(content_face_low2)
        generated = trans_tensor_to_pil(generated)
        generated = np.array(generated)

        face_image_enhanced = detect(generated)

        image_container = st.beta_container()
        container_col1, container_col2 = image_container.beta_columns([1, 1])

        with container_col1:
            st.header("Original Image")

        with container_col2:
            st.header("Enhanced Image")

        container_col1.image(org, clamp=True, output_format='PNG')

        container_col2.image(face_image_enhanced, clamp=True, output_format='PNG')

    except RuntimeError as e:
        if str(e).startswith('CUDA out of memory.'):
            st.warning("Image is loo large to handle, reduce size")
        else:
            st.warning(e)
