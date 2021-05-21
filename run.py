import streamlit as st
from web.view import image_input, face_detect

st.set_page_config(
    page_title="Zero-DCE",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement")
st.sidebar.title('Navigation')
method = st.sidebar.radio('Go To ->', options=['Low Light Image Enhancement', 'Low Light Face Detection'])
st.sidebar.header('Options')

if method == 'Low Light Image Enhancement':
    image_input()
else:
    face_detect()