import matplotlib.pyplot as plt
import cv2
import numpy as np
import streamlit as st

#define the functions of our  filters
def blur(img, kernel_size):
  return cv2.blur(img, (kernel_size, kernel_size))

def gaussian_blur(img, kernel_size):
  return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def median_blur(img, kernel_size):
  return cv2.medianBlur(img, kernel_size)

def bilateral_filter(img, kernel_size, sigma_color, sigma_space):
  return cv2.bilateralFilter(img, kernel_size, sigma_color, sigma_space)

# title of website
st.title('image processing with Streamlit')

# uploading the image
uploaded_file = st.file_uploader('select your image')

#select which filter to applyا
filter_type = st.selectbox('choose filter type:', ['Blur', 'Gaussian Blur', 'Median Blur', 'Bilateral Filter'])

# select kernel size
kernel_size = st.slider('choose kernel size:', 1, 15, 2)

if uploaded_file is not None:
  img = np.array(bytearray(uploaded_file.read()))
  img = cv2.imdecode(img, cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  if filter_type == 'Blur':
    filtered_img = blur(img, kernel_size)
  elif filter_type == 'Gaussian Blur':
    filtered_img = gaussian_blur(img, kernel_size)
  elif filter_type == 'Median Blur':
    filtered_img = median_blur(img, kernel_size)
  else:
    filtered_img = bilateral_filter(img, kernel_size, 75, 75)

if uploaded_file is not None:
  st.image(img, caption='original')
  st.image(filtered_img, caption=' filterd image')

