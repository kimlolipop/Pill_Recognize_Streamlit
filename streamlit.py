import streamlit as st
import numpy as np
import cv2
from main import main_classify

# st.title('My first app')S
st.write("""
# Pills Recognition

Version 0.0.1
""")

import os

def fun(img):
    #call classify

    ans, local_im = main_classify(img)

    for i in range(0, len(ans)):

        st.write('NDC: ' + ans[i])
        st.image(local_im[i])

    # st.write('fun click')
# return


uploaded_file = st.file_uploader("Choose an image file", type = ["jpg", "jpeg", "png"] )

if uploaded_file is not None:
    

    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    if st.button('Classify'):
        fun(img)

    
    # print(opencv_image)
    # Now do something with the image! For example, let's display it:
    st.image(img, channels="RGB")