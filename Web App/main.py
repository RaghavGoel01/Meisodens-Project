import streamlit as st
import torch
from PIL import Image
import os
import numpy as np
from ultralytics import YOLO

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = YOLO('../best.pt')
    #model = torch.hub.load('ultralytics/yolov11', 'custom', path='../best.pt', force_reload=True) 
    return model

model = load_model()

# Sidebar
st.sidebar.title("Dental AI Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Mesiodens Detection"])

# Home Page
if app_mode == "Home":
    st.title("Mesiodens Detection System 🦷")
    st.image("dental_photo.jpeg", use_container_width=True)
    st.markdown("""
        Welcome to the **Mesiodens Detection System** – a deep learning-based tool to assist dentists in identifying **mesiodens** in periapical radiographs.

        ### 🛠 How It Works:
        - Upload a periapical radiograph.
        - Our YOLOv5 model detects the presence of mesiodens.
        - Results are visualized and summarized below the image.

        Go to the **Mesiodens Detection** tab to begin!
    """)

# About Page
elif app_mode == "About":
    st.title("About the Project")
    st.markdown("""
        - **Mesiodens** are supernumerary teeth located in the maxillary central incisor region.
        - Early detection is crucial to prevent complications in eruption and alignment.
        - This tool uses a custom-trained YOLOv5 model on annotated periapical radiographs.
        - Model size: ~224x224 preprocessed input with bounding box labels.
    """)

# Detection Page
elif app_mode == "Mesiodens Detection":
    st.title("Mesiodens Detection from Dental Radiographs")

    uploaded_file = st.file_uploader("Upload a Periapical Radiograph Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Radiograph", use_container_width=True)

        if st.button("Run Detection"):
            with st.spinner("Detecting Mesiodens..."):
                # Run inference using YOLOv8
                results = model(image)

                # YOLOv8 returns a list of results; get the first one
                result = results[0]
                
                 # Get prediction dataframe
                df = result.to_df()

                # Plot and display annotated image
                annotated_image = result.plot()
                st.image(annotated_image, caption="Detection Result", use_container_width=True)

                # Display detection info
                if df.empty:
                    st.warning("No mesiodens detected.")
                else:
                    st.success("Mesiodens Detected ✅")
                    st.dataframe(df)

                # # Get the annotated image
                # annotated_image = result.plot()

                # st.image(annotated_image, caption="Detection Result", use_container_width=True)

                # # Show detection details as dataframe
                # #df = result.pandas().xyxy[0]
                # if len(df) == 0:
                #     st.warning("No mesiodens detected.")
                # else:
                #     st.success("Mesiodens Detected ✅")
                #     st.dataframe(df)

#  elif app_mode == "Mesiodens Detection":
#     st.title("Mesiodens Detection from Dental Radiographs")

#     uploaded_file = st.file_uploader("Upload a Periapical Radiograph Image", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Radiograph", use_container_width=True)

#         if st.button("Run Detection"):
#             with st.spinner("Detecting Mesiodens..."):
#                 # Run inference
#                 results = model(image)
#                 results.render()  # render boxes on image
#                 st.image(results.ims[0], caption="Detection Result", use_container_width=True)

#                 # Show detection details
#                 df = results.pandas().xyxy[0]
#                 if len(df) == 0:
#                     st.warning("No mesiodens detected.")
#                 else:
#                     st.success("Mesiodens Detected ✅")
#                     st.dataframe(df)
