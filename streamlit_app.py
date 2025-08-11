import traceback
import streamlit as st
import os, random 
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_autorefresh import st_autorefresh
from PIL import Image
from pathlib import Path

# CONFIG 
IMG_SIZE = 150
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
DATA_DIR = Path(r"C:\Users\maayi\IFU\classification\Data\seg_train")
MODE = "grid"
GRID_COLS = 4
GRID_COUNT = 8
DEFAULT_N = 15
REFERESH_MS = 5000

# Load Model 
@st.cache_resource
def load_trained_model():
    model_path = r"C:\Users\maayi\IFU\classification\model\custom_model.h5"
    model = load_model(model_path)
    return model

model = load_trained_model()

#  Preprocessing Function 
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img, img_array

# Streamlit UI 
with st.container():
    st.title("ㅤㅤㅤㅤPhoto Classifier")
    st.write("ㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤUpload a photo to predict its category:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img, img_array = preprocess_image(uploaded_file)

    if st.button("Classify"):
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = predictions[0][predicted_index]

        st.success(f"**Prediction:** {predicted_label}")
        st.info(f"**Confidence:** {confidence:.2f}")
       
st.header("Data Samples")
st_autorefresh(interval=REFERESH_MS, key="auto_refresh_samples")

@st.cache_data(show_spinner= False)
def list_image_paths():
    exts = {".jpg"}
    return [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in exts]

all_paths = list_image_paths()


# UI controls
if "order" not in st.session_state:
    order = all_paths.copy()
    random.shuffle(order)
    st.session_state.order = order
    st.session_state.idx = 0
else:
    order = st.session_state.order

# advance index every re-run
st.session_state.idx = (st.session_state.idx + 1) % len(order)
idx = st.session_state.idx

# reshuffle order occasionally to keep it fresh
if idx % max(1, len(order) // 3) == 0:   
    random.shuffle(order)

# render
window = [order[(idx + k) % len(order)] for k in range(GRID_COUNT)]
cols = st.columns(GRID_COLS)

for i, p in enumerate(window):
    with cols[i % GRID_COLS]:
        try:
            # Prefer direct path; it’s faster and more robust than PIL for display
            st.image(str(p), caption=f"{p.parent.name} · {p.name}", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not show: {p} → {e}") 

st.header("Evaluation Metrics")

st.subheader("Model loss and accuracy chart")
st.image(Image.open(r"C:\Users\maayi\IFU\classification\model\Figure_1.png"), use_container_width= True)
st.subheader("Confusion matrix")
st.image(Image.open(r"C:\Users\maayi\IFU\classification\model\Figure_2.png"), use_container_width= True)
 