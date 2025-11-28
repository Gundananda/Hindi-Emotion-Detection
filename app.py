# Streamlit Multi‑Page UI with Sidebar Navigation
# Pages: Home, Prediction, Dataset, Model Info, Results

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import streamlit as st
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import tensorflow as tf
from PIL import Image

# Disable problematic attention backends
try:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdpa"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
except:
    pass

# Load Keras Model
MODEL_PATH = "HindiEmotion.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load BERT

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = AutoModel.from_pretrained(
    "bert-base-multilingual-cased",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False
)
bert_model.eval()

emotion_map = {
    0: "😡 Anger",
    1: "😊 Joy",
    2: "😢 Sadness",
    3: "😲 Surprise",
    4: "😐 Neutral"
}

@torch.no_grad()
def get_bert_embedding(text: str):
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=50
    )
    outputs = bert_model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return emb

# ============= UI =============
st.set_page_config(page_title="Hindi Emotion Detector", page_icon="🎭", layout="wide")

# Sidebar Navigation
page = st.sidebar.selectbox(
    "📁 Navigation",
    ["Home", "Prediction", "Dataset", "Model Info", "Results"]
)

# --- HOME PAGE -------------------------------------------------
if page == "Home":
    st.title("🎭 Hindi Emotion Detection App")
    st.markdown(
        """
        Welcome to the **Hindi Emotion Detector App**.
        
        This app detects the **emotion** expressed in Hindi text using **mBERT + BiLSTM**.
        
        Use the sidebar to navigate!
        """
    )

# --- PREDICTION PAGE -------------------------------------------
elif page == "Prediction":
    st.title("🔍 Predict Emotion")
    st.write("Enter Hindi text to predict emotional sentiment.")
    text = st.text_area("✍️ Enter Hindi Sentence:")
    if st.button("Predict"):
        if text.strip():
            emb = get_bert_embedding(text).reshape(1,1,768)
            pred = np.argmax(model.predict(emb), axis=1)[0]
            st.success(f"### ✅ Prediction: {emotion_map[pred]}")
        else:
            st.warning("⚠️ Please enter valid text.")

# --- DATASET PAGE ---------------------------------------------
elif page == "Dataset":
    st.title("📂 Dataset View")

    try:
        df = pd.read_excel("Bhaav-Dataset.xlsx")
        st.dataframe(df)
    except Exception as e:
        st.warning("Dataset file not found.")
        st.error(str(e))


# --- MODEL INFO PAGE ------------------------------------------
elif page == "Model Info":
    st.title("🧠 Model Information")
    st.write(
        """
        ### Architecture
        - **Embedding:** mBERT (bert‑base‑multilingual‑cased)
        - **Classifier:** BiLSTM layers
        - **Training Enhancements:**
            - SMOTE used for class balance
        - **Metrics Visualized:**
            - ROC Curve
            - Confusion Matrix
            - Accuracy over Epochs
        """
    )

# --- RESULTS PAGE ---------------------------------------------
elif page == "Results":
    st.title("📊 Model Results & Visualizations")
    st.write("Below are the visuals from model training & evaluation.")

    image_captions = {
        "images/class_distribution.png": "Class Distribution Before Balancing",
        "images/smote_before_after.png": "Class Distribution After SMOTE",
        "images/accuracy_epochs1.png": "Training & Validation Accuracy (Graph 1)",
        "images/accuracy_epochs2.png": "Training & Validation Accuracy (Graph 2)",
        "images/roc_curve.png": "Multiclass ROC Curve",
        "images/confusion_matrix.png": "Confusion Matrix",
    }

    for img_path, caption in image_captions.items():
        try:
            img = Image.open(img_path)
            st.image(img, caption=caption, use_column_width=True)
        except Exception as e:
            st.warning(f"⚠️ Image not found: {img_path}")
            st.error(str(e))


# Footer
st.markdown("---")
st.caption("Made with ❤️ using mBERT + BiLSTM for Hindi Emotion Detection")
