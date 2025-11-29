# 🪔 Hindi Emotion Recognition with mBERT Embeddings + BiLSTM (TensorFlow/Keras)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Transformers](https://img.shields.io/badge/Transformers-🤗-yellow)](https://huggingface.co/transformers/)
[![PyTorch (embeddings)](https://img.shields.io/badge/PyTorch-embed_only-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![imblearn](https://img.shields.io/badge/imbalanced--learn-SMOTE-7B1FA2)](https://imbalanced-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End‑to‑end Hindi emotion classification using mBERT mean‑pooled sentence embeddings, class rebalancing with SMOTE, and a lightweight BiLSTM classifier.

</div>

---

## 📌 Overview

This repository builds a 5‑class Hindi emotion recognizer from an Excel dataset:
- Cleans/tokenizes Hindi text (stopwords + simple stemming)
- Extracts multilingual BERT embeddings (bert-base-multilingual-cased)
- Balances classes in embedding space with SMOTE
- Trains a BiLSTM classifier on the embeddings
- Evaluates with classification report, confusion matrix, and ROC curves

Note: Research/education project.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🔡 Hindi NLP Prep | Indic tokenization, Hindi stopwords removal, and simple suffix-based stemming. |
| 🌐 mBERT Embeddings | Mean‑pooled token embeddings from bert-base-multilingual-cased (768‑D). |
| ⚖️ Class Balancing | SMOTE oversampling in embedding space for balanced training data. |
| 🧠 Classifier | BiLSTM(64) → BN → Dropout → Dense(32) → BN → Dropout → Softmax(5). |
| 📊 Evaluation | Accuracy, precision/recall/F1 (per class + macro/weighted), confusion matrix, multi‑class ROC. |

---

## 📂 Project Structure

```plaintext
hindi-emotion-bilstm/
├── HindiBilstmFinal.ipynb        # Main notebook (embeddings → SMOTE → train → eval)
├── Bhaav-Dataset.xlsx            # Dataset (not tracked in LFS by default)
├── HindiEmotion.h5               # Trained Keras model (optional artifact)
├── app.py                        # Simple inference script/app (optional)
├── REVIEW_3.pdf                  # Report/notes (optional)
├── README.md
└── LICENSE
```

---

## 📦 Dataset

- File: Bhaav-Dataset.xlsx
- Columns:
  - Sentence: Hindi text
  - Annotation: integer label in [0..4]
- Example mapping used in notebook (update to match your sheet):
  - 0 → anger
  - 1 → joy
  - 2 → sad
  - 3 → surprise
  - 4 → neutral

Tip: Ensure the mapping is consistent everywhere (plots, label names, confusion matrix).

---

## 🧠 Technical Details

- Embeddings
  - Model: bert-base-multilingual-cased (Hugging Face)
  - Tokenization: max_length=50, truncation/padding to fixed length
  - Pooling: mean across token embeddings → 768‑D vector per sentence
- Rebalancing
  - SMOTE(random_state=42) on the 768‑D embeddings
- Split
  - Stratified 80/20 train/test after SMOTE
- Classifier
  - Input: (1, 768) sequence per sample
  - BiLSTM(64, return_sequences=False) → BatchNorm → Dropout(0.5)
  - Dense(32, relu) → BatchNorm → Dropout(0.5)
  - Dense(5, softmax)
  - Loss: sparse_categorical_crossentropy
  - Optimizer: Adam
  - Epochs: 100, batch size: 64
- Reported (your run)
  - Val accuracy ≈ 0.906 (see notebook for per‑class metrics)

Note: SMOTE on embeddings is pragmatic but can introduce artifacts; validate on a naturally distributed hold‑out if available.

---

## 🚀 Getting Started

### Installation
```bash
pip install tensorflow tensorflow-hub tensorflow-text -U
pip install transformers sentencepiece
pip install torch               # required because embeddings use AutoModel (PyTorch)
pip install imbalanced-learn
pip install indic-nlp-library
pip install openpyxl matplotlib seaborn scikit-learn tqdm
```

### Run the Notebook
1) Place Bhaav-Dataset.xlsx at the configured path (e.g., /content/Bhaav-Dataset.xlsx).
2) Execute cells in HindiBilstmFinal.ipynb:
   - Load + inspect dataset
   - Download Hindi stopwords (or provide stopwords-hi.txt locally)
   - Tokenize/clean (optional for analysis)
   - Encode with mBERT → X_bert (N, 768)
   - SMOTE → train/test split
   - Train BiLSTM and evaluate

---

## 🧪 Inference

Example snippet to predict a single sentence with the trained model and the same mBERT encoder:

```python
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.models import load_model

labels = {0:'anger',1:'joy',2:'sad',3:'surprise',4:'neutral'}
tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
bert = AutoModel.from_pretrained("bert-base-multilingual-cased").eval()
clf = load_model("HindiEmotion.h5")

def encode(text):
    inp = tok([text], return_tensors="pt", max_length=50, truncation=True, padding="max_length")
    with torch.no_grad():
        out = bert(**inp).last_hidden_state  # (1, seq, 768)
        vec = out.mean(dim=1).cpu().numpy()  # (1, 768)
    return vec.reshape(1, 1, 768)            # match BiLSTM input

x = encode("आज मन बहुत खुश है")
probs = clf.predict(x)[0]
pred = labels[int(probs.argmax())]
print(pred, probs.max())
```

Ensure the label order matches the model trained mapping.

---

## ⚖️ Limitations

- mBERT sentence mean‑pooling is simple; CLS pooling or sentence transformers may perform better.
- SMOTE on embeddings may not always reflect real text distributions.
- Basic Hindi stemmer and stopword list are heuristic; consider a stronger pipeline (morph analyzers, contextual normalization).
- Evaluate fairness across topics and dialects; avoid deployment without thorough review.

---

## 🧩 Next Steps

- Replace mean pooling with CLS token or use sentence‑transformers (e.g., paraphrase-multilingual-MiniLM-L12-v2).
- Add early stopping and LR schedules; tune epochs/batch size.
- Try class weights or focal loss instead of SMOTE.
- Calibrate probabilities (Platt/temperature scaling) for downstream use.

---

## 🧪 Reproducibility

- Fix random seeds and log package versions.
- Save:
  - stopwords-hi.txt
  - label mapping
  - trained model (HindiEmotion.h5)
- Keep tokenization/max_length identical between train and inference.

---

## 📄 License

Released under the MIT License. See LICENSE.


⭐️ If this repo helps, a star would be appreciated!
