import streamlit as st
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Advertisement Classification App",
    page_icon="🛒",
    layout="centered",
)

# Inject CSS for enhanced design
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e3f2fd; /* Light sky blue */
    }

    .main-container {
        background-color: #ffffff; 
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.2);
        margin: auto;
        max-width: 700px;
        text-align: center;
        transition: transform 0.2s;
    }

    .main-container:hover {
        transform: scale(1.02);
    }

    .header {
        color: #1565c0;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .subtitle {
        color: #455a64;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .stTextArea > div > textarea {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border: 2px solid #90caf9;
        transition: border-color 0.3s;
    }

    .stTextArea > div > textarea:focus {
        border-color: #42a5f5;
    }

    .stButton button {
        background-color: #42a5f5;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s, transform 0.2s;
    }

    .stButton button:hover {
        background-color: #1e88e5;
        transform: scale(1.05);
    }

    .result-box {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }

    .result-text {
        font-size: 20px;
        font-weight: bold;
        color: #2e7d32;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Map for class descriptions and emojis
class_map = {
    "art and music": "🎨🎵 Dive into creativity! This ad is about art and music.",
    "food": "🍔 Satisfy your cravings! This ad promotes food and beverages.",
    "history": "📜 Explore the past! This ad highlights historical content.",
    "manufacturing": "🏭 Build the future! This ad is related to manufacturing.",
    "science and technology": "🔬💻 Discover innovation! This ad focuses on science and technology.",
    "travel": "✈️ Explore the world! This ad is all about travel adventures.",
}

# Function to save label classes (Run this during training)
def save_label_encoder(label_encoder):
    np.save("label_classes.npy", label_encoder.classes_)
    print("Label classes saved to label_classes.npy")

# Function to load label classes
@st.cache_resource
def load_label_encoder():
    label_encoder = LabelEncoder()
    if os.path.exists("label_classes.npy"):
        label_encoder.classes_ = np.load("label_classes.npy", allow_pickle=True)
    else:
        raise FileNotFoundError("label_classes.npy not found. Please ensure it's in the project directory.")
    return label_encoder

# Function to load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = TFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_encoder.classes_),
    )
    model.load_weights("bert_classification_weights.h5")
    return tokenizer, model

# Function to preprocess text for prediction
def preprocess_text(text, tokenizer, max_length=128):
    encodings = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="tf",
    )
    return encodings["input_ids"], encodings["attention_mask"]

# Function to predict text
def predict_text(text, tokenizer, model, label_encoder):
    input_ids, attention_mask = preprocess_text(text, tokenizer)
    predictions = model.predict({"input_ids": input_ids, "attention_mask": attention_mask})
    predicted_class = np.argmax(predictions.logits, axis=1)
    class_label = label_encoder.inverse_transform(predicted_class)[0]
    confidence = tf.nn.softmax(predictions.logits[0])[predicted_class[0]].numpy()
    return class_label, confidence

# Load Label Encoder
label_encoder = load_label_encoder()

# Load Model and Tokenizer
tokenizer, model = load_model_and_tokenizer()

# Streamlit App Interface
st.markdown(
    """
    <div class="main-container">
        <div class="header">Advertisement Classification App</div>
        <p class="subtitle">
            This app classifies advertisements based on text data (e.g., title, description).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input text
user_input = st.text_area("Enter advertisement text for classification:", "")

if st.button("Classify"):
    if user_input.strip():
        predicted_class, confidence = predict_text(user_input, tokenizer, model, label_encoder)
        if predicted_class in class_map:
            description = class_map[predicted_class]
            st.markdown(
                f"""
                <div class="result-box">
                    <p class="result-text">{description}</p>
                    <p class="result-text">Confidence: {confidence * 100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("The predicted class is not part of the predefined categories.")
    else:
        st.warning("Please enter text to classify.")
