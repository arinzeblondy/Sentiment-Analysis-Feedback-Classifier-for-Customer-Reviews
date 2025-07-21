import streamlit as st
import pickle
import os
import re

# === Load model and vectorizer ===
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# === Preprocessing Function ===
def clean_text(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# === Sentiment Decoder ===
label_map = {
    0: ('Negative', '😠', '#FF4B4B'),  # red
    1: ('Neutral', '😐', '#FFD700'),   # yellow
    2: ('Positive', '😊', '#2ECC71')   # green
}

# === Custom CSS for Styling ===
st.markdown("""
    <style>
    .main {
        background-color: #f9fbfd;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
    }
    .stTextArea > label {
        font-size: 1.1rem;
        color: #555;
    }
    .stButton > button {
        background-color: #0066CC;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# === App Title ===
st.markdown("<h1 style='color:#0066CC;'>✈️ Airline Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:1.1rem;'>Paste a tweet or customer review to detect its <b>sentiment</b>. This helps airlines and brands understand customer emotions in real-time.</p>", unsafe_allow_html=True)

# === Input Section ===
with st.container():
    user_input = st.text_area("💬 Enter tweet or review:", "", height=150)

# === Prediction Button ===
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        label, emoji, color = label_map[prediction]

        st.markdown(f"""
            <div style='
                background-color:{color};
                padding: 1rem;
                border-radius: 10px;
                margin-top: 1.5rem;
                color: white;
                font-size: 1.3rem;
                font-weight: bold;
                text-align: center;'>
                {emoji} Predicted Sentiment: <span style='text-transform: uppercase'>{label}</span>
            </div>
        """, unsafe_allow_html=True)
