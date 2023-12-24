import streamlit as st
import google.generativeai as genai
from transformers import pipeline

def detect_spam_with_bert(text):
    classifier = pipeline('sentiment-analysis')
    result = classifier(text)[0]
    spam_threshold = 0.5
    is_spam = result['score'] < spam_threshold
    return is_spam, result

def main():
    st.title("Spam Detection App")
    st.write("Enter a text to check if it's spam or not.")

    # Memasukkan teks dari pengguna
    user_input = st.text_area("Input Text", "")

    if st.button("Check Spam"):
        # Menjalankan deteksi spam
        is_spam, classification_result = detect_spam_with_bert(user_input)

        # Menampilkan hasil deteksi
        if is_spam:
            st.error("SPAM")
        else:
            st.success("Not Spam")

if __name__ == "__main__":
    main()
