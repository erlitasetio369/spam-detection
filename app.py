import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import google.generativeai as genai
nltk.download('punkt')
nltk.download('stopwords')

# PorterStemmer object initiate
ps = PorterStemmer()

def transform_text(text):
    # lower casing
    text = text.lower()
    # converting text into list of words
    text = nltk.word_tokenize(text)

    y = []
    # removing special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # removing stopwords/helping words
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Normalization of word i.e converting words into their base form.
    for j in text:
        y.append(ps.stem(j))

    return " ".join(y)

tfidf = pd.read_pickle('models/vectorizer.pkl')
model = pd.read_pickle('models/model.pkl')

st.title('*SMS/Email Spam Detection*')
st.markdown("-------------------")
st.markdown('##### Discover if your text messages are safe or sneaky! Try this SMS Spam Detection now!')

st.markdown(" ")
user_input = st.text_input('Enter your text here')

if st.button("Check for Spam"):
    if user_input[:] == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess user input
        transformed_txt = transform_text(user_input)
        converted_num = tfidf.transform([transformed_txt])
        result = model.predict(converted_num)[0]

        # Display detection
        if result == 1:
            st.error("SPAM")
        else:
            st.success("Not Spam")


# import streamlit as st
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import google.generativeai as genai
# # from google_generative_ai import generate_response  # Placeholder for Google Generative AI

# # Load the spam detection model
# def load_spam_model():
#     vectorizer = TfidfVectorizer()
#     clf = MultinomialNB()
#     return vectorizer, clf

# # Placeholder for Google's Generative AI function
# def generate_google_response(text):
#     # Replace this function with the actual implementation using Google's Generative AI
#     return generate_response(text)

# def main():
#     st.title("Spam Detection with Streamlit and Google's Generative AI")

#     # Text input
#     user_input = st.text_area("Enter your text:", "")

#     # Check for spam
#     if st.button("Check for Spam"):
#         # Load the spam detection model
#         vectorizer, clf = load_spam_model()

#         # Transform user input
#         input_vectorized = vectorizer.transform([user_input])

#         # Make prediction
#         prediction = clf.predict(input_vectorized)

#         if prediction[0] == 0:
#             st.write("The text is not spam.")
#             # Generate a response using Google's Generative AI
#             google_response = generate_google_response(user_input)
#             st.write("Generated Response:", google_response)
#         else:
#             st.write("The text is spam.")

# if __name__ == "__main__":
#     main()