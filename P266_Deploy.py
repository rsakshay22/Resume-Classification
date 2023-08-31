import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from docx import Document

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenization (splitting the text into words)
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    # Join the words back into a preprocessed text
    preprocessed_text = ' '.join(words)
    
    return preprocessed_text

# Load the trained model and TF-IDF vectorizer
vectorizer = joblib.load('C://Users//Akshay//Desktop//DS Project2//G4___//tfidf_vectorizer_.pkl')
model = joblib.load('C://Users//Akshay//Desktop//DS Project2//G4___//trained_model.pkl')

# Streamlit UI
st.title("Resume Classification App")
st.write("Upload a Word document containing resumes (in .docx format):")

# User input for uploading a file
uploaded_file = st.file_uploader("Upload File", type=["docx"])

if uploaded_file is not None:
    doc = Document(uploaded_file)
    resumes = [para.text for para in doc.paragraphs if para.text.strip() != ""]
    
    if st.button("Classify Resumes"):
        st.write("Classifying Resumes:")
        for idx, resume in enumerate(resumes):
            preprocessed_resume = preprocess_text(resume)
            transformed_resume = vectorizer.transform([preprocessed_resume])
            
            # Predict the company name
            predicted_company = model.predict(transformed_resume)[0]
            confidence_score = model.predict_proba(transformed_resume).max()
            
            st.write(f"Resume {idx + 1}: Predicted Company - {predicted_company}, Confidence Score - {confidence_score:.2f}")
