import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
st.write("Enter the resume content:")

# User input for resume content
input_resume = st.text_area("Resume Content", "")

if st.button("Classify"):
    # Preprocess the user input
    preprocessed_resume = preprocess_text(input_resume)
    
    # Transform the preprocessed resume using the TF-IDF vectorizer
    transformed_resume = vectorizer.transform([preprocessed_resume])
    
    # Predict the company name
    predicted_company = model.predict(transformed_resume)[0]
    
    # Get the confidence score of the prediction
    confidence_score = model.predict_proba(transformed_resume).max()
    
    st.write("Predicted Company:", predicted_company)
    st.write("Confidence Score:", confidence_score)
    
    # Visualize the distribution of resumes among different companies
    company_names = ["peoplesoft resume", "reactJS developer", "SQL developer lighting insight", "workday resumes"]
    company_counts = [20, 24, 14, 21]  # Replace with actual counts
    
    st.bar_chart({"Company Names": company_names, "Number of Resumes": company_counts})
