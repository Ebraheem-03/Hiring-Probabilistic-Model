import os
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib  # To save and load the vectorizer

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''.join([page.extract_text() for page in reader.pages])
    return text

# In pipeline.py
def vectorize_text(text_data, fit=True):
    if fit:
        vectorizer = TfidfVectorizer(max_features=500)
        vectors = vectorizer.fit_transform(text_data)
        joblib.dump(vectorizer, "data/output/tfidf_vectorizer.pkl")  # Save vectorizer
    else:
        vectorizer = joblib.load("data/output/tfidf_vectorizer.pkl")
        vectors = vectorizer.transform(text_data)
    return vectors.toarray()


def cluster_resumes(vectors, n_clusters=5):
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(vectors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_vectors)
    return clusters

def process_department_resumes(department_dir):
    resumes = []
    resume_texts = []
    for filename in os.listdir(department_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(department_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            resumes.append(filename)
            resume_texts.append(text)
    vectors = vectorize_text(resume_texts, fit=True)  # Fit vectorizer during initial processing
    clusters = cluster_resumes(vectors)
    return resumes, resume_texts, clusters
