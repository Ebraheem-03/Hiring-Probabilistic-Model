import os
import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pipeline import process_department_resumes
from groq_api import create_job_description_for_clusters
import json

def evaluate_resume_with_job_description(resume_text, job_description):
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Combine resume and job description into a list
    documents = [resume_text, job_description]
    
    # Create TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Calculate cosine similarity between resume and job description
    cosine_similarity = (tfidf_matrix[0, :] @ tfidf_matrix[1, :].T).toarray()[0, 0]
    
    return cosine_similarity

def generate_training_data(department_dirs):
    all_job_descriptions = []  # List to store all job descriptions

    
    with open("data/output/trained_data.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Resume_ID", "Department", "Job Description #","Cosine Similarity Score", "Label"])
        
        for department_dir in department_dirs:
            print(f"Processing department: {department_dir}")
            resumes, resume_texts, _ = process_department_resumes(department_dir)
            print(f"Found {len(resumes)} resumes in {department_dir}")
            
            # Vectorize resume texts
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(resume_texts)
            
            # Clustering resumes into 6 clusters
            num_clusters = 6
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            clusters = kmeans.fit_predict(X)
            print(f"Clustering completed for {department_dir} into {num_clusters} clusters")
            
            # Generate job descriptions for each cluster
            job_descriptions = create_job_description_for_clusters(resume_texts, clusters)
            all_job_descriptions.extend(job_descriptions)  # Save job descriptions
            print(f"Generated job descriptions for clusters in {department_dir}")
            
            for i, resume in enumerate(resumes):
                cluster_id = clusters[i]
                job_description = job_descriptions[cluster_id]
                job_description_id = job_descriptions
                score = evaluate_resume_with_job_description(resume_texts[i], job_description)
                label = 1 if score >= 0.6 else 0
                writer.writerow([resume, os.path.basename(department_dir), cluster_id+1, score, label])
            print(f"Finished processing {department_dir}")
        
    print("Training data generation completed.")
    return all_job_descriptions  # Return the list of job descriptions


if __name__ == "__main__":
    department_dirs = [f"data/data/{dept}" for dept in os.listdir("data/data")]
    job_descriptions = generate_training_data(department_dirs)
    
     # Save all job descriptions to a JSON file
    job_descriptions_path = "data/output/job_descriptions.json"
    with open(job_descriptions_path, "w") as json_file:
        json.dump(job_descriptions, json_file, indent=4)
        
    print(f"Total job descriptions generated: {len(job_descriptions)}")
    print(f"All job descriptions have been saved to {job_descriptions_path}")  