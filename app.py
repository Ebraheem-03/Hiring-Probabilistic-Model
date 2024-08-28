import gradio as gr
import torch
from pipeline import extract_text_from_pdf, vectorize_text
from utils import evaluate_resume_with_job_description  # Import the function from utils.py

def load_pdf_to_text(pdf_file):
    text = extract_text_from_pdf(pdf_file.name)
    return text

def predict_resume_fit(resume_files, job_description):
    results = []
    
    # Extract and vectorize job description
    job_description_text = vectorize_text([job_description])
    
    for resume_file in resume_files:
        # Extract text from resume
        resume_text = load_pdf_to_text(resume_file)
        
        # Vectorize resume text
        resume_vector = vectorize_text([resume_text])
        
        # Calculate cosine similarity
        similarity_score = evaluate_resume_with_job_description(resume_text, job_description)
        fit_probability = round(similarity_score * 100, 2)
        results.append(f"Resume fit probability for {resume_file.name}: {fit_probability}%")
    
    return "\n".join(results)

# Updated Interface definition
interface = gr.Interface(
    fn=predict_resume_fit,
    inputs=[gr.File(file_count="multiple", label="Upload Resumes (PDF)"), gr.Textbox(label="Job Description")],
    outputs=gr.Textbox(label="Fit Probabilities"),
    title="Resume Fit Predictor",
    description="Upload resume PDFs and provide a job description to predict fit.",
)

if __name__ == "__main__":
    interface.launch()
