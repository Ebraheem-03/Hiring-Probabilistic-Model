title: Hiring Probabilistic Model
emoji: 📚
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 4.42.0
app_file: app.py
pinned: false

# Resume Fit Predictor

This project predicts the fit of a resume for a given job description using clustering and neural networks.

## Project Structure

- `app.py`: Main application file for running the Gradio frontend.
- `model.py`: Model definition, training, and evaluation.
- `pipeline.py`: Text extraction, vectorization, and clustering.
- `groq_api.py`: Interfacing with the GROQ API for generating job descriptions.
- `utils.py`: Utility functions for handling PDFs, data loading, and CSV operations.
- `requirements.txt`: Python dependencies.
- `data/`: Directory for input PDFs and output CSVs.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/resume-fit-predictor.git
   cd resume-fit-predictor
