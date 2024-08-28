import os
import time
from groq import Groq
from dotenv import load_dotenv
import numpy as np
from requests.exceptions import HTTPError

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

MAX_CONTEXT_LENGTH = 4096  # Maximum length of the context (adjust this as needed)

def generate_job_description(text):
    """
    Generate a job description based on the combined text of resumes in a cluster.
    
    Parameters:
    - text (str): Combined text from resumes in a cluster.
    
    Returns:
    - job_description (str): Generated job description.
    """
    client = Groq(api_key=api_key)
    retry_attempts = 5

    # Define a more specific prompt
    prompt = f"Create a detailed job description based on the following resume content:\n\n{text}\n\nJob Description:"

    for attempt in range(retry_attempts):
        try:
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Adjust temperature as needed
                max_tokens=1024,  # Ensure this is enough for the job description
                top_p=1,
            )
            job_description = completion.choices[0].message.content.strip()
            return job_description

        except HTTPError as e:
            error_message = str(e)
            if 'Rate limit reached' in error_message:
                if attempt < retry_attempts - 1:
                    wait_time = 40
                    print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retry attempts reached. Skipping...")
                    return ""
            else:
                print(f"HTTP error occurred: {e}")
                return ""

        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            return ""

def extract_wait_time(error_message):
    """
    Extract the wait time from the error message if rate limit is reached.
    
    Parameters:
    - error_message (str): The error message returned by the API.
    
    Returns:
    - wait_time (int): The time to wait in seconds.
    """
    import re
    match = re.search(r'try again in (\d+(\.\d+)?)s', error_message)
    if match:
        return int(float(match.group(1))) + 5  # Adding a buffer time of 5 seconds
    return 60  # Default wait time if not found

def create_job_description_for_clusters(resume_texts, clusters):
    job_descriptions = []
    unique_clusters = np.unique(clusters)
    
    for cluster_id in unique_clusters:
        cluster_texts = [resume_texts[i] for i in range(len(resume_texts)) if clusters[i] == cluster_id]
        # Select one random resume from the cluster
        resume_for_description = np.random.choice(cluster_texts)
        job_description = generate_job_description(resume_for_description)
        job_descriptions.append(job_description)
    
    return job_descriptions