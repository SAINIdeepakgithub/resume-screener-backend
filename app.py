from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text_from_pdf(file_path):
    pdf = PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    return text

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    scores = cosine_similarity([job_vector], resume_vectors).flatten()
    return [round(score * 100, 2) for score in scores]

@app.route('/upload', methods=['POST'])
def upload_files():
    job_description = request.form['job_description']
    files = request.files.getlist('resumes')

    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    resume_texts = []
    file_names = []

    for file in files:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        text = extract_text_from_pdf(filename)
        resume_texts.append(text)
        file_names.append(file.filename)

    scores = rank_resumes(job_description, resume_texts)

    results = [{"filename": name, "score": score} for name, score in zip(file_names, scores)]
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
