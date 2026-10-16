from flask import Flask, render_template, request, send_file, redirect, url_for
from keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import json
from datetime import datetime
from fpdf import FPDF

app = Flask(__name__)  # fixed _name to _name_

# Load the trained model
model = load_model('model/pneumonia-cnn.h5')
class_names = ['Normal', 'Pneumonia']
history_file = 'history.json'

# Save prediction result to history.json
def save_to_history(filename, prediction):
    record = {
        "filename": filename,
        "prediction": prediction,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append(record)
    with open(history_file, 'w') as f:
        json.dump(data, f, indent=4)

# Generate AI Summary
def generate_summary(prediction):
    if prediction == "Pneumonia":
        return "The uploaded X-ray suggests signs of Pneumonia. It is advised to consult a medical professional immediately."
    else:
        return "The X-ray appears to be normal. However, regular checkups are recommended for continued health monitoring."

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('error.html', message="No file uploaded.")
    file = request.files['file']
    if file.filename == '' or file.filename.startswith("._"):
        return render_template('error.html', message="Invalid file uploaded.")

    try:
        img = Image.open(file).convert('L')
    except UnidentifiedImageError:
        return render_template('error.html', message="Unrecognized or corrupted image file.")

    img = img.resize((150, 150))
    img_array = np.array(img).reshape(1, 150, 150, 1) / 255.0

    pred_prob = model.predict(img_array)[0][0]
    result = class_names[int(pred_prob > 0.5)]

    filename = file.filename
    save_to_history(filename, result)

    summary = generate_summary(result)

    pdf_path = f"{filename}_result.pdf"
    generate_pdf(filename, result, pdf_path)

    return render_template('result.html', prediction=result, summary=summary, pdf_path=pdf_path)

# PDF report generation
# PDF report generation without summary
def generate_pdf(filename, prediction, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Medical Image Analysis Report', ln=True, align='C')
    
    # Details
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f'File Name: {filename}', ln=True)
    pdf.cell(0, 10, f'Diagnosis Result: {prediction}', ln=True)
    pdf.cell(0, 10, f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
    
    # ❌ Not adding AI summary here
    
    pdf.output(pdf_path)
# Download route
@app.route('/download/<path:filename>')
def download_pdf(filename):
    return send_file(filename, as_attachment=True)

# History page
@app.route('/history')
def history():
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    return render_template('history.html', history=data)

# Start the Flask app
if __name__ == '_main_':
    print("🟢 Flask App is Starting...")
    app.run(host='127.0.0.1', port=5001, debug=True)