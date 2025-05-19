# 🧠 Live Skin Disease Detection System

A multimodal deep learning web application for real-time skin disease detection using both image and symptom-based inputs.

## 🚀 Project Overview
This project combines **computer vision** and **natural language processing** to diagnose skin diseases more accurately. The system accepts:
- 📷 An image of the affected skin area.
- ✍️ A brief description of the symptoms.

It uses:
- **EfficientNetB0** for image classification.
- **LSTM-based neural network** for processing symptom descriptions.
- A **multimodal fusion** technique to improve diagnostic performance.
- A user-friendly **Flask web interface** for live predictions.

## 🔬 Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Flask
- Scikit-learn
- Pandas, NumPy
- HTML/CSS

## 🌐 Features
- Upload a skin image and symptoms through the web interface
- Real-time disease prediction with high accuracy
- Trained on a curated dataset of skin diseases
- Handles both visual and textual symptom input for improved diagnosis

## 📁 Project Structure
- `skin_disease_detector.py`: Model training and multimodal fusion logic
- `app.py`: Flask backend for the web interface
- `templates/`: HTML files for the web interface
- `static/`: Image uploads and CSS
- `data.csv`: Symptom dataset (text input)
- `models/`: Saved model weights

## 🧪 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/live-skin-disease-detection.git
   cd live-skin-disease-detection
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Flask app:
   ```bash
   python app.py
4. Open http://127.0.0.1:5000 in your browser.



