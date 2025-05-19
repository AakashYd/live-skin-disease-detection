# Skin Disease Detection System

A multimodal deep learning system that can detect skin diseases and provide treatment recommendations based on both image and text descriptions.

## Features

- Image-based skin disease detection using EfficientNetB0
- Text-based symptom analysis using LSTM
- Treatment recommendations for various skin conditions
- Web interface for easy interaction
- Support for multiple skin conditions including:
  - Acne
  - Eczema
  - Psoriasis
  - Rosacea
  - Vitiligo
  - Hives (Urticaria)
  - Contact Dermatitis
  - Ringworm (Tinea Corporis)
  - Athlete's Foot (Tinea Pedis)
  - Scabies
  - Impetigo
  - Folliculitis
  - Shingles (Herpes Zoster)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd skin-disease-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload a skin image and provide a description of your symptoms.

4. The system will analyze the input and provide:
   - Predicted skin disease
   - Confidence score
   - Treatment recommendations

## Project Structure

- `skin_disease_detector.py`: Main model implementation
- `app.py`: Flask web application
- `templates/index.html`: Web interface
- `data.csv`: Text data for training
- `archive (1)/DermNetNZ/`: Image dataset

## Model Architecture

The system uses a multimodal approach combining:
- EfficientNetB0 for image processing
- LSTM for text processing
- Combined dense layers for final prediction

## Notes

- The system is trained on a limited dataset and should be used as a preliminary self-assessment tool
- Always consult a healthcare professional for proper diagnosis and treatment
- The model's predictions are not a substitute for professional medical advice

## License

This project is licensed under the MIT License - see the LICENSE file for details. 