import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import json
import cv2
from PIL import Image
import base64
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SkinDiseaseDetector:
    def __init__(self):
        self.image_size = (224, 224)
        self.max_text_length = 200
        self.num_words = 10000
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.treatment_recommendations = None
        
    def preprocess_text(self, text):
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        except LookupError:
            # If stopwords still not available, proceed without stopword removal
            pass
        
        return ' '.join(tokens)
    
    def preprocess_image(self, image_path=None, image_array=None):
        if image_path is not None:
            img = Image.open(image_path)
        elif image_array is not None:
            img = Image.fromarray(image_array)
        else:
            raise ValueError("Either image_path or image_array must be provided")
            
        img = img.resize(self.image_size)
        img_array = np.array(img)
        img_array = img_array / 255.0
        return np.expand_dims(img_array, axis=0)
    
    def preprocess_base64_image(self, base64_string):
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
            
        # Decode base64 string to image
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img = img.resize(self.image_size)
        img_array = np.array(img)
        img_array = img_array / 255.0
        return np.expand_dims(img_array, axis=0)

    def load_data(self, csv_path, image_dir):
        # Load text data
        df = pd.read_csv(csv_path)
        df['processed_text'] = df['Text'].apply(self.preprocess_text)
        
        # Prepare text data
        self.tokenizer = Tokenizer(num_words=self.num_words)
        self.tokenizer.fit_on_texts(df['processed_text'])
        sequences = self.tokenizer.texts_to_sequences(df['processed_text'])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_text_length)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(df['Disease name'])
        
        # Load treatment, prevention, and medicines info
        self.disease_info = {
            'Acne': {
                'treatment': 'Use gentle cleansers, avoid picking, consider topical retinoids or benzoyl peroxide',
                'prevention': 'Wash your face twice daily, avoid oily cosmetics, keep hair clean and off the face',
                'medicines': 'Benzoyl peroxide, Salicylic acid, Topical retinoids'
            },
            'Eczema': {
                'treatment': 'Moisturize regularly, use mild soaps, avoid triggers, consider topical corticosteroids',
                'prevention': 'Avoid known irritants, keep skin moisturized, wear soft fabrics',
                'medicines': 'Topical corticosteroids, Antihistamines, Moisturizers'
            },
            'Psoriasis': {
                'treatment': 'Moisturize, use medicated shampoos, consider phototherapy or systemic medications',
                'prevention': 'Avoid skin injuries, manage stress, avoid smoking and excessive alcohol',
                'medicines': 'Topical corticosteroids, Vitamin D analogues, Biologics'
            },
            'Rosacea': {
                'treatment': 'Use gentle skincare, avoid triggers, consider topical or oral antibiotics',
                'prevention': 'Avoid triggers such as spicy foods, alcohol, and sun exposure',
                'medicines': 'Metronidazole cream, Azelaic acid, Oral antibiotics'
            },
            'Vitiligo': {
                'treatment': 'Use sunscreen, consider phototherapy or topical corticosteroids',
                'prevention': 'Protect skin from sunburn, avoid skin trauma',
                'medicines': 'Topical corticosteroids, Calcineurin inhibitors'
            },
            'Hives (Urticaria)': {
                'treatment': 'Identify triggers, use antihistamines, avoid scratching',
                'prevention': 'Avoid known allergens, wear loose clothing',
                'medicines': 'Antihistamines, Corticosteroids (short-term)'
            },
            'Contact Dermatitis': {
                'treatment': 'Identify and avoid irritants, use gentle skincare, consider topical corticosteroids',
                'prevention': 'Avoid contact with known irritants/allergens, use protective gloves',
                'medicines': 'Topical corticosteroids, Emollients'
            },
            'Ringworm (Tinea Corporis)': {
                'treatment': 'Use antifungal creams, keep area dry and clean',
                'prevention': 'Keep skin clean and dry, avoid sharing personal items',
                'medicines': 'Clotrimazole, Terbinafine, Miconazole'
            },
            "Athlete's Foot (Tinea Pedis)": {
                'treatment': 'Keep feet dry, use antifungal powders or creams',
                'prevention': 'Wear breathable shoes, change socks regularly, keep feet dry',
                'medicines': 'Tolnaftate, Terbinafine, Clotrimazole'
            },
            'Scabies': {
                'treatment': 'Use prescribed scabicide, wash bedding and clothing, treat close contacts',
                'prevention': 'Avoid skin-to-skin contact with infected individuals, wash clothing and bedding',
                'medicines': 'Permethrin cream, Ivermectin'
            },
            'Impetigo': {
                'treatment': 'Use prescribed antibiotics, keep area clean and covered',
                'prevention': 'Maintain good hygiene, avoid scratching sores',
                'medicines': 'Mupirocin ointment, Oral antibiotics'
            },
            'Folliculitis': {
                'treatment': 'Keep area clean, use warm compresses, consider antibiotics if severe',
                'prevention': 'Avoid tight clothing, keep skin clean, avoid shaving over bumps',
                'medicines': 'Topical antibiotics, Antibacterial washes'
            },
            'Shingles (Herpes Zoster)': {
                'treatment': 'Use antiviral medications, manage pain, keep rash clean and covered',
                'prevention': 'Get vaccinated (shingles vaccine), avoid contact with people who have not had chickenpox',
                'medicines': 'Acyclovir, Valacyclovir, Famciclovir'
            }
        }
        
        return padded_sequences, encoded_labels
    
    def create_model(self, num_classes):
        # Image branch
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        
        image_input = layers.Input(shape=(224, 224, 3))
        x = base_model(image_input)
        x = layers.GlobalAveragePooling2D()(x)
        image_branch = layers.Dense(256, activation='relu')(x)
        
        # Text branch
        text_input = layers.Input(shape=(self.max_text_length,))
        y = layers.Embedding(self.num_words, 128)(text_input)
        y = layers.LSTM(128)(y)
        text_branch = layers.Dense(256, activation='relu')(y)
        
        # Combine branches
        combined = layers.concatenate([image_branch, text_branch])
        
        # Output layers
        z = layers.Dense(512, activation='relu')(combined)
        z = layers.Dropout(0.5)(z)
        z = layers.Dense(256, activation='relu')(z)
        z = layers.Dropout(0.3)(z)
        output = layers.Dense(num_classes, activation='softmax')(z)
        
        model = models.Model(inputs=[image_input, text_input], outputs=output)
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def train(self, csv_path, image_dir, epochs=10, batch_size=32):
        # Load and preprocess data
        padded_sequences, encoded_labels = self.load_data(csv_path, image_dir)
        
        # Create and train model
        self.model = self.create_model(len(self.label_encoder.classes_))
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            padded_sequences, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Train model
        history = self.model.fit(
            [np.zeros((len(X_train), 224, 224, 3)), X_train],  # Placeholder for images
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([np.zeros((len(X_val), 224, 224, 3)), X_val], y_val)
        )
        
        return history
    
    def predict(self, image_path, text_description):
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Preprocess inputs
        processed_text = self.preprocess_text(text_description)
        text_sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_text = pad_sequences(text_sequence, maxlen=self.max_text_length)
        
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        prediction = self.model.predict([img_array, padded_text])
        predicted_class = self.label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = float(np.max(prediction))
        
        # Get disease info
        info = self.disease_info.get(predicted_class, {})
        treatment = info.get('treatment', 'Please consult a dermatologist for specific treatment recommendations.')
        prevention = info.get('prevention', 'Please consult a dermatologist for prevention advice.')
        medicines = info.get('medicines', 'Please consult a dermatologist for medicine recommendations.')
        
        return {
            'predicted_disease': predicted_class,
            'confidence': confidence,
            'treatment_recommendation': treatment,
            'prevention': prevention,
            'medicines': medicines
        }

    def predict_realtime(self, frame, text_description):
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Preprocess frame
        processed_frame = self.preprocess_image(image_array=frame)
        
        # Preprocess text
        processed_text = self.preprocess_text(text_description)
        text_sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_text = pad_sequences(text_sequence, maxlen=self.max_text_length)
        
        # Make prediction
        prediction = self.model.predict([processed_frame, padded_text], verbose=0)
        predicted_class = self.label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = float(np.max(prediction))
        
        # Get disease info
        info = self.disease_info.get(predicted_class, {})
        treatment = info.get('treatment', 'Please consult a dermatologist for specific treatment recommendations.')
        prevention = info.get('prevention', 'Please consult a dermatologist for prevention advice.')
        medicines = info.get('medicines', 'Please consult a dermatologist for medicine recommendations.')
        
        return {
            'predicted_disease': predicted_class,
            'confidence': confidence,
            'treatment_recommendation': treatment,
            'prevention': prevention,
            'medicines': medicines
        }

def main():
    # Initialize detector
    detector = SkinDiseaseDetector()
    
    # Train model
    print("Training model...")
    history = detector.train('data.csv', 'archive (1)/DermNetNZ')
    print("Training completed!")
    
    # Example prediction
    image_path = "path_to_test_image.jpg"  # Replace with actual image path
    text_description = "I have these red, itchy patches on my skin that are getting worse."
    
    try:
        result = detector.predict(image_path, text_description)
        print("\nPrediction Results:")
        print(f"Predicted Disease: {result['predicted_disease']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Treatment Recommendation: {result['treatment_recommendation']}")
        print(f"Prevention Advice: {result['prevention']}")
        print(f"Medicine Recommendations: {result['medicines']}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main() 