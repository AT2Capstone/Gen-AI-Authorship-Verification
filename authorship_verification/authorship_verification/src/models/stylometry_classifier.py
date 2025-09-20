# File: src/models/simple_stylometry.py

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from collections import Counter
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def load_jsonl_data(filepath):
    """Load data directly from your JSONL file format"""
    texts = []
    labels = []
    
    print(f"📖 Loading data from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    texts.append(data['text'])
                    labels.append(data['label'])
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping line {line_num}: {e}")
                except KeyError as e:
                    print(f"⚠️ Missing field in line {line_num}: {e}")
    
    print(f"✅ Loaded {len(texts)} samples")
    print(f"📊 Class distribution: {Counter(labels)}")
    
    return texts, labels

class StylometryFeatures(BaseEstimator, TransformerMixin):
    """Extract stylometric features from text"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            features.append(self._get_features(text))
        return np.array(features)
    
    def _get_features(self, text):
        """Extract 15 key stylometric features"""
        if not text:
            return [0] * 15
        
        # Tokenize
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        
        if not words or not sentences:
            return [0] * 15
        
        # Basic stats
        char_count = len(text)
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Calculate features
        features = [
            char_count,
            word_count, 
            sentence_count,
            word_count / sentence_count,  # avg words per sentence
            np.mean([len(w) for w in words]),  # avg word length
            len(set(words)) / word_count,  # lexical diversity
            sum(1 for w in words if w in self.stop_words) / word_count,  # function words
            text.count('!') / char_count,  # exclamation ratio
            text.count('?') / char_count,  # question ratio
            text.count(',') / char_count,  # comma ratio
            sum(1 for c in text if c.isupper()) / char_count,  # uppercase ratio
            sum(1 for w in words if len(w) <= 3) / word_count,  # short words
            sum(1 for w in words if len(w) >= 7) / word_count,  # long words
            sum(1 for freq in Counter(words).values() if freq == 1) / word_count,  # hapax legomena
            text.count('"') / char_count,  # quotation ratio
        ]
        
        return features

class SimpleStylometryClassifier:
    """Simple stylometry classifier for your exact use case"""
    
    def __init__(self, model_type='logistic', max_features=3000):
        self.model_type = model_type
        self.max_features = max_features
        
        # Components
        self.stylometry_extractor = StylometryFeatures()
        self.text_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Reduced for speed
            stop_words='english',
            lowercase=True
        )
        self.scaler = StandardScaler()
        
        # Model
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def fit(self, texts, labels):
        """Train the classifier"""
        print("🔧 Extracting stylometric features...")
        style_features = self.stylometry_extractor.fit_transform(texts)
        
        print("🔧 Extracting text features...")
        text_features = self.text_vectorizer.fit_transform(texts).toarray()
        
        print("🔧 Combining features...")
        combined_features = np.hstack([style_features, text_features])
        combined_features = self.scaler.fit_transform(combined_features)
        
        print(f"🚀 Training {self.model_type} model...")
        self.model.fit(combined_features, labels)
        
        return self
    
    def predict_proba(self, texts):
        """Get probability scores for ensemble"""
        style_features = self.stylometry_extractor.transform(texts)
        text_features = self.text_vectorizer.transform(texts).toarray()
        combined_features = np.hstack([style_features, text_features])
        combined_features = self.scaler.transform(combined_features)
        
        return self.model.predict_proba(combined_features)
    
    def predict(self, texts):
        """Get predictions"""
        probabilities = self.predict_proba(texts)
        return np.argmax(probabilities, axis=1)
    
    def save_model(self, filepath):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"💾 Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load trained model"""
        model = joblib.load(filepath)
        print(f"📂 Model loaded from: {filepath}")
        return model

def train_stylometry_classifier(data_path, model_save_path='models/saved/stylometry.pkl', test_size=0.2):
    """Complete training pipeline"""
    
    # Load data
    texts, labels = load_jsonl_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    print(f"\n📊 Data split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train classifier
    classifier = SimpleStylometryClassifier(model_type='logistic', max_features=3000)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    train_pred = classifier.predict(X_train)
    test_pred = classifier.predict(X_test)
    test_probs = classifier.predict_proba(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n🎯 Results:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, test_pred))
    
    # Save model
    classifier.save_model(model_save_path)
    
    return classifier, test_acc, test_probs

def get_ensemble_probabilities(model_path, texts):
    """Load model and get probabilities for ensemble"""
    classifier = SimpleStylometryClassifier.load_model(model_path)
    probabilities = classifier.predict_proba(texts)
    
    print(f"✅ Generated probabilities for {len(texts)} samples")
    print(f"📊 Probability matrix shape: {probabilities.shape}")
    
    return probabilities

# Main execution
if __name__ == "__main__":
    # Your exact file path
    train_data_path = 'data/train/train.jsonl'  # Adjust if different
    model_save_path = 'models/saved/stylometry_classifier.pkl'
    
    print("🎯 Simple Stylometry Classifier for Your Project")
    print("=" * 50)
    
    # Check if model already exists
    if os.path.exists(model_save_path):
        print("🔄 Found existing model, loading...")
        classifier = SimpleStylometryClassifier.load_model(model_save_path)
        
        # Test with sample text
        sample_texts = ["This is a sample text for testing the classifier."]
        probs = classifier.predict_proba(sample_texts)
        print(f"Sample probability: {probs[0]}")
        
    else:
        print("🚀 Training new model...")
        classifier, accuracy, test_probs = train_stylometry_classifier(
            train_data_path, 
            model_save_path
        )
        print(f"\n✅ Training complete! Test accuracy: {accuracy:.4f}")
    
    print(f"\n🎯 Model ready for ensemble integration!")
    print(f"Use: probabilities = classifier.predict_proba(your_texts)")