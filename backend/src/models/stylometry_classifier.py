# File: src/models/simple_stylometry.py

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from collections import Counter
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab...")
    nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
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
        """Extract 15 key stylometric features with better normalization"""
        if not text or len(text.strip()) < 10:
            return [0] * 15
        
        # Tokenize with error handling
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha()]
        except:
            # Fallback to simple regex splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words or not sentences:
            return [0] * 15
        
        # Basic stats
        char_count = len(text)
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Add small epsilon to prevent division issues
        epsilon = 1e-8
        
        # Calculate features with better normalization
        features = [
            np.log(char_count + 1),  # Log transform for char count
            np.log(word_count + 1),  # Log transform for word count
            np.log(sentence_count + 1),  # Log transform for sentence count
            word_count / (sentence_count + epsilon),  # avg words per sentence
            np.mean([len(w) for w in words]),  # avg word length
            len(set(words)) / (word_count + epsilon),  # lexical diversity
            sum(1 for w in words if w in self.stop_words) / (word_count + epsilon),  # function words
            text.count('!') / (char_count + epsilon),  # exclamation ratio
            text.count('?') / (char_count + epsilon),  # question ratio
            text.count(',') / (char_count + epsilon),  # comma ratio
            sum(1 for c in text if c.isupper()) / (char_count + epsilon),  # uppercase ratio
            sum(1 for w in words if len(w) <= 3) / (word_count + epsilon),  # short words
            sum(1 for w in words if len(w) >= 7) / (word_count + epsilon),  # long words
            sum(1 for freq in Counter(words).values() if freq == 1) / (word_count + epsilon),  # hapax legomena
            text.count('"') / (char_count + epsilon),  # quotation ratio
        ]
        
        return features

class SimpleStylometryClassifier:
    """Improved stylometry classifier with reduced overconfidence"""
    
    def __init__(self, model_type='logistic', max_features=2000, use_calibration=True):
        self.model_type = model_type
        self.max_features = max_features
        self.use_calibration = use_calibration
        
        # Components
        self.stylometry_extractor = StylometryFeatures()
        self.text_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=3,  # Ignore rare terms
            max_df=0.95  # Ignore very common terms
        )
        self.scaler = StandardScaler()
        
        # Model with regularization
        if model_type == 'logistic':
            # Add strong regularization to reduce overconfidence
            base_model = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=0.1,  # Strong L2 regularization
                class_weight='balanced'  # Handle class imbalance
            )
        elif model_type == 'svm':
            base_model = SVC(
                probability=True, 
                random_state=42, 
                C=0.1,  # Strong regularization
                class_weight='balanced'
            )
        elif model_type == 'rf':
            base_model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,  # Prevent overfitting
                min_samples_split=10,  # Prevent overfitting
                class_weight='balanced'
            )
        
        # Use calibration to improve probability estimates
        if use_calibration:
            self.model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
        else:
            self.model = base_model
    
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
    """Complete training pipeline with overconfidence fixes"""
    
    # Load data
    texts, labels = load_jsonl_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    print(f"\n📊 Data split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train classifier with regularization and calibration
    classifier = SimpleStylometryClassifier(
        model_type='logistic', 
        max_features=2000,  # Reduced to prevent overfitting
        use_calibration=True  # Calibrate probabilities
    )
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
    print(f"Gap (overfitting check): {train_acc - test_acc:.4f}")
    
    # Analyze confidence distribution
    max_probs = np.max(test_probs, axis=1)
    print(f"\nConfidence Analysis:")
    print(f"Mean max probability: {np.mean(max_probs):.4f}")
    print(f"Std max probability: {np.std(max_probs):.4f}")
    print(f"% predictions >0.9 confidence: {np.mean(max_probs > 0.9)*100:.1f}%")
    print(f"% predictions >0.99 confidence: {np.mean(max_probs > 0.99)*100:.1f}%")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, test_pred))
    
    # Cross-validation for robustness check
    cv_scores = cross_val_score(classifier.model, 
                               classifier.scaler.transform(
                                   np.hstack([
                                       classifier.stylometry_extractor.transform(X_train),
                                       classifier.text_vectorizer.transform(X_train).toarray()
                                   ])
                               ), 
                               y_train, cv=5, scoring='accuracy')
    print(f"\n🔄 Cross-validation scores: {cv_scores}")
    print(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
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