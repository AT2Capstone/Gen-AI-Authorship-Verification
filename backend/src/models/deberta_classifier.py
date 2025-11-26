# File: src/models/deberta_classifier.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DebertaV2Tokenizer, 
    DebertaV2ForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json
import os
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_jsonl_data(filepath):
    """Load data from JSONL file"""
    texts = []
    labels = []
    
    print(f"Loading data from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    texts.append(data['text'])
                    labels.append(data['label'])
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num}: {e}")
                except KeyError as e:
                    print(f"Warning: Missing field in line {line_num}: {e}")
    
    print(f"Loaded {len(texts)} samples")
    print(f"Class distribution: {Counter(labels)}")
    
    return texts, labels

class AIDetectionDataset(Dataset):
    """Dataset class for AI detection task"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DebertaAIClassifier:
    """DeBERTa classifier for AI text detection"""
    
    def __init__(self, 
                 model_name='microsoft/deberta-base',  # Changed from deberta-v3-base
                 max_length=512,
                 num_classes=2,
                 device=None):
        
        self.model_name = model_name
        self.max_length = max_length
        self.num_classes = num_classes
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model with error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except Exception as e:
            print(f"Failed to load fast tokenizer, trying slow tokenizer: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                problem_type="single_label_classification"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.trained = False
    
    def prepare_datasets(self, texts, labels, test_size=0.2, val_size=0.1):
        """Prepare train, validation, and test datasets"""
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"Data splits:")
        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples") 
        print(f"Test: {len(X_test)} samples")
        
        # Create datasets
        train_dataset = AIDetectionDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = AIDetectionDataset(X_val, y_val, self.tokenizer, self.max_length)
        test_dataset = AIDetectionDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        return train_dataset, val_dataset, test_dataset, (X_test, y_test)
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, 
              train_dataset, 
              val_dataset, 
              output_dir='models/saved/deberta',
              epochs=3,
              batch_size=8,
              learning_rate=2e-5,
              warmup_steps=500,
              weight_decay=0.01,
              save_steps=1000,
              eval_steps=500,
              logging_steps=100):
        """Train the DeBERTa model"""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=f'{output_dir}/logs',
            logging_steps=logging_steps,
            eval_strategy="steps",  # Changed from evaluation_strategy
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            seed=42,
            dataloader_num_workers=0,  # Prevent multiprocessing issues
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        print("Starting training...")
        
        # Train the model
        trainer.train()
        
        # Save the final model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        self.trained = True
        
        return trainer
    
    def predict_proba(self, texts, batch_size=8):
        """Get probability predictions for ensemble integration"""
        
        if not self.trained and not os.path.exists('models/saved/deberta'):
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        # Create dataset for prediction
        dummy_labels = [0] * len(texts)  # Dummy labels for prediction
        dataset = AIDetectionDataset(texts, dummy_labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_probs = []
        
        print(f"Generating predictions for {len(texts)} samples...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Apply softmax to get probabilities
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())
        
        probabilities = np.array(all_probs)
        print(f"Generated probability matrix shape: {probabilities.shape}")
        
        return probabilities
    
    def predict(self, texts, batch_size=8):
        """Get class predictions"""
        probabilities = self.predict_proba(texts, batch_size)
        return np.argmax(probabilities, axis=1)
    
    def evaluate(self, test_texts, test_labels, batch_size=8):
        """Evaluate model on test set"""
        
        predictions = self.predict(test_texts, batch_size)
        probabilities = self.predict_proba(test_texts, batch_size)
        
        accuracy = accuracy_score(test_labels, predictions)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(test_labels, predictions))
        
        return accuracy, probabilities
    
    def save_model(self, save_path):
        """Save the trained model"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, model_path, device=None):
        """Load a trained model"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        print(f"Loading model from {model_path}")
        
        # Create instance
        instance = cls.__new__(cls)
        
        # Set device
        if device is None:
            instance.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            instance.device = device
        
        # Load tokenizer and model
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        instance.model.to(instance.device)
        
        # Set other attributes
        instance.max_length = 512
        instance.num_classes = 2
        instance.trained = True
        
        print(f"Model loaded successfully on {instance.device}")
        
        return instance

def train_deberta_classifier(data_path, 
                            model_save_path='models/saved/deberta',
                            epochs=3,
                            batch_size=16,  # Increased for lighter models
                            max_length=256,  # Reduced for speed
                            model_name='distilbert-base-uncased'):  # Default to lighter model
    """Complete training pipeline for transformer classifier"""
    
    # Load data
    texts, labels = load_jsonl_data(data_path)
    
    # Initialize classifier
    classifier = DebertaAIClassifier(
        model_name=model_name,
        max_length=max_length
    )
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset, (test_texts, test_labels) = classifier.prepare_datasets(
        texts, labels
    )
    
    # Train model
    trainer = classifier.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=model_save_path,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy, test_probabilities = classifier.evaluate(test_texts, test_labels)
    
    return classifier, test_accuracy, test_probabilities

def get_ensemble_probabilities(model_path, texts, batch_size=8):
    """Load DeBERTa model and get probabilities for ensemble"""
    classifier = DebertaAIClassifier.load_model(model_path)
    probabilities = classifier.predict_proba(texts, batch_size)
    
    print(f"Generated DeBERTa probabilities for ensemble integration")
    
    return probabilities

# Main execution
if __name__ == "__main__":
    # Configuration
    train_data_path = 'data/train/train.jsonl'
    model_save_path = 'models/saved/tinybert'
    
    print("TinyBERT AI Detection Classifier")
    print("=" * 50)
    
    # Check if model already exists
    if os.path.exists(model_save_path) and os.path.exists(os.path.join(model_save_path, 'pytorch_model.bin')):
        print("Found existing TinyBERT model, loading...")
        classifier = DebertaAIClassifier.load_model(model_save_path)
        
        # Test with sample
        sample_texts = ["Teamwork is the foundation of success in many areas of life. When people work together, they combine different strengths, ideas, and skills to achieve a common goal. A strong team values communication, respect, and trust, allowing members to rely on one another. Teamwork also teaches patience and cooperation, since challenges are easier to overcome when shared. Whether in school, sports, or the workplace, collaboration often leads to better results than individual effort alone. By supporting and motivating each other, a team creates an environment where everyone can grow, making teamwork an essential part of progress and achievement."]
        probs = classifier.predict_proba(sample_texts)
        print(f"Sample probabilities: {probs[0]}")
        
    else:
        print("Training new TinyBERT model...")
        classifier, accuracy, test_probs = train_deberta_classifier(
            train_data_path, 
            model_save_path,
            epochs=2,  # Reduced for speed
            batch_size=32,  # Larger batch for efficiency
            max_length=256,  # Shorter sequences
            model_name='huawei-noah/TinyBERT_General_4L_312D'  # Fastest model
        )
        print(f"\nTraining complete! Test accuracy: {accuracy:.4f}")
    
    print(f"\nTinyBERT model ready for ensemble integration!")
    print(f"Use: probabilities = classifier.predict_proba(your_texts)")