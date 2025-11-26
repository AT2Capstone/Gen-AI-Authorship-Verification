from src.models.deberta_classifier import train_deberta_classifier, DebertaAIClassifier
import os

# Ultra-fast TinyBERT configuration
train_file = 'data/train/train.jsonl'
model_path = 'models/saved/tinybert'

print("Training TinyBERT AI Detection Classifier (Fastest Option)")
print("Expected training time: 2-4 hours on CPU")

# Check if model exists
if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
    print("Loading existing TinyBERT model...")
    classifier = DebertaAIClassifier.load_model(model_path)
else:
    print("Training new TinyBERT model...")
    classifier, accuracy, probs = train_deberta_classifier(
        train_file, 
        model_path,
        epochs=2,  # Fewer epochs for speed
        batch_size=32,  # Large batch for efficiency
        max_length=256,  # Shorter sequences = faster
        model_name='huawei-noah/TinyBERT_General_4L_312D'  # Smallest, fastest model
    )
    print(f"Training complete! Test accuracy: {accuracy:.4f}")

# Test ensemble probabilities
sample_texts = ["This text appears to be AI-generated based on its structure and style."]
tinybert_probs = classifier.predict_proba(sample_texts)
print(f"Ready for ensemble! Probability shape: {tinybert_probs.shape}")
print(f"Sample probabilities: {tinybert_probs[0]}")