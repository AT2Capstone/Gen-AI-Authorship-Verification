from src.models.stylometry_classifier import train_stylometry_classifier, SimpleStylometryClassifier
import os

# Your exact file path
train_file = 'data/train/train.jsonl'  # Make sure this matches your actual file
model_path = 'models/saved/stylometry_classifier.pkl'

print("🎯 Training Stylometry Classifier")

# Check if model exists
if os.path.exists(model_path):
    print("📂 Loading existing model...")
    classifier = SimpleStylometryClassifier.load_model(model_path)
else:
    print("🚀 Training new model...")
    classifier, accuracy, probs = train_stylometry_classifier(train_file, model_path)

# Test ensemble probabilities
sample_texts = ["Recently I received an e-mail that wasn’t meant for me, but was about me. I’d been cc’d by accident. This is one of the darker hazards of electronic communication, Reason No. 697 Why the Internet Is Bad — the dreadful consequence of hitting 'reply all' instead of 'reply' or 'forward.' The context is that I had rented a herd of goats for reasons that aren’t relevant here and had sent out a mass e-mail with photographs of the goats attached to illustrate that a) I had goats, and b) it was good. Most of the responses I received expressed appropriate admiration and envy of my goats, but the message in question was intended not as a response to me but as an aside to some of the recipient’s co-workers, sighing over the kinds of expenditures on which I was frittering away my uncomfortable income. The word 'oof' was used."]
ensemble_probs = classifier.predict_proba(sample_texts)
print(f"✅ Ready for ensemble! Probability shape: {ensemble_probs.shape}")
print(f"Sample probabilities: {ensemble_probs[0]}")