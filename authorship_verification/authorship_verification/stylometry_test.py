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
sample_texts = ["Pollution happens when harmful substances get into the environment and start causing damage to nature and our health. It shows up in many forms—air, water, soil, and even noise. Factories, vehicles, waste, and chemicals are some of the biggest sources. Smoke and gases in the air can cause breathing issues and even drive climate change. Polluted water puts aquatic life at risk and can make drinking water unsafe, while soil pollution makes it harder for plants to grow and affects the food we eat. Beyond harming nature, pollution also makes daily life harder by lowering the quality of the air we breathe and the water we drink. The good news is that small changes like recycling, using cleaner energy, and cutting down on waste really do help. The more we understand pollution and work to reduce it, the better chance we have of keeping the planet healthy for ourselves and for future generations."]
ensemble_probs = classifier.predict_proba(sample_texts)
print(f"✅ Ready for ensemble! Probability shape: {ensemble_probs.shape}")
print(f"Sample probabilities: {ensemble_probs[0]}")