import os
import sys
import json
import pickle
import numpy as np
import logging
from io import StringIO

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# Correct imports for new cleaned project structure
# ---------------------------------------------------------
from src.models.stylometry_classifier import (
    train_stylometry_classifier,
    SimpleStylometryClassifier
)
from src.models.deberta_classifier import DebertaAIClassifier

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class StackingEnsemble:
    """
    Combined Stylometry + TinyBERT/DeBERTa ensemble model.
    Supports:
      - Weighted fusion (default)
      - Meta-classifier stacking
    """

    def __init__(self, meta_classifier=None):
        self.stylometry_classifier = None
        self.deberta_classifier = None
        self.meta_classifier = meta_classifier or LogisticRegression(random_state=42, max_iter=200)
        self.is_trained = False

    # ---------------------------------------------------------
    # Base model loading
    # ---------------------------------------------------------
    def load_base_models(self, stylometry_path, deberta_path, train_file=None):
        """Load stylometry + deberta models."""
        # Stylometry
        if os.path.exists(stylometry_path):
            logger.info("📂 Loading existing Stylometry model...")
            self.stylometry_classifier = SimpleStylometryClassifier.load_model(stylometry_path)
        else:
            if train_file is None:
                raise ValueError("Stylometry model not found and no train_file provided!")
            logger.info("🚀 Training new Stylometry model...")
            self.stylometry_classifier, _, _ = train_stylometry_classifier(train_file, stylometry_path)

        # TinyBERT / DeBERTa
        logger.info("📂 Loading TinyBERT/DeBERTa model...")
        self.deberta_classifier = DebertaAIClassifier.load_model(deberta_path)

    # ---------------------------------------------------------
    # Feature stacking utilities
    # ---------------------------------------------------------
    def _combine_probs(self, stylo_probs, deberta_probs):
        return np.array([np.concatenate([s, d]) for s, d in zip(stylo_probs, deberta_probs)])

    def _get_base_predictions(self, texts):
        stylo_probs = self.stylometry_classifier.predict_proba(texts)
        deberta_probs = self.deberta_classifier.predict_proba(texts)
        return self._combine_probs(stylo_probs, deberta_probs)

    # ---------------------------------------------------------
    # Weighted fusion (default)
    # ---------------------------------------------------------
    def weighted_fusion(self, texts, weight_deberta=0.8, threshold=0.6):
        """
        Weighted fusion with TinyBERT dominance unless confidence is low.
        Returns Nx2 array: [Human_prob, AI_prob]
        """
        stylo_probs = self.stylometry_classifier.predict_proba(texts)
        deberta_probs = self.deberta_classifier.predict_proba(texts)

        fused = []
        for s, d in zip(stylo_probs, deberta_probs):
            d_conf = float(np.max(d))

            if d_conf < threshold:
                w_d, w_s = 0.6, 0.4
            else:
                w_d, w_s = weight_deberta, 1 - weight_deberta

            f = w_d * d + w_s * s
            f = f / (np.sum(f) + 1e-12)
            fused.append(f)

        return np.vstack(fused)

    # ---------------------------------------------------------
    # Meta-classifier training
    # ---------------------------------------------------------
    def train_meta_classifier(self, train_file, use_cv=False, sample_size=None):
        logger.info("🎯 Training meta-classifier...")

        texts, labels = self._load_training_data(train_file)
        labels = np.array(labels)

        if sample_size and len(texts) > sample_size:
            idx = np.random.choice(len(texts), sample_size, replace=False)
            texts = [texts[i] for i in idx]
            labels = labels[idx]

        if use_cv:
            return self._train_with_cv(texts, labels)
        else:
            return self._train_fast(texts, labels)

    def _train_fast(self, texts, labels):
        logger.info(f"📊 Extracting base model predictions for {len(texts)} samples...")
        meta_features = self._get_base_predictions(texts)

        X_train, X_val, y_train, y_val = train_test_split(
            meta_features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        logger.info("Training meta-classifier...")
        self.meta_classifier.fit(X_train, y_train)

        val_pred = self.meta_classifier.predict(X_val)
        train_pred = self.meta_classifier.predict(X_train)

        logger.info(f"Train acc: {accuracy_score(y_train, train_pred):.4f}")
        logger.info(f"Val acc: {accuracy_score(y_val, val_pred):.4f}")
        logger.info("\n" + classification_report(y_val, val_pred, target_names=["Human", "AI"]))

        self._show_feature_importance()
        self.is_trained = True
        return accuracy_score(y_val, val_pred)

    def _train_with_cv(self, texts, labels, cv_folds=5):
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        stylo_oof = np.zeros((len(texts), 2))
        deberta_oof = np.zeros((len(texts), 2))

        for fold, (_, val_idx) in enumerate(skf.split(texts, labels)):
            logger.info(f"  Fold {fold+1}/{cv_folds}...")
            val_texts = [texts[i] for i in val_idx]
            stylo_oof[val_idx] = self.stylometry_classifier.predict_proba(val_texts)
            deberta_oof[val_idx] = self.deberta_classifier.predict_proba(val_texts)

        meta_features = self._combine_probs(stylo_oof, deberta_oof)

        self.meta_classifier.fit(meta_features, labels)
        preds = self.meta_classifier.predict(meta_features)

        acc = accuracy_score(labels, preds)
        logger.info(f"Cross-validated acc: {acc:.4f}")
        logger.info("\n" + classification_report(labels, preds, target_names=["Human", "AI"]))

        self._show_feature_importance()
        self.is_trained = True
        return acc

    # ---------------------------------------------------------
    # Introspection helpers
    # ---------------------------------------------------------
    def _show_feature_importance(self):
        if hasattr(self.meta_classifier, "coef_"):
            coef = self.meta_classifier.coef_[0]
            logger.info(f"Stylometry weights: {coef[:2]}")
            logger.info(f"DeBERTa weights: {coef[2:]}")

    # ---------------------------------------------------------
    # Prediction API
    # ---------------------------------------------------------
    def predict_proba(self, texts, use_weighted=True, **kwargs):
        if use_weighted:
            return self.weighted_fusion(texts, **kwargs)

        if not self.is_trained:
            raise ValueError("Meta-classifier not trained!")
        return self.meta_classifier.predict_proba(self._get_base_predictions(texts))

    def predict(self, texts, use_weighted=True, **kwargs):
        return np.argmax(self.predict_proba(texts, use_weighted=use_weighted, **kwargs), axis=1)

    # ---------------------------------------------------------
    # Save / load ensemble
    # ---------------------------------------------------------
    def save_model(self, save_path):
        data = {
            "meta_classifier": self.meta_classifier,
            "is_trained": self.is_trained
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"💾 Saved ensemble to {save_path}")

    @classmethod
    def load_model(cls, save_path, stylometry_path, deberta_path):
        ensemble = cls()

        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                data = pickle.load(f)
                ensemble.meta_classifier = data.get("meta_classifier", ensemble.meta_classifier)
                ensemble.is_trained = data.get("is_trained", False)
        else:
            logger.warning("⚠ No ensemble.pkl found — loading base models only.")

        ensemble.load_base_models(stylometry_path, deberta_path)
        return ensemble


# ---------------------------------------------------------
# Extra utilities
# ---------------------------------------------------------
def read_texts_from_file(file_path="./sample-input.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return [text] if text else []


def entropy(probs):
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    return -np.sum(probs * np.log(probs))
