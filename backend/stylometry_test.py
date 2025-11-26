# STEP 1: Open your stylometry_classifier.py file located at:
# E:\capstone\capstone\Gen-AI-Authorship-Verification\authorship_verification\authorship_verification\src\models\stylometry_classifier.py

# STEP 2: At the very top of the file (after any existing imports), add this line:
import re

# STEP 3: Find the _get_features method (around line 91) and look for this line:
# sentences = sent_tokenize(text)

# STEP 4: Replace it with these lines:
try:
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
except:
    # Fallback to regex if NLTK fails
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [text]

# STEP 5: Also find any other NLTK-related imports and wrap them in try-except blocks

# Here's what the top of your stylometry_classifier.py should look like:
import re  # ADD THIS LINE
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import json

# Wrap NLTK imports in try-except
try:
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK not available, using fallback methods")

# Add these helper functions to your stylometry_classifier.py:
def safe_sent_tokenize(text):
    """Safe sentence tokenization with fallback"""
    if NLTK_AVAILABLE:
        try:
            return sent_tokenize(text)
        except:
            pass
    # Regex fallback
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences if sentences else [text]

def safe_word_tokenize(text):
    """Safe word tokenization with fallback"""
    if NLTK_AVAILABLE:
        try:
            return word_tokenize(text.lower())
        except:
            pass
    # Regex fallback
    return re.findall(r'\b\w+\b', text.lower())

def safe_get_stopwords():
    """Get stopwords with fallback"""
    if NLTK_AVAILABLE:
        try:
            return set(stopwords.words('english'))
        except:
            pass
    # Basic stopwords fallback
    return {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
        'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 
        'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
        'further', 'then', 'once'
    }

# STEP 6: In your _get_features method, replace NLTK calls with safe versions:
# OLD: sentences = sent_tokenize(text)
# NEW: sentences = safe_sent_tokenize(text)

# OLD: words = word_tokenize(text)  
# NEW: words = safe_word_tokenize(text)

# OLD: stop_words = set(stopwords.words('english'))
# NEW: stop_words = safe_get_stopwords()