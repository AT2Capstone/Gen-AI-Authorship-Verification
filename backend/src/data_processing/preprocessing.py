import re
import unicodedata
from typing import List
import nltk
class TextPreprocessor:
    def __init__(self, preserve_style: bool = True):
        self.preserve_style = preserve_style
        # Download minimal punkt if not present (safe to call)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def basic_clean(self, text: str) -> str:
        if text is None:
            return ''
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Preserve paragraph breaks but collapse repeated spaces
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        if not self.preserve_style:
            # remove non-word punctuation except basic sentence punctuation
            text = re.sub(r'[^\w\s\.,!\?;:\'"()\-]', '', text)
        return text

    def normalize_unicode(self, text: str) -> str:
        if text is None:
            return ''
        return unicodedata.normalize('NFKD', text)

    def preprocess(self, text: str) -> str:
        t = self.basic_clean(text)
        t = self.normalize_unicode(t)
        return t

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.preprocess(t) for t in texts]
