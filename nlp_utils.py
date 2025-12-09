import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)

def clean_text(text: str) -> str:
    """Lowercase, links/non-letters remove, stopwords hatao."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # links hatao
    text = re.sub(r"http\S+", " ", text)
    # sirf alphabets + space
    text = re.sub(r"[^a-z\s]", " ", text)
    # tokens bana ke stopwords hatao
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)
