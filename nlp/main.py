import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

#nltk.download("stopwords")

stop_words = stopwords.words("english")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

paragraphs = [
    "This is a simple paragraph to test vectorizer.",
    "Vectorizer is a tool to convert text to numbers!",
    "Numbers are useful in machine learning."
]

cleaned = [clean_text(p) for p in paragraphs]
print("Cleaned paragraphs:")
print(cleaned)
print("-" * 50)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned)

print("Features (unique words):")
print(vectorizer.get_feature_names_out())
print("-" * 50)

print("Count matrix:")
print(X.toarray())
