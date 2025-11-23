import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# فقط بار اول نیاز داری دانلودش کنی
nltk.download("stopwords")

# گرفتن لیست stopwords انگلیسی
stop_words = stopwords.words("english")

# تابع ساده برای تمیز کردن هر پاراگراف
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # حذف علامت‌های نگارشی
    words = text.split()                  # تبدیل جمله به لیست کلمات
    words = [w for w in words if w not in stop_words]  # حذف stopwords
    return " ".join(words)                # تبدیل دوباره لیست به متن

# چند متن نمونه (می‌تونی عوضشون کنی)
paragraphs = [
    "This is a simple paragraph to test vectorizer.",
    "Vectorizer is a tool to convert text to numbers!",
    "Numbers are useful in machine learning."
]

# تمیز کردن کل پاراگراف‌ها
cleaned = [clean_text(p) for p in paragraphs]
print("Cleaned paragraphs:")
print(cleaned)
print("-" * 50)

# تبدیل متن‌های تمیزشده به بردار
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned)

print("Features (unique words):")
print(vectorizer.get_feature_names_out())
print("-" * 50)

print("Count matrix:")
print(X.toarray())
