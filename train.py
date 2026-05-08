import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
df = pd.read_csv("spam.csv", encoding='latin-1')
print("Columns found:")
print(df.columns)
df = df.iloc[:, :2]
df.columns = ['label', 'message']
df['message'] = df['message'].fillna('')
df['label'] = df['label'].astype(str).str.lower()
df = df[df['label'].isin(['spam', 'ham'])]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)
model = MultinomialNB()
model.fit(X_vectorized, y)
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("Model trained successfully!")