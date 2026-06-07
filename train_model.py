import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("gender_dataset.csv")

# Features and labels
X = data["Name"]
y = data["Gender"]

# Convert names into character patterns
vectorizer = CountVectorizer(
    analyzer='char',
    ngram_range=(2, 3)
)

X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_vec, y)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")
