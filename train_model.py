import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("gender_dataset.csv")

# Features and labels
X = data["Name"]
y = data["Gender"]

# Convert names to numerical features
vectorizer = CountVectorizer(analyzer='char')

X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")