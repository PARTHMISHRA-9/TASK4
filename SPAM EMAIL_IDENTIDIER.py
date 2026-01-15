import pandas as pd
import numpy as np

# Import necessary modules from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Keep only relevant columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Map labels: ham -> 0, spam -> 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Separate features and target
X = data['message']
y = data['label']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict on test set
y_pred = model.predict(X_test_vec)

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Test model on sample emails
sample_emails = [
    "Congratulations! You have won a free gift card",
    "Please find the meeting notes attached"
]

sample_vec = vectorizer.transform(sample_emails)
predictions = model.predict(sample_vec)

# Print predictions for sample emails
for email, pred in zip(sample_emails, predictions):
    print(email, "->", "Spam" if pred == 1 else "Not Spam")
