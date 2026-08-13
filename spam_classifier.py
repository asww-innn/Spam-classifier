"""
SMS Spam Classifier
--------------------
A beginner-friendly machine learning project that classifies SMS messages
as "spam" or "ham" (not spam) using scikit-learn.

Pipeline:
1. Load and clean the dataset
2. Convert text messages into numerical features (TF-IDF)
3. Train a Naive Bayes classifier
4. Evaluate accuracy, precision, recall
5. Test it on new, made-up messages

Run this after installing dependencies:
    pip install pandas scikit-learn

Usage:
    python spam_classifier.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------------
# STEP 1: Load the data
# ---------------------------------------------------------------
# Dataset: SMS Spam Collection (public dataset, ~5,500 labelled messages)
# Download it from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Save the file as "spam.csv" in the same folder as this script.

df = pd.read_csv("spam.csv", encoding="latin-1")

# The raw file has some extra empty columns and short column names — clean it up
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

print(f"Loaded {len(df)} messages")
print(df["label"].value_counts())
print()

# ---------------------------------------------------------------
# STEP 2: Split into training and testing sets
# ---------------------------------------------------------------
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------
# STEP 3: Convert text into numbers (TF-IDF)
# ---------------------------------------------------------------
# Machine learning models can't read raw text — TF-IDF turns each message
# into a vector of numbers based on which words appear and how important
# (rare/frequent) each word is across the whole dataset.
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------------------------------------------
# STEP 4: Train the model
# ---------------------------------------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ---------------------------------------------------------------
# STEP 5: Evaluate performance
# ---------------------------------------------------------------
y_pred = model.predict(X_test_vec)

print("=== Model Performance ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {precision_score(y_test, y_pred, pos_label='spam'):.2%}")
print(f"Recall:    {recall_score(y_test, y_pred, pos_label='spam'):.2%}")
print(f"F1 Score:  {f1_score(y_test, y_pred, pos_label='spam'):.2%}")
print()
print("Confusion Matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=["ham", "spam"]),
    index=["actual: ham", "actual: spam"],
    columns=["predicted: ham", "predicted: spam"],
))
print()

# ---------------------------------------------------------------
# STEP 6: Try it on your own messages
# ---------------------------------------------------------------
def predict_message(message: str) -> str:
    vec = vectorizer.transform([message])
    return model.predict(vec)[0]

sample_messages = [
    "Congratulations! You've WON a free iPhone. Click here to claim now!!!",
    "Hey, are we still on for lunch tomorrow?",
    "URGENT: Your account will be suspended. Verify your details immediately.",
    "Can you send me the notes from today's class?",
]

print("=== Testing on new messages ===")
for msg in sample_messages:
    result = predict_message(msg)
    print(f"[{result.upper():5}] {msg}")
