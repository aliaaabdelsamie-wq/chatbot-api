import pandas as pd
import joblib
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
#import sklearn
#print(sklearn.__version__)

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'(.)\1+', r'\1', text)

    text = re.sub(r'[أإآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[ؤئ]', 'ء', text)

    text = re.sub(r'[\u064B-\u0652]', '', text)

    text = re.sub(r'[^\w\s]', ' ', text)

    return text.strip()

df = pd.read_csv("Intents.csv")

df = df.dropna(subset=["message", "intent", "type", "emotion"])
df["combined_labels"] = df.apply(lambda row: [str(row['intent']).strip(),str(row['type']).strip(),str(row['emotion']).strip()], axis=1)

df["message_clean"] = df["message"].apply(clean_text)

#df["intents"] = df["intent"].fillna("").apply(lambda x: x.split(";") if x != "" else [])

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Data loaded: {len(df)}")

X = df["message_clean"]

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["combined_labels"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

features = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1,2), max_features=3000)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=5000))
])

base_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    solver="lbfgs"
)

model = Pipeline([
    ("features", features),
    ("clf", OneVsRestClassifier(base_model))
])

print("Training...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=mlb.classes_))

joblib.dump(model, "multi_intent_model.pkl")
joblib.dump(mlb, "mlb.pkl")

print("Saved!")

def predict(text):
    cleaned = clean_text(text)

    probs = model.predict_proba([cleaned])[0]

    max_index = np.argmax(probs)
    max_prob = probs[max_index]
    if max_prob < 0.6:
        print("\nInput:", text)
        print("Output: unknown (no confident intent)")
        return {}

    result = {}

    for i, label in enumerate(mlb.classes_):
        if probs[i] > 0.6:
            result[label] = float(probs[i])

    print("\nInput:", text)
    print("Output:", result)

    return result

# test
#predict("how can i book session")