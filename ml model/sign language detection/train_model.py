# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv("data/real_signs.csv")

# Map English to Hindi
HINDI_MAP = {
    "hello": "नमस्ते",
    "thank_you": "धन्यवाद",
    "yes": "हाँ",
    "no": "नहीं",
    "love": "प्यार",
    "stop": "रुको"
}
df["label_hi"] = df["label_en"].map(HINDI_MAP)

# Features and labels
X = df.drop(["label_en","label_hi"], axis=1)
y = df["label_en"]

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(X, y_enc)

# Save model and label encoder
pickle.dump(model, open("real_model.pkl","wb"))
pickle.dump(le, open("le.pkl","wb"))
print("Model trained and saved as real_model.pkl and le.pkl")
