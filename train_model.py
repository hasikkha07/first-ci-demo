
import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Mushroom dataset
data = pd.read_csv("mushrooms.csv")

# Separate input features and target
X = data.drop("class", axis=1)
y = data["class"]

# Convert categorical features into numbers
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         X.columns.tolist())
    ]
)

# Create the machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))

# Save the trained model and metrics
joblib.dump(model, "mushroom_model.pkl")

with open("metrics.json", "w") as file:
    json.dump({"accuracy": accuracy}, file, indent=4)

print("Model and metrics saved successfully!")
