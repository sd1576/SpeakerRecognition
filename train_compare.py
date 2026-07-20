import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Load the combined feature dataset
df = pd.read_csv("combined_features.csv")

print("Dataset shape:", df.shape)
print(df.head())


# Separate features and speaker labels
X = df.drop(columns=["Speaker"])
y = df["Speaker"]


# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create SVM model
model = SVC(
    kernel="rbf"
)


# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================================
# TRAIN ANN
# ==========================================================
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

print("\n" + "=" * 60)
print("TRAINING ANN")
print("=" * 60)

# Encode string speaker labels to integers for the ANN
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
num_classes = len(le.classes_)

y_train_cat = to_categorical(y_train_encoded, num_classes)
y_test_cat = to_categorical(y_test_encoded, num_classes)

ann_model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

ann_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

ann_model.summary()

history = ann_model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=100,
    batch_size=16,
    verbose=0
)

ann_loss, ann_acc = ann_model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nANN Test Accuracy: {ann_acc:.4f}")

ann_model.save("ann_speaker_model.keras")
print("ANN model saved -> ann_speaker_model.keras")

ann_pred_encoded = np.argmax(ann_model.predict(X_test, verbose=0), axis=1)
ann_pred = le.inverse_transform(ann_pred_encoded)

print("\nANN Classification Report:")
print(classification_report(y_test, ann_pred, zero_division=0))

# ==========================================================
# FINAL COMPARISON
# ==========================================================
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(f"{'Model':<10} | {'Test Accuracy':<15}")
print("-" * 30)
print(f"{'SVM':<10} | {accuracy:.4f}")
print(f"{'ANN':<10} | {ann_acc:.4f}")