import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier

X = np.load("X_temporal.npy")
y = np.load("y.npy")

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nUnique labels in y.npy:")
print(np.unique(y))

print("\nSamples per label:")

for label in np.unique(y):
    print(
        f"Label {label}: "
        f"{np.sum(y == label)} samples"
    )

class_names = [
    "ab_chagi",
    "ab_dollyo_chogi",
    "anmoki",
    "aremoki",
    "banda_dollyo_chagi",
    "dollyo_chagi",
    "double_chagi",
    "double_dollyo_chagi",
    "lei_lei_chagi",
    "mom_dollyo_chagi",
    "momtong_maki",
    "ner_yo_chagi",
    "olgol_maki",
    "sonnal_maki",
    "tichogi",
    "yop_chogi"
]

keep_classes = [0, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19]

# Keep class names in exactly the same order as the remapped labels.
filtered_class_names = class_names.copy()
mask = np.isin(y, keep_classes)

X = X[mask]
y = y[mask]

class_mapping = {
    old: new
    for new, old in enumerate(keep_classes)
}

y = np.array([
    class_mapping[label]
    for label in y
])

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train/test split
if np.min(np.bincount(y)) < 2:
    raise ValueError("Every class needs at least 2 samples. Add more videos before training.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=16, random_state=42, stratify=y
)

# Feature selection must be fitted on training data only.
importance_model = RandomForestClassifier(
    n_estimators=1000,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)
importance_model.fit(X_train, y_train)
importance = importance_model.feature_importances_
selected_indices = np.argsort(importance)[::-1][:min(75, X_train.shape[1])]

X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

# Evaluation model
eval_model = RandomForestClassifier(
    n_estimators=1000,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)
eval_model.fit(X_train_selected, y_train)
y_pred = eval_model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, labels=np.arange(len(filtered_class_names)),
    target_names=filtered_class_names, zero_division=0
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=np.arange(len(filtered_class_names))))

# Final production model: train on all samples using the selected features.
X_selected = X[:, selected_indices]
final_model = RandomForestClassifier(
    n_estimators=1000,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)
final_model.fit(X_selected, y)

joblib.dump(
    final_model,
    "taekwondo_rf_final.pkl"
)

np.save(
    "selected_features.npy",
    selected_indices
)

with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(
        filtered_class_names,
        f,
        ensure_ascii=False,
        indent=4
    )

model_info = {
    "model": "Random Forest",
    "classes": len(filtered_class_names),
    "samples": int(X.shape[0]),
    "features_before_selection": 360,
    "features_after_selection": int(len(selected_indices)),
    "test_accuracy": float(accuracy),
    "class_names": filtered_class_names,
    "feature_type": "Temporal Features",
    "estimators": 1000,
    "class_weight": "balanced"
}

with open("model_info.json", "w", encoding="utf-8") as f:
    json.dump(
        model_info,
        f,
        ensure_ascii=False,
        indent=4
    )

print("Training completed")
print("Samples:", X.shape[0])
print("Original features:", X.shape[1])
print("Selected features:", X_selected.shape[1])
print("Classes:", len(class_names))
print("Model saved: taekwondo_rf_final.pkl")
print("Features saved: selected_features.npy")
print("Labels saved: class_names.json")
print("Info saved: model_info.json")