# main.py - Bankruptcy Prediction with Extended Models and Graphs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("data/data.csv")
print("Dataset Shape:", df.shape)
print("First 5 rows:\n", df.head())

X = df.drop("Bankrupt?", axis=1)
y = df["Bankrupt?"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# Define Models
# ==============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss'),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Perceptron": Perceptron(),
    "AdaBoost": AdaBoostClassifier(),
    "Neural Network (MLP)": MLPClassifier(max_iter=500)
}

results = {}

# ==============================
# Train & Evaluate
# ==============================
for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
    acc = model.score(X_test, y_test)
    
    results[name] = {"Accuracy": acc, "ROC-AUC": roc}

# Convert to DataFrame
results_df = pd.DataFrame(results).T
print("\n=== Model Performance Summary ===")
print(results_df)

# Mark Suitable / Not Suitable
results_df["Suitability"] = results_df.apply(
    lambda row: "✅ Suitable" if row["Accuracy"] > 0.85 and row["ROC-AUC"] > 0.80 else "❌ Not Suitable",
    axis=1
)
print("\n=== Suitability Check ===")
print(results_df)

# ==============================
# Visualization
# ==============================

# --- Performance Comparison (Accuracy & ROC-AUC) ---
fig, ax = plt.subplots(figsize=(12,6))
results_df[["Accuracy", "ROC-AUC"]].plot(
    kind="bar", ax=ax, color=["#1abc9c", "#9b59b6"], edgecolor="black"
)
plt.title("Model Performance Comparison", fontsize=14, fontweight="bold")
plt.ylabel("Score")
plt.xticks(rotation=30, ha="right")
plt.ylim(0,1.05)

# Add value labels
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=8, color="black", rotation=90)

plt.legend()
plt.show()

# --- Suitability Graph ---
fig, ax = plt.subplots(figsize=(12,6))
colors = ["green" if s == "✅ Suitable" else "red" for s in results_df["Suitability"]]
bars = ax.bar(results_df.index, results_df["Accuracy"], color=colors, alpha=0.8, edgecolor="black")
plt.title("Model Suitability (Green = Suitable, Red = Not Suitable)", fontsize=14, fontweight="bold")
plt.ylabel("Accuracy")
plt.xticks(rotation=30, ha="right")

# Add labels ✅ ❌
for bar, label in zip(bars, results_df["Suitability"]):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, label, ha='center', fontsize=9, fontweight="bold")

plt.ylim(0,1.05)
plt.show()

# --- ROC Curves ---
plt.figure(figsize=(10,7))
colors = [
    "blue","orange","green","red","purple","brown","pink","gray","cyan","magenta",
    "lime","gold","darkblue"
]
for (name, model), c in zip(models.items(), colors):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, color=c, linewidth=2, label=name)

plt.plot([0,1],[0,1],'k--', linewidth=1)
plt.title("ROC Curves of Models", fontsize=14, fontweight="bold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()