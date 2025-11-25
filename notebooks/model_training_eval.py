import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/processed_data_cleaned.csv")
print(f"Dataset Loaded: {df.shape}")

X = df.drop(columns=["addiction_level", "average_playtime"])
y = df["addiction_level"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("\n--- Random Forest Results ---")
print(f"Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, rf_pred, target_names=["Low", "Medium", "High"]))

plt.figure(figsize=(5, 4))
sns.heatmap(
    confusion_matrix(y_test, rf_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"],
)
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("../results/confusion_matrix_rf.png", dpi=300)
plt.close()

feat_imp_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
    ascending=False
)
plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp_rf.values, y=feat_imp_rf.index, palette="crest")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("../results/feature_importance_rf.png", dpi=300)
plt.close()

xgb_model = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=300,
    max_depth=6,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1,
)
xgb_model.fit(X_train_scaled, y_train)

xgb_pred = xgb_model.predict(X_test_scaled)
xgb_acc = accuracy_score(y_test, xgb_pred)

print("\n--- XGBoost Results ---")
print(f"Accuracy: {xgb_acc:.4f}")
print(classification_report(y_test, xgb_pred, target_names=["Low", "Medium", "High"]))

plt.figure(figsize=(5, 4))
sns.heatmap(
    confusion_matrix(y_test, xgb_pred),
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"],
)
plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()
plt.savefig("../results/confusion_matrix_xgb.png", dpi=300)
plt.close()

feat_imp_xgb = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(
    ascending=False
)
plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp_xgb.values, y=feat_imp_xgb.index, palette="mako")
plt.title("Feature Importance - XGBoost")
plt.tight_layout()
plt.savefig("../results/feature_importance_xgb.png", dpi=300)
plt.close()


joblib.dump(rf, "../models/random_forest_model.pkl")
joblib.dump(xgb_model, "../models/xgboost_model.pkl")
print("💾 Models saved to 'models/' directory")

print("\n--- Model Comparison ---")
print(f"Random Forest Accuracy: {rf_acc:.3f}")
print(f"XGBoost Accuracy:       {xgb_acc:.3f}")

best_model = "XGBoost" if xgb_acc > rf_acc else "Random Forest"
print(f"\nBest Performing Model: {best_model}")
print("All results saved in the 'results/' directory.")
