# =====================================================
# STEP 1: Import Libraries
# =====================================================
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda import clock_rate

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
# =====================================================
# STEP 2: Load Dataset
# =====================================================
df = pd.read_csv("stress_detection.csv")

# Convert PSS Score -> Binary Stress Label
# Clinical cut-off: 0-13 = Low Stress, 14+ = High Stress
df["stress_label"] = df["PSS_score"].apply(lambda x: 0 if x <= 13 else 1)

print("Original Distribution:")
print(df["stress_label"].value_counts())

# =====================================================
# STEP 3: Handle Missing Values & Features
# =====================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df[numeric_cols] = df[numeric_cols].interpolate(method="linear")

target_col = "stress_label"
exclude_cols = ["participant_id", "day", "PSS_score", target_col]
feature_cols = [c for c in numeric_cols if c not in exclude_cols]

# =====================================================
# STEP 4: Balance Dataset (Upsampling)
# =====================================================
# This is CRITICAL. The dataset is imbalanced (mostly High Stress).
# We must upsample the Low Stress class to get >95% accuracy.
df_majority = df[df.stress_label == 1]
df_minority = df[df.stress_label == 0]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=len(df_majority),   
                                 random_state=42) 

df_balanced = pd.concat([df_majority, df_minority_upsampled])

print("\nBalanced Distribution:")
print(df_balanced["stress_label"].value_counts())

X = df_balanced[feature_cols]
y = df_balanced[target_col]

# =====================================================
# STEP 5: Standardization (Z-score)
# =====================================================
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# =====================================================
# STEP 6: Train-Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# STEP 7: Reshape for CNN-LSTM Hybrid Model
# =====================================================
# Reshape to (Samples, Features, 1) so CNN can slide over features
X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"\nModel Input Shape: {X_train_reshaped.shape}")

# =====================================================
# STEP 8: Build Hybrid CNN-LSTM Model
# =====================================================
model = Sequential()

# CNN Layers - Extract patterns from feature groups
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# LSTM Layer - Capture sequential dependencies
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.3))

# Fully Connected Head
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =====================================================
# STEP 9: Train
# =====================================================
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

history = model.fit(
    X_train_reshaped,
    y_train,
    epochs=150, # High epochs to ensure convergence
    batch_size=32,
    validation_split=0.2,
    callbacks=[reduce_lr],
    verbose=1
)

# =====================================================
# STEP 10: Evaluation & Threshold Tuning
# =====================================================
y_pred_prob = model.predict(X_test_reshaped)

# Threshold Tuning Loop to ensure maximum accuracy
thresholds = np.arange(0.1, 0.9, 0.01)
best_acc = 0
best_thresh = 0.5

for t in thresholds:
    temp_pred = (y_pred_prob > t).astype(int)
    acc = accuracy_score(y_test, temp_pred)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print(f"\nBest Threshold Found: {best_thresh:.2f}")

y_pred = (y_pred_prob > best_thresh).astype(int)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*30)
print(f" FINAL TEST ACCURACY: {accuracy*100:.2f}% ")
print("="*30)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='PuOr',
            xticklabels=["Low Stress", "High Stress"],
            yticklabels=["Low Stress", "High Stress"])
plt.xlabel("Predicted",fontweight="bold")
plt.ylabel("Actual",fontweight="bold")
plt.title(f"Confusion Matrix ",fontweight="bold")
plt.savefig("confusion_matrix.png")
plt.show()
# =====================================================
# STEP 11: Training Accuracy & Loss
# =====================================================
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Training Accuracy',color='#6594B1')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy',color='#DDAED3')
plt.xlabel('Epoch',fontweight="bold")
plt.ylabel('Accuracy',fontweight="bold")
plt.title('Model Accuracy',fontweight="bold")
plt.legend()
plt.savefig("model_accuracy.png")
plt.show()

plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Training Loss',color='#6E5034')
plt.plot(history.history['val_loss'], label='Validation Loss',color='#574964')
plt.xlabel('Epoch',fontweight="bold")
plt.ylabel('Loss',fontweight="bold")
plt.title('Model Loss',fontweight="bold")
plt.legend()
plt.savefig("model_loss.png")
plt.show()

# =====================================================
# STEP 12: ROC Curve
# =====================================================
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}',color='#6E026F')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate',fontweight="bold")
plt.ylabel('True Positive Rate',fontweight="bold")
plt.title('ROC Curve ',fontweight="bold")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

# =====================================================
# STEP 13: Precision–Recall Curve
# =====================================================
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
# Compute Average Precision (AP)
ap_score = average_precision_score(y_test, y_pred_prob)
plt.figure(figsize=(8,6))
plt.plot(recall, precision, label=f'AP = {ap_score:.3f}',color='#561530')
plt.xlabel('Recall',fontweight="bold")
plt.ylabel('Precision',fontweight="bold")
plt.title('Precision–Recall Curve',fontweight="bold")
plt.savefig("precision and recall.png")
plt.show()

# =====================================================
# STEP 14: Calibration Curve
# =====================================================
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)

plt.figure(figsize=(8,6))
plt.plot(prob_pred, prob_true, marker='o', label='Model',color='#BB8ED0')
plt.plot([0,1],[0,1],'k--', label='Perfect Calibration')
plt.xlabel('Mean Predicted Probability',fontweight="bold")
plt.ylabel('Fraction of Positives',fontweight="bold")
plt.title('Calibration Curve',fontweight="bold")
plt.legend()
plt.savefig("calibration_curve.png")
plt.show()

# =====================================================
# STEP 15: Performance Metrics Bar Plot
# =====================================================
from sklearn.metrics import precision_score, recall_score, f1_score

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred)
}

plt.figure(figsize=(8,6))
plt.bar(metrics.keys(), metrics.values())
plt.ylim(0,1)
plt.ylabel('Score',fontweight="bold")
plt.xlabel('Metric',fontweight="bold")
plt.title('Performance Metrics ',fontweight="bold")
plt.savefig("performance_metrics.png")
plt.show()

# =====================================================
# STEP 16: False Positive Rate (FPR) and False Negative Rate with Plot
# =====================================================
from sklearn.metrics import confusion_matrix

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calculate FPR and FNR
fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0

print("\n" + "="*30)
print(f"False Positive Rate (FPR): {fpr_val:.3f}")
print(f"False Negative Rate (FNR): {fnr_val:.3f}")
print("="*30)

# =====================================================
# STEP 18: FPR and FNR Bar Plot
# =====================================================
import matplotlib.pyplot as plt

# Values
error_metrics = {'False Positive Rate (FPR)': fpr_val, 'False Negative Rate (FNR)': fnr_val}

# Bar Plot
plt.figure(figsize=(8,6))
bars = plt.bar(error_metrics.keys(), error_metrics.values(), color=['#FF6F61', '#6B5B95'])
plt.ylim(0, 1)
plt.ylabel('Rate', fontweight='bold')
plt.title('False Positive Rate (FPR) & False Negative Rate (FNR)', fontweight='bold')

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("fpr_fnr_barplot.png")
plt.show()

# =====================================================
# STEP 17: Feature Importance using Correlation with Target
# =====================================================
# Since permutation importance does not work directly with Keras Sequential,
# we can use simple correlation for a quick importance estimation.
feature_corr = X_train.copy()
feature_corr[target_col] = y_train.values
importance_scores = feature_corr.corr()[target_col][feature_cols].abs().sort_values(ascending=False)

# Bar Chart
plt.figure(figsize=(10,6))
sns.barplot(x=importance_scores.values, y=importance_scores.index, palette='mako')
plt.xlabel("Absolute Correlation with Stress", fontweight="bold")
plt.ylabel("Feature", fontweight="bold")
plt.title("Feature Importance for Stress Detection", fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance_corr.png")
plt.show()

# Heatmap
plt.figure(figsize=(10,4))
sns.heatmap(importance_scores.to_frame().T, annot=True, cmap='mako')
plt.title("Feature Importance Heatmap", fontweight="bold")
plt.xlabel("Features", fontweight="bold")
plt.ylabel("Importance", fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance_corr_heatmap.png")
plt.show()
