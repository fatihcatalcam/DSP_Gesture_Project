import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Aynı data klasörü
DATA_DIR = r"C:\Users\Fatih\DSP_Gesture_Project\data"

# 1) Low pass filter
def low_pass_filter(signal, alpha=0.85):
    filtered = []
    prev = signal.iloc[0]
    for x in signal:
        y = alpha * prev + (1 - alpha) * x
        filtered.append(y)
        prev = y
    return np.array(filtered)

# 2) Feature extraction - main.py ile aynı mantık
def extract_features(df):
    ax = low_pass_filter(df.iloc[:, 1])
    ay = low_pass_filter(df.iloc[:, 2])
    az = low_pass_filter(df.iloc[:, 3])

    magnitude = np.sqrt(ax**2 + ay**2 + az**2)

    features = {
        "mean":     np.mean(magnitude),
        "variance": np.var(magnitude),
        "max_peak": np.max(magnitude),
        "min_peak": np.min(magnitude),
        "energy":   np.sum(magnitude**2),
        "x_peak":   np.max(ax),
        "x_low":    np.min(ax),
        "z_peak":   np.max(az),
        "z_low":    np.min(az),
    }
    return features

# 3) Dataset i oluştur - her csv dosyası bir örnek
X = []
y = []
files = []

print("Data klasörü:", DATA_DIR)
print("Dosyalar:", os.listdir(DATA_DIR))

for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        full_path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(full_path)

        feats = extract_features(df)
        X.append([
            feats["mean"],
            feats["variance"],
            feats["max_peak"],
            feats["min_peak"],
            feats["energy"],
            feats["x_peak"],
            feats["x_low"],
            feats["z_peak"],
            feats["z_low"],
        ])

        # Label i dosya isminden al
        fname = file.lower()
        if "left" in fname:
            label = "LEFT"
        elif "right" in fname:
            label = "RIGHT"
        elif "up" in fname:
            label = "UP"
        else:
            label = "UNKNOWN"

        y.append(label)
        files.append(file)

X = np.array(X)
y = np.array(y)

print("\nToplam örnek sayısı:", X.shape[0])
print("Özellik sayısı:", X.shape[1])

# 4) Train test split
X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
    X, y, files, test_size=0.3, random_state=42, stratify=y
)

print("\nTrain örnek:", X_train.shape[0])
print("Test örnek:", X_test.shape[0])

# 5) Özellikleri ölçekle (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6) KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

# 7) Sonuçlar
print("\n===== MACHINE LEARNING RESULTS (KNN) =====")
print("Test accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["LEFT","RIGHT","UP"]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=["LEFT","RIGHT","UP"]))

# 8) Hangi dosya nasıl sınıflanmış görmek için
print("\nDetaylı test örnekleri:")
for fname, true_label, pred_label in zip(files_test, y_test, y_pred):
    print(f"{fname:12s}  true={true_label:5s}  pred={pred_label:5s}")
