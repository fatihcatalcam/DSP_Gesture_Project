import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.decomposition import PCA

# ==============================
# 0) Data klasörü
# ==============================

DATA_DIR = r"C:\Users\Fatih\DSP_Gesture_Project\data"

print("Data klasoru:", DATA_DIR)
print("Dosyalar:", os.listdir(DATA_DIR))

# ==============================
# 1) Low pass filter
# ==============================

def low_pass_filter(signal, alpha=0.85):
    filtered = []
    prev = signal.iloc[0]
    for x in signal:
        y = alpha * prev + (1 - alpha) * x
        filtered.append(y)
        prev = y
    return np.array(filtered)

# ==============================
# 2) Feature extraction
# ==============================

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

# ==============================
# 3) Dataset olustur
# ==============================

X = []
y = []
files = []

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

print("\nToplam ornek sayisi:", X.shape[0])
print("Ozellik sayisi:", X.shape[1])

# UNKNOWN varsa at
mask_known = y != "UNKNOWN"
X = X[mask_known]
y = y[mask_known]
files = np.array(files)[mask_known]

print("UNKNOWN olmayan ornek sayisi:", X.shape[0])

# ==============================
# 4) Train-test split
# ==============================

X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
    X, y, files, test_size=0.3, random_state=42, stratify=y
)

print("\nTrain ornek:", X_train.shape[0])
print("Test ornek:", X_test.shape[0])

# ==============================
# 5) Scaling
# ==============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 6) Modeller: KNN, SVM, RandomForest
# ==============================

models = {}

models["KNN (k=3)"] = KNeighborsClassifier(n_neighbors=3)

models["SVM (RBF)"] = SVC(
    kernel="rbf",
    C=10.0,
    gamma="scale",
    random_state=42
)

models["RandomForest"] = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

# ==============================
# 7) Her model icin egitim ve degerlendirme
# ==============================

for name, model in models.items():
    print("\n===================================")
    print("MODEL:", name)
    print("===================================")

    # KNN ve SVM scaled veri ile, RF istersen raw da calisir ama tutarlilik icin scaled kullaniyoruz
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", acc)

    cm = confusion_matrix(y_test, y_pred, labels=["LEFT", "RIGHT", "UP"])
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=["LEFT","RIGHT","UP"]))

    print("\nDetayli test ornekleri:")
    for fname, true_label, pred_label in zip(files_test, y_test, y_pred):
        print(f"{fname:12s}  true={true_label:5s}  pred={pred_label:5s}")

    # Confusion matrixi basit bir sekilde gorsellestir ve kaydet
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()
    tick_marks = np.arange(3)
    classes = ["LEFT", "RIGHT", "UP"]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_png = f"confusion_{name.replace(' ','_').replace('(','').replace(')','')}.png"
    plt.savefig(out_png)
    print(f"\nConfusion matrix gorseli kaydedildi: {out_png}\n")
    plt.close()

# ==============================
# 8) PCA ile 2D gorsellestirme
# ==============================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure()
for label, marker in zip(["LEFT","RIGHT","UP"], ["o","s","^"]):
    mask = (y_train == label)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], marker=marker, label=label, alpha=0.7)

plt.title("PCA Projection of Gestures (Train Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig("pca_gesture_train.png")
print("PCA gorseli kaydedildi: pca_gesture_train.png")
plt.close()
