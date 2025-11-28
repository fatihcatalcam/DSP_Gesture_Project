import os
import numpy as np
import pandas as pd

# ==========================
# 1) Low Pass Filter
# ==========================

def low_pass_filter(signal, alpha=0.85):
    filtered = []
    prev = signal.iloc[0]
    for x in signal:
        y = alpha * prev + (1 - alpha) * x
        filtered.append(y)
        prev = y
    return np.array(filtered)

# ==========================
# 2) Feature Extraction
# ==========================

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

# ==========================
# 3) Classification Rules (DÜZELTİLMİŞ)
# ==========================

def classify(features):
    x_peak = features["x_peak"]
    x_low  = features["x_low"]
    z_peak = features["z_peak"]

    # ❗ Önce UP — Z baskın ise UP
    if z_peak > 6 and abs(x_peak) < 25 and abs(x_low) < 25:
        return "UP"

    # ❗ LEFT — negatif ivme pozitif ivmeden daha baskınsa
    if abs(x_low) > abs(x_peak) and x_low < -18:
        return "LEFT"

    # ❗ RIGHT — pozitif ivme negatiften daha büyükse
    if abs(x_peak) > abs(x_low) and x_peak > 18:
        return "RIGHT"

    return "UNKNOWN"



# ==========================
# 4) Tüm CSV'leri data klasöründen oku
# ==========================

DATA_DIR = r"C:\Users\Fatih\DSP_Gesture_Project\data"

results = []

print("📂 Data klasörü:", DATA_DIR)
print("📄 Dosyalar:", os.listdir(DATA_DIR))

for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        full_path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(full_path)

        feats = extract_features(df)
        gesture = classify(feats)

        results.append([
            file,
            gesture,
            feats["x_peak"],
            feats["x_low"],
            feats["z_peak"],
            feats["z_low"],
        ])

        print(f"📌 {file}  ➜  {gesture}")

# Sonuç tablosu
df_results = pd.DataFrame(
    results,
    columns=["file", "prediction", "x_peak", "x_low", "z_peak", "z_low"]
)

print("\n============================")
print(" FINAL CLASSIFICATION RESULT ")
print("============================")
print(df_results)

# Ana proje klasörüne kaydet
out_path = r"C:\Users\Fatih\DSP_Gesture_Project\gesture_results.csv"
df_results.to_csv(out_path, index=False)
print(f"\n🎉 SONUÇLAR '{out_path}' DOSYASINA KAYDEDİLDİ")
