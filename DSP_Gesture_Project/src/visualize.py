import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def low_pass_filter(sig, alpha=0.85):
    out = []
    prev = sig.iloc[0]
    for v in sig:
        y = alpha*prev + (1-alpha)*v
        out.append(y)
        prev = y
    return np.array(out)

# 🔥 Artık absolute path ile çalışıyor
DATA_DIR = r"C:\Users\Fatih\DSP_Gesture_Project\data"

print("📂 Data klasörü:", DATA_DIR)
print("📄 İçerik:", os.listdir(DATA_DIR))

for file in os.listdir(DATA_DIR):
    if file.endswith(".csv") and "gesture_results" not in file:

        full_path = os.path.join(DATA_DIR, file)
        print(f"\n📌 Çiziliyor → {full_path}")

        df = pd.read_csv(full_path)

        time = df.iloc[:,0]
        ax = df.iloc[:,1]
        ay = df.iloc[:,2]
        az = df.iloc[:,3]

        ax_f = low_pass_filter(ax)
        ay_f = low_pass_filter(ay)
        az_f = low_pass_filter(az)
        mag  = np.sqrt(ax_f**2 + ay_f**2 + az_f**2)

        plt.figure(figsize=(10,6))
        plt.suptitle(f"Gesture Plot → {file}")

        plt.subplot(2,1,1)
        plt.plot(time, ax, label="X raw")
        plt.plot(time, ay, label="Y raw")
        plt.plot(time, az, label="Z raw")
        plt.legend()
        plt.title("Raw Accelerometer Data")

        plt.subplot(2,1,2)
        plt.plot(time, mag, color="black")
        plt.title("Filtered Magnitude")
        plt.xlabel("Time")
        plt.ylabel("Magnitude")

        plt.tight_layout()
        plt.show()
