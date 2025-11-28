import pandas as pd

# CSV dosyasının tam yolu
df = pd.read_csv(r"C:\Users\Fatih\DSP_Gesture_Project\gesture_results.csv")

# gerçek etiketleri belirleme (dosya adından)
df["actual"] = df["file"].apply(lambda x:
    "LEFT"  if "left" in x.lower() else
    "RIGHT" if "right" in x.lower() else
    "UP"
)

# Doğru tahmin kontrolü
df["correct"] = df["prediction"] == df["actual"]

accuracy = df["correct"].mean() * 100

print("\n===== CLASSIFICATION RESULTS =====")
print(df)
print(f"\n🎯 MODEL ACCURACY = {accuracy:.2f}%\n")
