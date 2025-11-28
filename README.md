🚀 DSP Gesture Recognition
Accelerometer-Based Machine Learning Gesture Classifier

By Fatih Çatalçam

<p align="center"> <img src="pca_gesture_train.png" width="500px"> </p>
🧠 Project Overview

This project performs gesture recognition using accelerometer data.
Three different hand movements were recorded, processed and classified using Machine Learning models:

Gesture	Symbol
LEFT	⬅️
RIGHT	➡️
UP	⬆️

The goal is real-time recognition of motion patterns for future embedded usage.

📂 Folder Structure
DSP_Gesture_Project
│── data/                # Raw CSV gesture recordings
│── src/                 # All Python files
│   ├── ml_classifier.py
│   ├── ml_advanced.py
│   ├── visualize.py
│   └── main.py
│
│── confusion_KNN_k=3.png
│── confusion_SVM_RBF.png
│── confusion_RandomForest.png
│── pca_gesture_train.png
│
└── README.md

🔬 Methodology Pipeline
Step	Description
1. Data Acquisition	Movements collected via accelerometer sensor
2. Feature Extraction	Peaks, mins, signal energy, std, mean etc.
3. Classification	ML models trained & tested
4. Evaluation	Performance metrics & confusion matrices
📈 Model Performance

| Model | Accuracy | Note |
|---|---|
| SVM (RBF) | 🟩 High (~90%) | Best Consistency |
| Random Forest | 🟨 Medium | Feature dependent |
| KNN (k=3) | 🟥 Lower (~60-70%) | Sensitive to dataset size |

🔥 Result Visualizations
<p align="center"> <img src="confusion_SVM_RBF.png" width="400"> <img src="confusion_RandomForest.png" width="400"><br> <img src="confusion_KNN_k=3.png" width="400"> </p>

📍 PCA Gesture Distribution

<p align="center"> <img src="pca_gesture_train.png" width="500"> </p>
⚙️ Run The Project
cd src
python ml_classifier.py


For visualization:

python visualize.py


Dataset must be inside /data.

📝 Notes & Experience

✔ Data collection required multiple attempts due to hand-movement noise
✔ Keeping gestures stable was challenging
✔ Preprocessing strongly affects classification quality
✔ SVM produced the most reliable results

👤 Author

Fatih Çatalçam
Computer Engineering — DSP Term Project
📩 Contact: (Eklenecekse mail yazabilirsin)

If this repo helped you, leave a ⭐ — it motivates more work!
