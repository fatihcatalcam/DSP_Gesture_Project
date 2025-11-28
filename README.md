🎯 DSP-Based Gesture Recognition Project

Accelerometer Signal Processing + Machine Learning Classification

This project focuses on recognizing hand gestures using accelerometer signals and classifying them into three directional movements:

👉 LEFT
👉 RIGHT
👉 UP

The system was built entirely by Fatih Çatalçam as part of the DSP Course Term Project.

📂 Project Structure
Folder / File	Description
src/	All Python source codes (signal processing, ML models, visualization scripts)
data/	Recorded accelerometer CSV datasets for gestures
confusion_KNN_k=3.png	KNN confusion matrix visualization
confusion_SVM_RBF.png	SVM confusion matrix visualization
confusion_RandomForest.png	Random Forest confusion matrix visualization
pca_gesture_train.png	PCA projection of gesture clusters in feature space
gesture_results.csv	Peak-feature based classification results
README.md	Documentation for users & developers
🔧 Used Technologies
Component	Usage
Python	Main development language
Numpy, Pandas	Feature extraction from signal data
Matplotlib, Seaborn	Visualization & PCA plots
Scikit-Learn	ML models (KNN, SVM, Random Forest)
🔬 Methodology
1. Data Collection

Gesture movements were recorded through an accelerometer and exported as .csv files.

2. Feature Extraction

From each gesture file, the following characteristics were extracted:

📌 X Peak, X Minimum
📌 Z Peak, Z Minimum
📌 Mean, Standard Deviation, Signal Energy, etc.

3. Training & Testing

ML classification was performed using:

Model	Accuracy
KNN (k=3)	~66%
Random Forest	~72%
SVM (RBF Kernel)	~90% Best Performance 🏆

Models were trained using 70-30 split based on available dataset count.

📈 Results
Model	Performance
🔵 SVM → Best and most stable	
🟠 RandomForest → Medium performance	
🟢 KNN → Lower but functional	

Visualization examples:

Figure	Output
Confusion Matrix – SVM	confusion_SVM_RBF.png
PCA Gesture Distribution	pca_gesture_train.png
🚀 How to Run
cd src
python ml_classifier.py


Or for visualization:

python visualize.py


Ensure that your /data folder is located in the same directory.

👤 Author

Fatih Çatalçam
DSP Course — Computer Engineering

If you find this work useful, drop a ⭐ star on GitHub :)
