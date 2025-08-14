# ECG-Classifier-LLM
## 1D CNN-based ECG Classifier with LLM-powered clinical explanation.
This project implements a 1D Convolutional Neural Network (CNN) to classify ECG signals from the MIT-BIH Arrhythmia Database into five heartbeat classes. It integrates a Large Language Model (LLM) (Gemini API) to provide human-readable clinical explanations for predicted heart conditions, enhancing interpretability for non-technical users and healthcare professionals. The project includes signal visualization for better insight into ECG patterns.

📂 Dataset

MIT-BIH Arrhythmia Dataset from Kaggle.

Preprocessing includes normalization, segmentation, and SMOTE for class balance.

🛠️ Tech Stack

Python

TensorFlow / Keras – Model training

Matplotlib – ECG signal visualization

Pandas / NumPy – Data handling

Scikit-learn – Data preprocessing & evaluation

Google Gemini API – LLM-based diagnosis generation
