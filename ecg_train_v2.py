import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from keras.utils import to_categorical

# Load your dataset
df = pd.read_csv("balanced_ecg_dataset.csv")  # Or your original dataset if SMOTE caused issues

# Normalize ECG signal
def normalize_ecg(ecg):
    ecg = np.array(eval(ecg))  # Assuming the signal is stored as a string
    return (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg)) * 2 - 1

df['signal'] = df['signal'].apply(normalize_ecg)

# Extract features and labels
X = np.stack(df['signal'].values)
y = df['label'].values

# Reshape X for Conv1D input
X = X.reshape(-1, X.shape[1], 1)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Train-test split
X_train, X_val, y_train_encoded, y_val_encoded, y_train, y_val = train_test_split(
    X, y_onehot, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Define improved CNN model
model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(256, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train_encoded,
    validation_data=(X_val, y_val_encoded),
    epochs=15,
    batch_size=32,
    class_weight=class_weights
)

# Save model
model.save("ecg_cnn_model_improved.h5")
print("✅ Model saved as ecg_cnn_model_improved.h5")

# Evaluate and visualize
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val_encoded, axis=1)

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Save class-wise input signals for LLM (if needed)
for class_label in np.unique(y_encoded):
    sample = X[y_encoded == class_label][0]
    np.save(f"data/sample_input_class_{class_label}.npy", sample)
