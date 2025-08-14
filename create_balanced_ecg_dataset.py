import pandas as pd
from imblearn.over_sampling import SMOTE

# Load datasets
train = pd.read_csv("ECG-Heartbeat-Categorization-Dataset-main/mitbih_train.csv", header=None)
test = pd.read_csv("ECG-Heartbeat-Categorization-Dataset-main/mitbih_test.csv", header=None)

# Combine datasets
data = pd.concat([train, test], axis=0)

# Split features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create a 'signal' column by converting each row to a list
signals = X_resampled.apply(lambda row: list(row), axis=1)

# Final DataFrame with 'signal' and 'label'
balanced_df = pd.DataFrame({
    'signal': signals,
    'label': y_resampled
})

# Save to CSV
balanced_df.to_csv("balanced_ecg_dataset.csv", index=False)
print("✅ balanced_ecg_dataset.csv created with signal as list and label.")
