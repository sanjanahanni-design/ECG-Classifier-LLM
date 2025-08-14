import numpy as np
from keras.models import load_model
from google.generativeai import configure, GenerativeModel

# Load CNN model
model = load_model("ecg_cnn_model_improved.h5")

# Configure Gemini
configure(api_key="your api key")  # Replace with your actual key
genai = GenerativeModel("gemini-1.5-flash")

# Predict and explain
for class_label in range(5):
    try:
        sample_path = f"data/sample_input_class_{class_label}.npy"
        sample = np.load(sample_path).reshape(1, -1, 1)  # (1, time_steps, 1)

        prediction = model.predict(sample)
        predicted_class = np.argmax(prediction)
        probs = prediction.flatten()
        probs_str = ", ".join([f"Class {i}: {p:.2f}" for i, p in enumerate(probs)])

        # 👇 ECG plot section
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(sample[0].squeeze(), color='blue')
        plt.title(f"ECG Signal (Class {class_label}) - Predicted: Class {predicted_class}")
        plt.xlabel("Time Steps")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"\n🔍 Loaded sample from {sample_path}")
        print(f"📊 CNN Predicted Class: {predicted_class}")

        # Better prompt
        prompt = (
            f"This is an ECG signal predicted by a CNN model as class {predicted_class}.\n"
            f"Class meanings are:\n"
            f"0 - Normal beat\n"
            f"1 - Supraventricular ectopic beat\n"
            f"2 - Ventricular ectopic beat\n"
            f"3 - Fusion beat\n"
            f"4 - Unknown beat\n\n"
            f"Predicted class probabilities: {probs_str}\n\n"
            f"Explain what this class means clinically and what could be the possible diagnosis or underlying condition."
        )

        response = genai.generate_content(prompt)
        print(f"💬 Gemini Diagnosis Explanation: {response.text.strip()}")

    except FileNotFoundError:
        print(f"❌ Sample file not found: {sample_path}")
    except Exception as e:
        print(f"⚠️ Error processing class {class_label}: {e}")
