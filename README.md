# Facial Emotion Recognition (FER) System

A real-time facial emotion recognition system powered by a Deep Convolutional Neural Network (CNN) built with TensorFlow/Keras. The model is trained on the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and detects **7 emotions** from a live webcam feed.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 🎭 Detects 7 emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise**
- 📷 Real-time inference using your webcam
- 🧠 Custom CNN architecture with Batch Normalization and Dropout regularization
- 📦 Automatic dataset download via the Kaggle API
- 💾 Saves the best model checkpoint during training

---

## 📁 Project Structure

```
fer_system/
├── model.py          # CNN architecture definition
├── train.py          # Model training script
├── inference.py      # Real-time webcam emotion detection
├── download_data.py  # Standalone dataset download utility
├── requirements.txt  # Python dependencies
├── emotion_model.h5  # Saved model weights (generated after training)
└── data/
    ├── train/        # Training images (generated after download)
    └── test/         # Validation images (generated after download)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A webcam (for real-time inference)
- A [Kaggle account](https://www.kaggle.com/) with an API token

### 1. Clone the Repository

```bash
git clone https://github.com/chirag-dua123/fer_system.git
cd fer_system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Kaggle API

The training script automatically downloads the FER-2013 dataset from Kaggle. You need a Kaggle API token:

1. Go to your [Kaggle account settings](https://www.kaggle.com/settings) → **API** → **Create New Token**. This downloads `kaggle.json`.
2. Place the file in the appropriate location:

   **Linux / macOS / Google Colab:**
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

   **Windows:**
   ```
   C:\Users\<YourUsername>\.kaggle\kaggle.json
   ```

   **Google Colab (one-time setup):**
   ```python
   import os, shutil
   os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
   shutil.copy("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
   os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
   ```

### 4. Train the Model

Run the training script. It will automatically download the dataset if it is not present and save the best model to `emotion_model.h5`.

```bash
python train.py
```

| Hyperparameter | Value |
|---|---|
| Input size | 48 × 48 (grayscale) |
| Batch size | 64 |
| Epochs | 15 (with early stopping) |
| Optimizer | Adam (lr = 0.0001) |
| Loss | Categorical Cross-Entropy |

### 5. Real-time Inference

Once training is complete, launch the live emotion detector:

```bash
python inference.py
```

Press **`q`** to quit the webcam window.

---

## 🧠 Model Architecture

The CNN consists of **3 convolutional blocks** followed by a fully connected classification head:

```
Input (48×48×1)
  │
  ├─ Block 1: Conv2D(32) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  ├─ Block 2: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
  ├─ Block 3: Conv2D(256) → BN → Conv2D(256) → BN → MaxPool → Dropout(0.25)
  │
  ├─ Flatten
  ├─ Dense(512) → BN → Dropout(0.5)
  │
  └─ Output: Dense(7, softmax)
```

- **Batch Normalization** stabilizes and accelerates training.
- **Dropout** prevents overfitting.
- **Early Stopping** + **ReduceLROnPlateau** callbacks optimize training automatically.

---

## 📊 Dataset

The model is trained on [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013), a widely-used benchmark for facial expression recognition:

| Split | Images |
|---|---|
| Train | ~28,709 |
| Test | ~3,589 |

Images are 48×48 grayscale, labeled across 7 emotion categories.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | Model building and training |
| `opencv-python` | Webcam capture and face detection |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting (optional) |
| `scipy` | Scientific utilities |
| `kaggle` | Dataset download |

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) by [msambare](https://www.kaggle.com/msambare)
- Face detection: [OpenCV Haar Cascades](https://opencv.org/)
