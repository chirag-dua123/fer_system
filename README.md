# Facial Emotion Recognition (FER) System

This project implements a real-time facial emotion recognition system using a Deep Convolutional Neural Network (CNN) built with Keras/TensorFlow.

## Directory Structure
- `model.py`: Contains the CNN architecture.
- `train.py`: Script to train the model on the FER-2013 dataset.
- `inference.py`: Real-time emotion detection using webcam.
- `requirements.txt`: Python dependencies.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Google Colab Setup (Kaggle Dataset)
If you are running this in Google Colab, you can download the dataset automatically:

1.  **Get Kaggle API Token**: Go to your Kaggle account -> Settings -> Create New API Token. This will download `kaggle.json`.
2.  **Upload to Colab**: Upload `kaggle.json` to the Colab runtime.
3.  **Run Setup**:
    ```python
    import os
    # Move kaggle.json to the required location
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    import shutil
    shutil.copy("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    ```

The `train.py` script will now automatically download and extract the FER-2013 dataset using the Kaggle API.

### 3. Training
Run the training script to generate `emotion_model.h5`:
```bash
python train.py
```

### 4. Real-time Inference
Once the model is trained, start the real-time detector:
```bash
python inference.py
```

## Model Architecture Explanation
The CNN consists of 4 main convolutional blocks:
1. **Feature Extraction**: Each block uses `Conv2D` layers followed by `BatchNormalization` to stabilize learning and `MaxPooling2D` to reduce spatial dimensions.
2. **Regularization**: `Dropout` layers are used in each block to prevent overfitting by randomly setting a fraction of input units to 0 during training.
3. **Classification**: A `Flatten` layer converts the 2D feature maps into a 1D vector, which is then passed through a `Dense` layer (512 units) and finally a `Softmax` output layer with 7 units (one for each emotion).
