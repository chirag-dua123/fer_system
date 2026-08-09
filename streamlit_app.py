import os

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("emotion_model.h5 not found. Train the model first.")
    model = load_model(MODEL_PATH)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise ValueError("Unable to load Haar Cascade for face detection.")
    return model, detector


def preprocess_face(face_gray: np.ndarray) -> np.ndarray:
    roi_gray = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    roi = roi_gray.astype("float32") / 255.0
    roi = img_to_array(roi)
    return np.expand_dims(roi, axis=0)


def detect_and_predict(image_bgr: np.ndarray, model, face_detector):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    annotated = image_bgr.copy()
    predictions = []

    for (x, y, w, h) in faces:
        face_gray = gray[y : y + h, x : x + w]
        if face_gray.size == 0:
            continue

        roi = preprocess_face(face_gray)
        scores = model.predict(roi, verbose=0)[0]
        emotion_idx = int(np.argmax(scores))
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = float(scores[emotion_idx])
        predictions.append((emotion, confidence))

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            annotated,
            f"{emotion} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return annotated, predictions


def decode_uploaded_image(file_bytes: bytes):
    data = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    st.set_page_config(page_title="FER Streamlit Frontend", page_icon="🎭", layout="wide")
    st.title("🎭 Facial Emotion Recognition")
    st.write("Upload an image or capture one from camera to detect facial emotions.")

    try:
        model, face_detector = load_resources()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return

    input_mode = st.radio("Choose input source", ["Upload image", "Use camera"])
    image_bgr = None

    if input_mode == "Upload image":
        uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
        if uploaded:
            image_bgr = decode_uploaded_image(uploaded.read())
    else:
        camera_image = st.camera_input("Capture image")
        if camera_image:
            image_bgr = decode_uploaded_image(camera_image.getvalue())

    if image_bgr is None:
        return

    if image_bgr.size == 0:
        st.error("Unable to read the image. Please try another image.")
        return

    annotated_bgr, predictions = detect_and_predict(image_bgr, model, face_detector)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Prediction")
        st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    if not predictions:
        st.warning("No face detected in the image.")
        return

    st.subheader("Detected emotions")
    for index, (emotion, confidence) in enumerate(predictions, start=1):
        st.write(f"Face {index}: **{emotion}** ({confidence:.2%})")


if __name__ == "__main__":
    main()
