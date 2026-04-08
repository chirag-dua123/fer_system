import os
import tensorflow as tf
import zipfile
import subprocess
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import build_model

# Constants
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/test'
BATCH_SIZE = 64
EPOCHS = 15
IMG_HEIGHT, IMG_WIDTH = 48, 48
NUM_CLASSES = 7

def setup_kaggle_and_download():
    """
    Downloads and extracts the FER-2013 dataset using the Kaggle API.
    """
    dataset_name = "msambare/fer2013"
    download_dir = Path("data")
    zip_file = "fer2013.zip"

    if (download_dir / "train").exists() and (download_dir / "test").exists():
        print("Dataset already exists in 'data/'. Skipping download.")
        return

    if not download_dir.exists():
        download_dir.mkdir(parents=True)

    print(f"Downloading {dataset_name} from Kaggle...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name], check=True)
    except Exception as e:
        print(f"Error: {e}. Make sure kaggle.json is in ~/.kaggle/ and permissions are set.")
        return

    print(f"Extracting {zip_file}...")
    if not os.path.exists(zip_file):
        # Handle case where kaggle downloads it with a different name
        possible_zips = list(Path('.').glob('*.zip'))
        if possible_zips:
            zip_file = possible_zips[0]
        else:
            print("Zip file not found.")
            return

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    
    if os.path.exists(zip_file):
        os.remove(zip_file)
    print("Dataset preparation complete.")

def train_model():
    """
    Trains the FER model using the FER-2013 dataset.
    Automatically downloads dataset from Kaggle if not present.
    """
    # 0. Download dataset if needed
    setup_kaggle_and_download()
    
    # 1. Data Augmentation and Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        shear_range=0.3,
        zoom_range=0.3,
        width_shift_range=0.4,
        height_shift_range=0.4,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Note: Using grayscale as FER-2013 is grayscale
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        color_mode='grayscale',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        color_mode='grayscale',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    # 2. Build and Compile Model
    model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_classes=NUM_CLASSES)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    # 3. Callbacks
    checkpoint = ModelCheckpoint(
        'emotion_model.h5', 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True, 
        mode='max'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=1,
        restore_best_weights=True
    )

    reduce_learningrate = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1,
        min_delta=0.0001
    )

    callbacks_list = [early_stopping, checkpoint, reduce_learningrate]

    # 4. Training
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Dataset not found at {TRAIN_DIR}.")
        return

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        callbacks=callbacks_list
    )

    return history

if __name__ == "__main__":
    train_model()
