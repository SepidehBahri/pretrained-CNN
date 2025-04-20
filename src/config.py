import os

# === Dataset Configuration ===
DATASET_PATH = "data"  # Root folder containing class subfolders
TRAIN_SPLIT = 0.8       # Percentage for training split
IMAGE_FORMAT = "jpg"    # Optional: for future filtering

# === Model Configuration ===
MODEL_NAME = "InceptionResNetV2"  # Options: ResNet50, EfficientNetB0, DenseNet201, etc.
FREEZE_LAYERS = 150               # Number of layers to freeze in base model
USE_PREPROCESS_INPUT = True       # Use Keras' preprocess_input
NUM_CLASSES = 2                  # Change based on your dataset

# === Image & Batch Config ===
IMAGE_SIZE = (299, 299) if MODEL_NAME == "InceptionResNetV2" else (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
SEED = 42

# === Augmentation Settings ===
ROTATION_RANGE = 30
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False

# === Optimizer Settings ===
LEARNING_RATE = 0.001


# === Output Paths ===
MODEL_DIR = "models"
LOG_DIR = "logs"
PLOT_DIR = "plots"

# === Auto-generated Model ID ===
MODEL_ID = f"{MODEL_NAME}_frozen{FREEZE_LAYERS}_{NUM_CLASSES}class_lr{LEARNING_RATE}_bs{BATCH_SIZE}"

# === File Paths ===
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_ID}.keras")
WEIGHTS_PATH = os.path.join(MODEL_DIR, f"{MODEL_ID}.weights.h5")
LOG_PATH = os.path.join(LOG_DIR, f"{MODEL_ID}.json")
CONFUSION_MATRIX_PATH = os.path.join(PLOT_DIR, f"{MODEL_ID}_confusion_matrix.png")
