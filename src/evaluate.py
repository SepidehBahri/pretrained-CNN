import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from src import config
from src.data_loader import create_data_generators

def main():
    # === Load validation data ===
    _, val_gen = create_data_generators()

    # === Load best model ===
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {config.MODEL_PATH}")

    print(f"Loading model from: {config.MODEL_PATH}")
    model = load_model(config.MODEL_PATH)

    # === Evaluate on validation data ===
    print("🔍 Evaluating model...")
    y_true = val_gen.classes
    y_pred = model.predict(val_gen, verbose=1)

    y_pred_labels = np.argmax(y_pred, axis=1)

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred_labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=list(val_gen.class_indices.keys()),
        yticklabels=list(val_gen.class_indices.keys())

    )
    plt.title("Confusion Matrix (%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH)
    plt.show()

    print(f"Confusion matrix saved to: {config.CONFUSION_MATRIX_PATH}")

    # === Load Training Log and Plot Metrics ===
    if not os.path.exists(config.LOG_PATH):
        raise FileNotFoundError(f"Training log not found at {config.LOG_PATH}")

    train_acc, val_acc, train_loss, val_loss = [], [], [], []

    with open(config.LOG_PATH, "r") as f:
        for line in f:
            record = json.loads(line)
            train_acc.append(record.get("accuracy"))
            val_acc.append(record.get("val_accuracy"))
            train_loss.append(record.get("loss"))
            val_loss.append(record.get("val_loss"))

    epochs = range(1, len(train_acc) + 1)

    # === Accuracy Plot ===
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, 'bo-', label="Train Accuracy")
    plt.plot(epochs, val_acc, 'r*-', label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, f"{config.MODEL_ID}_accuracy.png"))
    plt.show()

    # === Loss Plot ===
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'bo-', label="Train Loss")
    plt.plot(epochs, val_loss, 'r*-', label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, f"{config.MODEL_ID}_loss.png"))
    plt.show()
