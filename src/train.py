import os
import json
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import load_model
from src import config
from src.data_loader import create_data_generators
from src.model_builder import build_model

def main():
    # Create output directories if they don’t exist
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)

    # === Load Data ===
    train_gen, val_gen = create_data_generators()

    # === Build or Load Model ===
    if os.path.exists(config.MODEL_PATH):
        print(f" Loading existing model from {config.MODEL_PATH}")
        model = load_model(config.MODEL_PATH)
    else:
        print("Building new model from scratch.")
        model = build_model()
        model.summary()

    # === Custom Callback to Save Logs as JSON ===
    class HistoryLogger(Callback):
        def on_epoch_end(self, epoch, logs=None):
            with open(config.LOG_PATH, "a") as f:
                json.dump(logs, f)
                f.write("\n")

    # === Setup Callbacks ===
    callbacks = [
        ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            filepath=config.WEIGHTS_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max',
            save_weights_only=True
        ),
        HistoryLogger()
    ]

    # === Train the Model ===
    print("Training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print(" Training complete.")
