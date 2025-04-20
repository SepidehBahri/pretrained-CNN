from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from src import config

def get_base_model():
    """Returns a pre-trained CNN backbone based on the configuration."""
    from tensorflow.keras.applications import (
        InceptionResNetV2, ResNet50, EfficientNetB0,
        EfficientNetV2S, DenseNet201, InceptionV3
    )

    input_shape = (*config.IMAGE_SIZE, 3)

    model_map = {
        "InceptionResNetV2": InceptionResNetV2,
        "ResNet50": ResNet50,
        "EfficientNetB0": EfficientNetB0,
        "EfficientNetV2S": EfficientNetV2S,
        "DenseNet201": DenseNet201,
        "InceptionV3": InceptionV3,
    }

    model_class = model_map.get(config.MODEL_NAME)
    if model_class is None:
        raise ValueError(f"Model '{config.MODEL_NAME}' not supported.")

    base_model = model_class(include_top=False, weights="imagenet", input_shape=input_shape)
    return base_model

def build_model():
    """Builds the full transfer learning model using a pretrained base."""
    base_model = get_base_model()

    # Freeze first N layers if specified
    if config.FREEZE_LAYERS is not None:
        for layer in base_model.layers[:config.FREEZE_LAYERS]:
            layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dense(config.NUM_CLASSES, activation='softmax')
    ])

    optimizer = Adam(learning_rate=config.LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'AUC']
    )

    return model
