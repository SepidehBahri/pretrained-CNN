import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src import config

def get_preprocess_function():
    """Returns the appropriate preprocess_input function for the chosen model."""
    if not config.USE_PREPROCESS_INPUT:
        return None

    from tensorflow.keras.applications import (
        InceptionResNetV2, ResNet50, EfficientNetB0, EfficientNetV2S,
        DenseNet201, InceptionV3
    )

    preprocess_map = {
        "InceptionResNetV2": InceptionResNetV2.preprocess_input,
        "ResNet50": ResNet50.preprocess_input,
        "EfficientNetB0": EfficientNetB0.preprocess_input,
        "EfficientNetV2S": EfficientNetV2S.preprocess_input,
        "DenseNet201": DenseNet201.preprocess_input,
        "InceptionV3": InceptionV3.preprocess_input,
    }

    return preprocess_map.get(config.MODEL_NAME)

def create_data_generators():
    """Prepares train and validation generators from image folders."""
    all_data = []
    class_names = sorted(os.listdir(config.DATASET_PATH))
    for class_name in class_names:
        class_dir = os.path.join(config.DATASET_PATH, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(config.IMAGE_FORMAT):
                    all_data.append((os.path.join(class_name, file), class_name))

    df = pd.DataFrame(all_data, columns=["filename", "class"])

    train_df, val_df = train_test_split(
        df,
        train_size=config.TRAIN_SPLIT,
        random_state=config.SEED,
        stratify=df["class"]
    )

    # ImageDataGenerators
    preprocess_fn = get_preprocess_function()

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=config.ROTATION_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        vertical_flip=config.VERTICAL_FLIP
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=config.DATASET_PATH,
        x_col='filename',
        y_col='class',
        target_size=config.IMAGE_SIZE,
        class_mode='categorical',
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        seed=config.SEED
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=config.DATASET_PATH,
        x_col='filename',
        y_col='class',
        target_size=config.IMAGE_SIZE,
        class_mode='categorical',
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        seed=config.SEED
    )

    return train_generator, val_generator
