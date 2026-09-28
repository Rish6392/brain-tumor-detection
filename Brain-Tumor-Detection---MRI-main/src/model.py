import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



# 1. Tumor Detection Model 

def build_detection_model(input_shape=(224, 224, 3)):

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # KEEP FULLY FROZEN
    base_model.trainable = False

    model = Sequential([
        tf.keras.Input(shape=input_shape),

        Lambda(preprocess_input),

        base_model,

        GlobalAveragePooling2D(),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),   # slightly reduced

        Dense(1, activation='sigmoid')
    ])

    return model



# 2. Tumor Type Model 

def build_classification_model(input_shape=(224, 224, 1), num_classes=3):

    model = Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2,2))

    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2,2))

    model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2,2))

    model.add(tf.keras.layers.Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    return model



# 3. Compile Functions (FINAL)

def compile_detection_model(model):
    model.compile(
        optimizer=Adam(learning_rate=1e-4),  
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def compile_classification_model(model):
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model