import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight


class_labels = ["Empty", "Low", "Medium", "High", "Traffic Jam"]
dataset_path = "D:/vscode/Python/Final Dataset/training"
validation_path="D:/vscode/Python/Final Dataset/validation"

class_counts = {label: len(os.listdir(os.path.join(dataset_path, label))) for label in class_labels}
class_counts_list = [class_counts[label] for label in class_labels]
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2, 3, 4]),
    y=np.concatenate([np.full(count, i) for i, count in enumerate(class_counts_list)])
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest",
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    validation_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False
)

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(class_labels), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)

model.compile(optimizer=Adam(learning_rate=0.00005),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

early_stopping = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
epochs = 30

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    class_weight=class_weight_dict,
    callbacks=[early_stopping]
)

model.save("updated_traffic_model.keras")
print("Model training complete and saved.")