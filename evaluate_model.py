import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model("D:\\vscode\\Python\\new_traffic_model.keras")

test_dir = "D:/vscode/Python/Final Dataset/testing"
img_size = (224, 224)
batch_size = 32
class_labels = ["Empty", "Low", "Medium", "High", "Traffic Jam"]

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)
print(f"\n Test Accuracy: {accuracy*100:.2f}%")
print(f" Test Loss: {loss:.4f}")

predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = test_generator.classes

print("\n Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_labels))

cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
