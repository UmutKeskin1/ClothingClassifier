import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import random as r
import numpy as np
(train_data,train_labels),(test_data,test_labels)=fashion_mnist.load_data()
###############################################################################
tf.random.set_seed(42)
model=tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(128,activation="relu"),
  tf.keras.layers.Dense(10,activation="softmax")
])
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)
train_data=train_data/255.0
test_data=test_data/255.0
model.fit(train_data,
          train_labels,
          validation_data=(test_data,test_labels),
          epochs=20)
predictions = model.predict(test_data)

max_indices = np.argmax(predictions, axis=1)
names=["T-shirt/üst","Pantolon","Kazak","Elbise","Ceket","Sandalet","Gömlek","Spor Ayakkabı","Çanta","Bot"]
predicted_classes = [names[max_index] for max_index in max_indices]
random_indices = r.sample(range(len(test_data)), 2)
for i in random_indices:
  plt.imshow(test_data[i],cmap="binary")
  plt.title(f"Tahmin: {predicted_classes[i]}")
  plt.savefig("1.eps", dpi=200)
  plt.show()
  plt.imshow(test_data[i+1],cmap="binary")
  plt.savefig("2.eps", dpi=200)
  plt.title(f"Tahmin: {predicted_classes[i+1]}")
  plt.show()
