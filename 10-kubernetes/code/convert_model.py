import tensorflow as tf
from tensorflow import keras

model_path = "clothing-model-v4.h5"
model = keras.models.load_model(model_path)
tf.saved_model.save(model, "clothing-model")
