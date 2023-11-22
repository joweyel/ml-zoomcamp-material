#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

# TF-Less preprocessing
preprocessor = create_preprocessor("xception", target_size=(299, 299))

# Load TF-Lite model and get input-/output-indices for the model
interpreter = tflite.Interpreter(model_path="clothing_model.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Loading the data
# url = "http://bit.ly/mlbookcamp-pants"

classes = [
    "dress",
    "hat",
    "longsleeve",
    "outwear",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "t-shirt"
]

def predict(url):
    X = preprocessor.from_url(url)
    # Running the inference / doing prediction
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_preds = preds[0].tolist()

    return dict(zip(classes, float_preds))


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result