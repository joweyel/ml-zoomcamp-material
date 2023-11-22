#!/usr/bin/env python
# coding: utf-8

from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

def download_image(url):
    '''Gets image from url and converts to PIL.Image'''
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# Preprocessing ([0, 255] uint8 -> [0, 1] float)
def preprocess_input(x):
    x /= 255.0
    return x

tflite_model_path = "bees-wasps-v2.tflite"
image_size = 150

# TF-Less preprocessing
preprocessor = create_preprocessor("xception", target_size=(image_size, image_size))

# Load TF-Lite model and get input-/output-indices for the model
interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(image_size, image_size))
    x = np.array(img, dtype=np.float32)[None, ...]
    X = preprocess_input(x)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    float_pred = pred[0].tolist()
    return float_pred

def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result