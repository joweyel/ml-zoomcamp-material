## 10. Kubernetes and TensorFlow Serving

- 10.1 [Overview](#01-overview)
- 10.2 [TensorFlow Serving](#02-tensorflow-serving)
- 10.3 [Creating a pre-processing service](#03-preprocessing)
- 10.4 [Running everything locally with Docker-compose](#04-docker-compose)
- 10.5 [Introduction to Kubernetes](#05-kubernetes-intro)
- 10.6 [Deploying a simple service to Kubernetes](#06-kubernetes-simple-service)
- 10.7 [Deploying TensorFlow models to Kubernetes](#07-kubernetes-tf-serving)
- 10.8 [Deploying to EKS](#08-eks)
- 10.9 [Summary](#09-summary)
- 10.10 [Explore more](#10-explore-more)
- 10.11 [Homework](#homework)

<a id="01-overview"></a>
## 10.1 Overview

- What we will cover this week
- Two-tier architecture


In this Section the same scenario as in previous weeks is used (`clothing classification`) while utilizing `Kubernetes` and `Tensorflow Serving`

### Efficient TensorFlow Serving for image prediction
- TensorFlow-library written in C++ that is specialized to serve trained models
- Can only do inference

### Get predictions from uploaded images

**Website (W):**
- Front-end where a user can provide an Image-URL and then receives a prediction for the given URL
- (W $\rightarrow$ G): Website passes URL to Gateway
- (W $\leftarrow$ G): Receive predictions from Gateway in JSON-format and visualizes them

**Gateway (G):**
- Intermediate service for pre- and post-processing, that uses `Flask`
- Inference-direction
    - (W $\rightarrow$ G): Gets URL from Website, downloads the image and preprocesses it
    - (G $\rightarrow$ T): Sends numpy array to TF-Serving module
- Result-direction
    - (G $\leftarrow$ T): Receives predictions with gRPC and processes it
    - (W $\leftarrow$ G): Sends processed inference-results to the website in JSON-format

**TF-Serving (T):**
- Runs inference on provided numpy arrays
- (G $\rightarrow$ T): Receive pre-processed numpy array
- (G $\leftarrow$ T): Returns predictions with gRPC-protocol

![overview](imgs/overview.jpg)


<a id="02-tensorflow-serving"></a>
## 10.2 TensorFlow Serving

- The saved model format
- Running TF-Serving locally with Docker
- Invoking the model from Jupyter

### Converting Keras model into SavedModel format
- The first step is to download the required model
```bash
# Model was saved to the code-directory
wget -c https://github.com/DataTalksClub/machine-learning-zoomcamp/releases/download/chapter7-model/xception_v4_large_08_0.894.h5 -O code/clothing-model-v4.h5
```
- Now the model has to be loaded and converted with the following code:
```python
import tensorflow as tf
from tensorflow import keras

model_path = "clothing-model-v4.h5"
model = keras.models.load_model(model_path)
tf.saved_model.save(model, "clothing-model")
```
- There will be a newly generated folder, in which the `SavedModel` is saved to. To visualize it's content the utility `tree` can be used:
```sh
$ tree clothing-model
 
# Returns:
# clothing-model
# ├── assets
# ├── fingerprint.pb
# ├── saved_model.pb
# └── variables
#     ├── variables.data-00000-of-00001
#     └── variables.index
# 
# 2 directories, 4 files 
```
- Here the `saved_model.pb` contains the model-definition and the content of the variable folder contains the weights of the model
- To get an more in-depth insight into the saved model, the following code can be used:
```sh
saved_model_cli show --dir clothing-model --all
```
- There is a particular section in the output that is interesting. This is because it contains the model-description, which ist the folowing:
```yaml
...
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_8'] tensor_info:       # 
        dtype: DT_FLOAT                  # Input
        shape: (-1, 299, 299, 3)         # 
        name: serving_default_input_8:0  # 
  The given SavedModel SignatureDef contains the following output(s):
    outputs['dense_7'] tensor_info:      #  
        dtype: DT_FLOAT                  # Output
        shape: (-1, 10)                  # 
        name: StatefulPartitionedCall:0  # 
  Method name is: tensorflow/serving/predict
...
```
- The importance of this particular section comes the need to know the signature of the model when it is invoked (here: `serving_defaults`)
- For later usage the signature, as well as the input and output tensor-info have to be saved in `model-description.txt`
```
serving_default
input_8 - input
dense_7 - output
```

### Running TF-Serving locally with Docker
- To run tf-Serving inside a Docker container you have to follow a scecific procedure
```sh
docker run -it --rm \
    -p 8500:8500 \
    -v "$(pwd)/clothing-model:/models/clothing-model/1" \
    -e MODEL_NAME="clothing-model" \
    tensorflow/serving:latest # example in video was :2.7.0
```
- **Meaning of all the parameters**
    - **`-p`**: Maps the host port (`8500:`) to the container port (`:8500`)
    - **`-v`**: Maps host volume/folder to container volume/folder (host: `$(pwd)/clothing-model`, container: `/models/clothing-model/1`)
    - **`-e`**: To set environment variable inside container (here: `MODEL_NAME`)
    - **`tensorflow/serving:tag`**: The TF-Serving Docker-Image (with specified tag)
- After running the command above the docker container should be up and running. You can now send requests to the TF-Serving model inside the container.

### Invoking the model from Jupyter
- The relevant code for this can be found inside [this](code/tf-serving-connect.ipynb) notebook.

<a id="03-preprocessing"></a>
## 10.3 Creating a pre-processing service

- Converting the notebook to a Python script
- Wrapping the script into a Flask app
- Putting everything into Pipenv

### Converting the code from the notebook to a Flask-Application
- `Previously`: Used Notebook for communicating with TF-Serving service
- `Now`: Exporting Notebook to Python-Script
```sh
jupyter nbconvert --to script tf-serving-connect.ipynb
```
- Renaming the code to `gateway.py`. This is the software component from the first subsection that connects Website-Inputs to the TF-Serving model and the other way around.


The final code for the `Gateway` can be found [here](code/gateway.py). To utilize communicate with the TF-Serving model you have to use the following code (each command in separate console):
```sh
# Starting the TF-Serving service
docker run -it --rm \
    -p 8500:8500 \
    -v "$(pwd)/clothing-model:/models/clothing-model/1" \
    -e MODEL_NAME="clothing-model" \
    tensorflow/serving:latest

# Gateway Flask-App for communicating with the TF-Serving model
python3 gateway.py

# Script for sending requests to the TF-Serving model
python3 test.py
```
The code:
- `Gateway`: [gateway.py](code/gateway.py)
- `Test-Script`: [test.py](code/test.py)


The results:
```sh
{'dress': -1.868287444114685, 'hat': -4.761244297027588, 'longsleeve': -2.316981554031372, 'outwear': -1.0625675916671753, 'pants': 9.887153625488281, 'shirt': -2.8124289512634277, 'shoes': -3.6662802696228027, 'shorts': 3.200357437133789, 'skirt': -2.6023383140563965, 't-shirt': -4.835044860839844}
```

### Putting everything into Pipenv
**Step 1:** Creating Pipenv-environment with required dependencies
```sh
pipenv install grpcio flask gunicorn keras-image-helper
# Everythign relevant for TF-Serving using gPRC and protobuf
pipenv install tensroflow-protobuf
``` 
- Using [`tensorflow-protobuf`](https://github.com/alexeygrigorev/tensorflow-protobuf) instead of `TensorFlow` (1.7GB dependency)
  - [proto.py](code/proto.py) contains everything relevant for tensorflow

<a id="04-docker-compose"></a>
## 10.4 Running everything locally with Docker-compose

- Preparing the images
- Installing docker-compose
- Running the service
- Testing the service

**Previously**: website/test-script and Gateway outside docker container and TF-Serving model inside docker container
**Now**: Putting Gateway and TF-Serving model into docker container

### Preparing the docker-images
Previously used code for the docker-container of the TF-Serving model
```sh
docker run -it --rm \
    -p 8500:8500 \
    -v $(pwd)/clothing-model:/models/clothing-model/1 \
    -e MODEL_NAME=clothing-model \
    tensorflow/serving:2.7
```
Putting everything into a [Dockerfile](code/image-model.dockerfile) to make it reproducible
```dockerfile
FROM tensorflow/serving:2.7.0
ss
COPY clothing-model /models/clothing-model/1
ENV MODEL_NAME="clothing-model"
```
Building the docker-container:
```sh
docker build -t zoomcamp-test-10-model:xception-v4-001 -f image-model.dockerfile .
```
Runnung the docker-container
```sh
docker run -it --rm \
    -p 8500:8500 \
    zoomcamp-test-10-model:xception-v4-001
```
Testing the model
```sh
pipenv run python3 gateway.py
```

Now the gateway has to be put in a docker-container:

The [dockerfile](code/image-gateway.dockerfile) for the Gateway
```dockerfile
FROM python:3.8.12-slim

RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy
RUN pip install numpy
COPY ["gateway.py", "proto.py", "./"]
EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "gateway:app" ]
```
Building the docker-container for the Gateway
```sh
docker build -t zoomcamp-10-gateway:001 -f image-gateway.dockerfile .
```
Running the docker-container of the Gateway
```sh
docker run -it --rm \
    -p 9696:9696 \
    zoomcamp-10-gateway:001
```

***`TODO`***


<a id="05-kubernetes-intro"></a>
## 10.5 Introduction to Kubernetes

- The anatomy of a Kubernetes cluster


<a id="06-kubernetes-simple-service"></a>
## 10.6 Deploying a simple service to Kubernetes

- Installing `kubectl`
- Setting up a local Kubernetes cluster with Kind
- Create a deployment
- Creating a service


<a id="07-kubernetes-tf-serving"></a>
## 10.7 Deploying TensorFlow models to Kubernetes

- Deploying the TF-Service model
- Deploying the Gateway
- Testing the service


<a id="08-eks"></a>
## 10.8 Deploying to EKS

- Creating a EKS cluster on AWS
- Publishing the image to ECR
- Configuring kubectl


<a id="09-summary"></a>
## 10.9 Summary

- TF-Serving is a system for deploying TensorFlow models
- When using TF-Serving, we need a component for pre-processing
- Kubernetes is a container orchestration platform
- To deploy something on Kubernetes, we need to specify a deployment and a service
- You can use Docker compose and Kind for local experiments

<a id="10-explore-more"></a>
## 10.10 Explore more

<a id="homework"></a>
## 10.11 Homework
