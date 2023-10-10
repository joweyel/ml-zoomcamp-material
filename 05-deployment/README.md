# 5. Deploying Machine Learning Models

- 5.1 [Intro / Session overview](#01-intro)
- 5.2 [Saving and loading the model](#02-pickle)
- 5.3 [Web services: introduction to Flask](#03-flask-intro)
- 5.4 [Serving the churn model with Flask](#04-flask-deployment)
- 5.5 [Python virtual environment: Pipenv](#05-pipenv)
- 5.6 [Environment management: Docker](#06-docker)
- 5.7 [Deployment to the cloud: AWS Elastic Beanstalk (optional)](#07-aws-eb)
- 5.8 [Summary](#08-summary)
- 5.9 [Explore more](#09-explore-more)
- 5.10 [Homework](#homework)

<a id="01-intro"></a>
## 5.1 Intro / Session overview

![this-week](./imgs/deployment.png)

- Deployment is to use machine learning model inside of an application an make it accessible
- Models are trained and saved
- Users of the machine learning model should be able to send requests to the model and get a prediction in return

### Plan for this Week (General overview)
![plan](./imgs/layers.png)

- Use a trained "churn-prediction" model (from the last weeks) and save it with `pickle`
- Turn the notebook into a Python script (the required parts)
- Put the trained model into a web-service (`Flaks`)
- Isolate all dependencies inside a specific python environment with `pipenv`
- Encapsulating all system dependencies inside a `Docker`-container
- Running the `Docker`-container inside AWS Elastic Beanstalk


<a id="02-pickle"></a>
## 5.2 Saving and loading the model

- Saving the model to `pickle`
- Loading the model from `pickle`
- Turning our notebook into a Python script

```python
import pickle
filanem = "model.bin"

# Saving
with open(filename, "wb") as f_out:
    pickle.dump((dv, model), f_out)

# Loading
with open(filename, "rb") as f_in.
    dv, model = pickle.load(f_in)
```

To export the Notebook as `Python`-file you can call the following command
```bash
jupyter nbconvert --to python notebook.ipynb
```


<a id="03-flask-intro"></a>
## 5.3 Web services: introduction to Flask

- Writing a simple ping/pong app
- Querying it with `curl` and browser

### What is a Web-Service?
- Program that is availbable over the Web
- A user sends a request / query from the local machine and gets a result from the web-service according to the query

### Creating a simple web-service
- Sending a `/ping`-request to the Flask Web-Service
- Receiving `pong`-result from the Flask Web-Service
- The Code can be found in the file [ping.py](./code/ping.py)

```python
from flask import Flask

app = Flask("ping")

@app.route("/ping", methods=["GET"])
def ping():
    return "PONG"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696) 
```

You can access the flask-app by using the browser
```bash
0.0.0.0:9696/ping
```
or by using the `curl` command
```bash
curl 0.0.0.0:9696/ping
``` 

<a id="04-flask-deployment"></a>
## 5.4 Serving the churn model with Flask

- Wrapping the predict script into a `Flask` app
- Querying it with production: `gunicorn`
- Running it on Windows with `waitress`

<a id="05-pipenv"></a>
## 5.5 Python virtual environment: Pipenv

- Why we need virtual environments
- Installing `Pipenv`
- Installing libraries with `Pipenv`
- Running things with `Pipenv`

<a id="06-docker"></a>
## 5.6 Environment management: Docker

- Why we need `Docker`
- Running a Python image with `Docker`
- `Dockerfile`
- Building a `Docker` image
- Running a `Docker` image

<a id="07-aws-eb"></a>
## 5.7 Deployment to the cloud: AWS Elastic Beanstalk (optional)

- Installing the `eb cli`
- Running `eb` locally
- Deploying the model

<a id="08-summary"></a>
## 5.8 Summary

- Save the model with `pickle`
- Use `Flask` to turn the model into a web service
- Use a dependency & env manager
- Package it in `Docker`
- Deploy to the cloud


<a id="09-explore-more"></a>
## 5.9 Explore more

- `Flask` is not the only framework for creating web services. Try others, e.g. `FastAPI`
- Experiment with other ways of managing environment, e.g. `virtualenv`, `conda`, `poetry`
- Explore other ways of deploying web services, e.g. `GCP`, `Azure`, `Heroku`, `Python Anywhere`, etc.

<a id="homework"></a>
## 5.10 Homework
