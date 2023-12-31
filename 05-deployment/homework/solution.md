## Question 1

* Install Pipenv
* What's the version of pipenv you installed?
* Use `--version` to find out

**Solution:** `pipenv, version 2023.10.3`

## Question 2
- Use Pipenv to install Scikit-Learn version 1.3.1
- What's the first hash for scikit-learn you get in Pipfile.lock?

**Solution:** `sha256:0c275a06c5190c5ce00af0acbb61c06374087949f643ef32d355ece12c4db043`


## Question 3

Let's use these models!

* Write a script for loading these models with pickle
* Score this client:

```json
{"job": "retired", "duration": 445, "poutcome": "success"}
```

**Execution of Code:**
```
python3 predict-test.py ./customers/customer3.json
```

**Result from Code:** `0.901931`

**Solution:** `0.902`
- Solution Code: [q3_code.py](q3_code.py) 


## Question 4

Now let's serve this model as a web service

- Install Flask and gunicorn (or waitress, if you're on Windows)
- Write Flask code for serving the model
- Now score this client using `requests`:

```python
url = "YOUR_URL"
client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
requests.post(url, json=client).json()
```

What's the probability that this client will get a credit?

**Execution of Code:**
```bash
# In a console
python3 predict.py

# In another console
python3 predict-test.py ./customers/customer4.json
```
**Result from Code:** `0.13968947052356817`

**Solution:**  `0.140`

- Solution Code: [predict-test.py](predict-test.py), [predict.py](predict.py) 

## Docker
Obtaining the Docker-Image with:
```bash
docker pull svizor/zoomcamp-model:3.10.12-slim
```

## Question 5

Download the base image `svizor/zoomcamp-model:3.10.12-slim`. You can easily make it by using [docker pull](https://docs.docker.com/engine/reference/commandline/pull/) command.

**Results from code**
```bash
svizor/zoomcamp-model   3.10.12-slim   08266c8f0c4b   2 days ago     147MB
```
**Solution:** `147MB`

## Dockerfile

```Dockerfile
FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "q6_predict:app"]
```

Building the image in the homework-folder
```bash
docker build -t hw5-q6-container .
```

## Question 6

Let's run your docker container!

After running it, score this client once again:

```python
url = "YOUR_URL"
client = {"job": "retired", "duration": 445, "poutcome": "success"}
requests.post(url, json=client).json()
```

What's the probability that this client will get a credit now?

Running the created docker-container
```bash
    docker run -it --rm -p 9696:9696 hw5-q6-container
```
Calling the web-app inside the docker container
```bash
python3 predict-test.py ./customers/customer6.json
```

- Solution Code: [predict-test.py](predict-test.py), [q6_predict.py](q6_predict.py), [Dockerfile](Dockerfile)

**Result from code:**
```python
{'card_decision': True, 'credit_probability': 0.726936946355423}
```
**Solution:** `0.730`