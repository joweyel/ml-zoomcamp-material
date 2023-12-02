## Homework Solutions

In this homework, we'll deploy the credit scoring model from the homework 5.
We already have a docker image for this model - we'll use it for 
deploying the model to Kubernetes.


## Bulding the image

Clone the course repo if you haven't:

```bash
git clone https://github.com/DataTalksClub/machine-learning-zoomcamp.git
```

Copy the content of the folder `machine-learning-zoomcamp/cohorts/2023/05-deployment/homework` to the current directory:
```bash
export HW10_PATH=$(pwd)
export HW5_PATH=machine-learning-zoomcamp/cohorts/2023/05-deployment/homework/
cd ${HW5_PATH} 
docker build -t zoomcamp-model:hw10 .
cp q6_test.py ${HW10_PATH}
cd ${HW10_PATH}
rm -rf machine-learning-zoomcamp  # Remove unused code
```

**Alternative**: Obtaining the container from docker-hub
```bash
docker pull svizor/zoomcamp-model:hw10
```

## Question 1

Run it to test that it's working locally:

```bash
docker run -it --rm -p 9696:9696 zoomcamp-model:hw10
```

Sending a request to the model with [q6_test.py](q6_test.py):
```bash
python3 q6_test.py
```

You should see this:

```python
{'get_credit': True, 'get_credit_probability': <value>}
```

Here `<value>` is the probability of getting a credit card. You need to choose the right one.

* 0.3269
* 0.5269
* 0.7269
* 0.9269


Outputs:
```yaml
{'get_credit': True, 'get_credit_probability': 0.726936946355423}
```

**Answer**: 0.7269

Now the docker-container has to be stopped.

## Installing `kubectl` and `kind`
* `kubectl` - https://kubernetes.io/docs/tasks/tools/ (you might already have it - check before installing)
* `kind` - https://kind.sigs.k8s.io/docs/user/quick-start/

To install both programs at once, use the following command:
```bash
python3 install_deps.py
```

## Question 2

What's the version of `kind` that you have? 

Use `kind --version` to find out.

**Answer**: `kind version 0.20.0`

## Creating a cluster
Now let's create a cluster with `kind`:

```bash
kind create cluster
```

## Question 3

Now let's test if everything works. Use `kubectl` to get the list of running services. 

What's `CLUSTER-IP` of the service that is already running there? 

**Answer**: `10.96.0.1`
- Used command: `kubectl get service`
- Cluster-IP of the kubernetes-service

## Question 4

To be able to use the docker image we previously created (`zoomcamp-model:hw10`),
we need to register it with `kind`.

What's the command we need to run for that?

* `kind create cluster`
* `kind build node-image`
* `kind load docker-image`
* `kubectl apply`

**Answer**: `kind load docker-image`

The full command to register the aforementioned docker-container is:
```bash
kind load docker-image zoomcamp-model:hw10
```

## Question 5

Now let's create a deployment config (e.g. `deployment.yaml`):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit
spec:
  selector:
    matchLabels:
      app: credit
  replicas: 1
  template:
    metadata:
      labels:
        app: credit
    spec:
      containers:
      - name: credit
        image: <Image>
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"            
          limits:
            memory: <Memory>
            cpu: <CPU>
        ports:
        - containerPort: <Port>
```

Replace `<Image>`, `<Memory>`, `<CPU>`, `<Port>` with the correct values.

What is the value for `<Port>`?

**Answer**: 
- Port: `9696` (used in the predict script [q6_predict.py](q6_predict.py)) inside the docker-container.
- The deployment-config with filled in values can be found [here](deployment.yaml).


## Question 6

Let's create a service for this deployment (`service.yaml`):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: <Service name>
spec:
  type: LoadBalancer
  selector:
    app: <???>
  ports:
  - port: 80
    targetPort: <PORT>
```

Fill it in. What do we need to write instead of `<???>`?

Apply this config file.

```bash
kubectl apply -f service.yaml
```

**Answer**: 
- `credit` is inserted instead of `<???>` (and `<Service name>`)
- The filled-in version of the config [service.yaml](service.yaml)

## Testing the service

We can test our service locally by forwarding the port 9696 on our computer 
to the port 80 on the service:

```bash
kubectl port-forward service/<Service name> 9696:80
```