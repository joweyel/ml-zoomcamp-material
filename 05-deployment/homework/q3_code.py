import pickle

# Load model and DictVectorizer
def load(filaname):
    with open(filaname, "rb") as f:
        data = pickle.load(f)
    return data

model = load("model1.bin")
dv = load("dv.bin")

client = { "job": "retired", "duration": 445, "poutcome": "success" }

X = dv.transform([client])
y_pred = model.predict_proba(X)[0, 1]
print(f"churn-prob.: {y_pred:.6f}")