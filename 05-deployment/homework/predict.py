import pickle
from flask import Flask
from flask import request, jsonify


def load(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

model = load("model1.bin")
dv = load("dv.bin")

app = Flask("credit")

@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    decision = y_pred >= 0.5

    result = {
        "credit_probability": y_pred,
        "card_decision": bool(decision)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)