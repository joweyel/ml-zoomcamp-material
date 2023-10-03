# 4. Evaluation Metrics for Classification

- 4.1 [Evaluation metrics: session overview](#01-overview)
- 4.2 [Accuracy and dummy model](#02-accuracy)
- 4.3 [Confusion table](#03-confusion-table)
- 4.4 [Precision and Recall](#04-precision-recall)
- 4.5 [ROC Curves](#05-roc)
- 4.6 [ROC AUC](#06-auc)
- 4.7 [Cross-Validation](#07-cross-validation)
- 4.8 [Summary](#08-summary)
- 4.9 [Explore more](#09-explore-more)
- 4.10 [Homework](#homework)


<a id="01-overview"></a>
## 4.1 Evaluation metrics: session overview

In [Week 3](../03-classification/README.md) we looked at the problem of `Churn Prediction`, where we built a model (Logistic Regression) for scoring existing customers and assigning probabilities of a customer leaving the company.

In this section we look into other ways to evaluate binary models.

The code of this section can be found in the accomanying notebook [here](notebooks/section4-notebook.ipynb).


<a id="02-accuracy"></a>
## 4.2 Accuracy and dummy model

### Accuracy calculation explanation (6 samples)
The accuracy is the number of correct predictions divided by all predictions. 
```python
predictions: [1] [1] [1] [1] [0] [0]
real labels: [1] [0] [0] [1] [0] [1]
-------------------------------------
correct?     [1] [0] [0] [1] [1] [0]
correct predcitions: 3
accuracy: 3 / 6 = 1 / 2 = 0.5
```

### Example for Accuracy on all customers
Here we look at the validation set, that has 1409 customers. Prediction is done (for now) with a decision threshold of $0.5$. This results in the following numbers:
```bash
all:     1409
correct: 1132
-------------
accuracy: 1132 / 1409 = 0.8034 (80%)
```

The code for this looks like this:
```python
y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
acc = (y_val == churn_decision).mean()
print(f"accuracy: {acc:.2f} | {acc*100:.2f}%")
```


### Is the used churn threshold good?
- The dscision threshold at $\tau = 0.5$ is not fixed but can be changed.
- Changing the threshold can change the accuracy (better or worse) and has to be tested with different values between $0$ and $1$.

### Accuracy metric from `scikit-learn`
- There is already a function in `scikit-learn` to compute the accuracy of a model, provided the predictions and true labels
```python
from sklearn.metrics import accuracy_score
score = accuracy_score(y, y_pred)
```

### Interpreting the first and last threshold ($\tau = 0$ and $\tau = 1$)
- $\tau = 1$: Here the threshold is set so high, that everything is classified as `False`
- $\tau = 0$: Here the threshold is set so low, that everything is classified as `True` 

### Why bother? Comparing the accuracy of our model to the dummy model
The model, that classifies every customer of non-churning ($\tau = 1$) still has an accuracy of about $73\%$. With this we now have 2 models:
| `Model`    | `Accuracy` |
| ---------- | ---------- |
|  Our Model |   $80\%$   |
| Dummy Model|   $73\%$   |

There is only a $7\%$ difference between the best model and a "deficient" model. The accuracy is all nice and good, however ins such situations it can be insufficient and other metrics have to be looked at. 


### Issues due to class imbalances in datasets
The relatively high accuracy in the dummy model is caused by class-imbalance. The distribution of the classes here is:
- **Non-Churning**: $\approx 73\%$
- **Churning**: $\approx 27\%$  

There are clearly more non-churning chustomers. This directly results in an accuracy of $73\%$ in when threshold is $\tau = 1.0$. With such an imbalance you can achieve a reasonably good accuracy while applying a brute force "everything is true" or "everything is false" at the $0$ and $1$ thresholds. To get a better insight into the performance of the trained model, other metrics and different types of errors are being considered, that are examined in the following sections.


<a id="03-confusion-table"></a>
## 4.3 Confusion table

<a id="04-precision-recall"></a>
## 4.4 Precision and Recall

<a id="05-roc"></a>
## 4.5 ROC Curves

<a id="06-auc"></a>
## 4.6 ROC AUC

<a id="07-cross-validation"></a>
## 4.7 Cross-Validation

<a id="08-summary"></a>
## 4.8 Summary

<a id="09-explore-more"></a>
## 4.9 Explore more

<a id="homework"></a>
## 4.10 Homework
