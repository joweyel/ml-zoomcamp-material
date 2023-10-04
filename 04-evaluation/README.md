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
- Table that is used to store different types of errors
- **There are two types of errors**:
    - <u>False Positive</u>: True (No churn) & Predicted (churn)
    - <u>False Negative</u>: True (churn) & Predicted (No churn)

- Visualization of the different possible predictions:

![conf-tree](imgs/confusion_matrix.png)

- In table form we get this (numbers come from this sections notebook!)

**Predictions**
|  True \ Predicted | $g(x_i) < t$  (Negative) | $g(x_i) \ge t$  (Positive) | 
| ----------------- | ------------------------ | -------------------------- |
| $y=0$ (Negative)  | **`TN`**: 922 (65%)      | **`FP`**: 101 (8%)         |
| $y=1$ (Positive)  | **`FN`**: 176 (12%)      | **`TP`**: 210 (15%)        | 

- We can derive the `Accuracy` metric from the values from the confusion matrix:
     - $\text{accuracy} = \frac{\text{tp} + \text{tn}}{\text{tn} + \text{fp} + \text{fn} + \text{tp}} = \frac{\text{tp} + \text{tn}}{\text{\#all samples}}$

### Conclusion
- The confusion matrix helps to analyze what type of error we make


<a id="04-precision-recall"></a>
## 4.4 Precision and Recall

The metrics of this section can be constructed from the entries of the confusion matrix (`tp`, `tn`, `fp`, `fn`).

### Precision Definition
- Fraction of positive predictions that are correct
- $\textbf{precision} = \frac{tp}{\text{\#positive}} = \frac{tp}{tp + fp}$
```python
# Computation example
true label:      [1] [0] [1] [1]
predicted churn: [1] [1] [1] [1]
--------------------------------
correct:         [1] [0] [1] [1] 
tp = 3, fp = 1
precision = tp / (tp + fp) = 3 / 4 = 0.75
```

## Recall Definition
- Fraction of correctly identified positive examples (churning users)
- $\textbf{recall} = \frac{tp}{tp + fn}$

```python
# Computation example           
                    < t                   >= t 
 true label:  [0] [0] [0] [1]   |   [1] [0] [1] [1]
 predictions: [0] [0] [0] [0]   |   [1] [1] [1] [1]                

correct (positive):       [0]       [1]     [1] [1]
                          fn        tp      tp  tp
tp = 3, fn = 1
recall = tp / (tp + fn) = 3 / 4 = 0.75

```

### Why Accuracy is misleading
We have the following results from the given data:
- $\text{precision} = \frac{tp}{tp + fp} = 67\%\quad (33\%)$
- $\text{recall} = \frac{tp}{tp + fn} = 54\%\quad (46\%)$
- $\text{accuracy} = \frac{tp + tn}{tp + tn + fp + fn} = 80\%\quad (20\%)$

**Insights from the 3 metrics**:
- Even though the accuracy is relatively high, the model is not that good. This can be caused by imbalanced data coupled with a ill-chosen decision-treshold at $\tau = 0$ or $\tau = 1$.
- To see if the model is good overall, you have to look at the `precision` and `recall` in conjunction with the `accuracy`.
    - `Precision`: Looking at all the customers, that we think are abount to churn (tp + fp) and what fraction of this is correctly classified (tp) as churning.
    - `Recall`: Looking at all the customers, that are actually churning (tp + fn) and what fraction of this is correctly classified (tp) as churning.

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
