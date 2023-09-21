# 2. Machine Learning for Regression

- 2.1 [Car price prediction project](##01-car-price-intro)
- 2.2 [Data preparation](#02-data-preparation)
- 2.3 [Exploratory data analysis](#03-eda)
- 2.4 [Setting up the validation framework](#04-validation-framework)
- 2.5 [Linear regression](#05-linear-regression-simple)
- 2.6 [Linear regression: vector form](#06-linear-regression-vector)
- 2.7 [Training linear regression: Normal equation](#07-linear-regression-training)
- 2.8 [Baseline model for car price prediction project](#08-baseline-model)
- 2.9 [Root mean squared error](#09-rmse)
- 2.10 [Using RMSE on validation data](#10-car-price-validation)
- 2.11 [Feature engineering](#11-feature-engineering)
- 2.12 [Categorical variables](#12-categorical-variables)
- 2.13 [Regularization](#13-regularization)
- 2.14 [Tuning the model](#14-tuning-model)
- 2.15 [Using the model](#15-using-model)
- 2.16 [Car price prediction project summary](#16-summary)
- 2.17 [Explore more](#17-explore-more)
- 2.18 [Homework](#homework)


<a id="01-car-price-intro"></a>
## 2.1 Car price prediction project

### Problem Description
- Develop model that helps the user to estimates the best price for a car

### Dataset Information
- Car-Dataset from Kaggle: [Link](https://www.kaggle.com/datasets/CooperUnion/cardataset)

### Project Plan

1. Prepare data and do EDA (Exploratory Data Analysis)
2. Use Linear Regression for price prediction
3. Understand the internals of Linear Regression
4. Evalue the model with RMSE (Root Mean Squared Error)
5. Feature Engineering
6. Regularization

### The Code
- Can be found [here](notebooks/carprice.ipynb)

<a id="02-data-preparation"></a>
## 2.2 Data preparation
- Section 2.2 in Notebook [here](notebooks/02-price-prediction.ipynb)

<a id="03-eda"></a>
## 2.3 Exploratory data analysis
- Section 2.3 in Notebook [here](notebooks/02-price-prediction.ipynb)


<a id="04-validation-framework"></a>
## 2.4 Setting up the validation framework

### Intro
- Splitting the obtained and transformed data into 3 subsets
    - **TRAIN** ($\approx 60\%$): $X_T, y_T$
    - **VAL** ($\approx 20\%$): $X_v, y_v$
    - **TEST** ($\approx 20\%$): $X_{Test}, y_{T
    est}$

### The Code
- Section 2.4 in Notebook [here](notebooks/02-price-prediction.ipynb)

<a id="05-linear-regression-simple"></a>
## 2.5 Linear regression

### What is Linear Regression
- Statistical method to best fit a line / plane / hyperplane to given features
- Best fitting model obtained by changing the regression parameters of the ML-Model s.t. the error between true target value and the predicted target value become smaller
- $g(x_i) \approx y_i \rightarrow g(x_{i1}, x_{i2}, ..., x_{in}) \approx y_i$
    - **Regression Model:** $g(\cdot)$
    - **Feature-Column:** $x_i = (x_{i1}, x_{i2}, ..., x_{in})\in\mathbb{R}^n$
        - One vector of dimension $n$
    - **Feature-Matrix:** $X \in \mathbb{R}^{m\times n}$
        - $m$ vectors of dimension $n$
    - **Target-Value:**
        - Scalar: $y_i\in\mathbb{R}$
        - Vector: $y\in\mathbb{R}^m$

### The Code
- Section 2.5 in Notebook [here](notebooks/02-price-prediction.ipynb)

<a id="06-linear-regression-vector"></a>
## 2.6 Linear regression: vector form

### The Code
- Section 2.6 in Notebook [here](notebooks/02-price-prediction.ipynb)

<a id="07-linear-regression-training"></a>
## 2.7 Training linear regression: Normal equation

### The Code
- Section 2.7 in Notebook [here](notebooks/02-price-prediction.ipynb)

### The Problem
- Normal Equation solves the Linear Least Squares (LLS) objective
    - Since the optimization problem is quadratic, there is an analytic solution
- Detailed explanation in Video fromat can be found [here](https://www.youtube.com/watch?v=NN7mBupK-8o)

### Detailed Derivation

$$\mathbf{X}\in\mathbb{R}^{m\times n},\mathbf{w}\in\mathbb{R}^{n},\mathbf{y}\in\mathbb{R}^{m}$$
$$\mathbf{w}^*=\arg\min_{\mathbf{w}}J=\arg\min_{\mathbf{w}}\frac{1}{2}\|\mathbf{X}\mathbf{w} - y\|^2$$

$$J = \frac{1}{2}\|\mathbf{X}\mathbf{w}-\mathbf{y}\|^2 = \frac{1}{2}(\mathbf{X}\mathbf{w}-\mathbf{y})^T(\mathbf{X}\mathbf{w}-\mathbf{y}) =
\frac{1}{2}(\mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w} -2\mathbf{w}^T\mathbf{X}^Ty + \mathbf{y}^T\mathbf{y})
$$

$$\nabla_{\mathbf{w}}J = \nabla_{\mathbf{w}}\frac{1}{2}(\mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w} -2\mathbf{w}^T\mathbf{X}^Ty + \mathbf{y}^T\mathbf{y}) = 0$$
$$\frac{1}{2}(2\mathbf{X}^T\mathbf{X}\mathbf{w}-2\mathbf{X}^T\mathbf{y}) = 0$$
$$\mathbf{X}^T\mathbf{X}\mathbf{y}-\mathbf{X}^T\mathbf{y} = 0$$
$$\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}$$
$$\Rightarrow \mathbf{w}^* = \left(\mathbf{X}^T\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{y}$$


<a id="08-baseline-model"></a>
## 2.8 Baseline model for car price prediction project

- Section 2.8 in Notebook [here](notebooks/02-price-prediction.ipynb)

<a id="09-rmse"></a>
## 2.9 Root mean squared error

- Section 2.9 in Notebook [here](notebooks/02-price-prediction.ipynb)

- Evaluation Metric for Regression Models
- Computes the square-root of the average squared error
    - $\text{RMSE}(y, \hat{y}) = \sqrt{\frac{1}{m}\sum_{i=1}^m (y_i - \hat y_i)^2}$

<a id="10-car-price-validation"></a>
## 2.10 Using RMSE on validation data

- Section 2.10 in Notebook [here](notebooks/02-price-prediction.ipynb)

- **Summary:** Evaluating the predictions from the validation-set with the RMSE-Metric from Section 2.9


<a id="11-feature-engineering"></a>
## 2.11 Feature engineering

- Section 2.11 in Notebook [here](notebooks/02-price-prediction.ipynb)

- **Summary:** Creating new features could help reduce the error of the model.


<a id="12-categorical-variables"></a>
## 2.12 Categorical variables

- Section 2.12 in Notebook [here](notebooks/02-price-prediction.ipynb)

- **Summary**:
    - Variables that can take on discrete values / category
    - Can typically be found in columns with strings, however numerical variables can also be categorical


<a id="13-regularization"></a>
## 2.13 Regularization

- Section 2.13 in Notebook [here](notebooks/02-price-prediction.ipynb)

- **Summary**:
    - Problem in Gram-Matrix $(X^TX)^{-1}$
        - Singular matrix, which is not invertible
        - Matrix usually invertible, because of small numerical errors in the matrix
        - Resulting parameter become very large
    - Regularization adds a value to the diagonal of the gram matrix, that influences the eigenvalues of the matrix s.t. the inversion process becomes more stable and steers away from poles. 
    - **Before:** 
        - $w = (X^TX)^{-1}X^Ty$
    - **After:** 
        - $w_r = (X^TX + rI)^{-1}X^Ty$

<a id="14-tuning-model"></a>
## 2.14 Tuning the model

- Section 2.14 in Notebook [here](notebooks/02-price-prediction.ipynb)

- **Summary**:
    - Finding the best parameters to reduce the evaluation metric as much as possible
    - Important step during most machine learning training-procedures

<a id="15-using-model"></a>
## 2.15 Using the model

- Section 2.15 in Notebook [here](notebooks/02-price-prediction.ipynb)

- **Summary**:
    - Using the "full"-dataset (train-data and validation-data) as one dataset
    - Evaluation done on test-dataset


<a id="16-summary"></a>
## 2.16 Car price prediction project summary

- Section 2.16 in Notebook [here](notebooks/02-price-prediction.ipynb)
- Detailed explanation [here](https://www.youtube.com/watch?v=_qI01YXbyro&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=26)

<a id="17-explore-more"></a>
## 2.17 Explore more

<a id="homework"></a>
## 2.18 Homework
