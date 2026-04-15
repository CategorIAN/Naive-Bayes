# [Naive Bayes Model](https://github.com/CategorIAN/Naive-Bayes/blob/main/NaiveBayes.py)

## Overview

This project uses a **Naive Bayes classifier** to perform classification on structured datasets.  
The implementation combines:

- Discretization of numeric features via **binning**
- **Laplace smoothing** to handle sparse observations
- A **functional programming design** for clarity and modularity

The result is a simple, interpretable, and efficient probabilistic model.

---

## The Core Idea

Naive Bayes is based on the following relationship:

$$
P(Y \mid X) \propto P(Y) \times \prod_j P(X_j \mid Y)
$$

Where:

- $Y$ is the class label  
- $X = (X_1, ..., X_d)$ are the features  
- Each feature is assumed **conditionally independent given the class**

This assumption is rarely true in practice, but it allows the model to scale efficiently and often performs surprisingly well.

---

## Handling Continuous Features

Many datasets contain continuous values, but Naive Bayes in this implementation operates on **discrete values**.

To address this, each numeric feature is divided into `n` bins:

- Numeric values are partitioned into intervals  
- Each value is mapped to a bin index

This transformation allows the model to treat all features uniformly as categorical variables.

---

## Training the Model

### 1. Class Probabilities

The model first estimates the probability of each class $c$:
$$
P(Y = c)
$$
by counting how frequently each class appears in the dataset.

---

### 2. Feature Likelihoods

For each feature $X_j$, the model counts occurrences of:

(Class $c$, Feature Value $v$)

These counts are used to estimate:
$$
P(X_j = v | Y = c)
$$
---

## Laplace Smoothing

To avoid zero probabilities (which would eliminate a class entirely during prediction), the model uses **Laplace smoothing**:
$$
P = \frac{\text{count} + \alpha}{\text{total} + \alpha \times K}
$$

Where:

- $\alpha$ is the smoothing parameter  
- $K$ is the number of possible values for the feature  

This ensures that even unseen feature values receive a small, non-zero probability.

---

## Training Code
In summary, the model is trained using the code below:
```python
@dataclass(frozen=True)
class NaiveBayes:
    n: int # Bin Size
    alpha: float # Smoothing Parameter

    ...

    def __call__(self, data: Dataset) -> Callable[[pd.Series], object]:
        binner = self.bin_map(data) # Create binning function.
        binned_data = self.binned(binner, data) # Get binned data.
        Q = self.getQ(binned_data) # Get class counts.
        count_map = self.multi_count(binned_data) # Get class and feature value counts.
        return partial(self.predict, binner, Q, count_map) # Function that inputs features and outputs predicted class
```
where `n` and `alpha` are our hyperparameters.

## Hyperparameters
### Bin Size (n)
- Controls how numeric features are discretized
- Larger values $\rightarrow$ finer resolution
- Smaller values $\rightarrow$ more general groupings

### Smoothing Parameter
- Prevents zero probabilities
- Larger values increase smoothing
- Smaller values rely more on observed data

## Prediction

Given a new data point $X$, the model:

1. Applies the same binning transformation  
2. Computes the probability of each class $c$:
$$
P(Y=c)\times \prod P(X_j \mid Y = c)
$$
3. Selects the class with the highest probability

The model is implemented in a functional style:
```python
model = NaiveBayes(bin_size, alpha)
classifier = model(training_data)
prediction = classifier(x)
```



