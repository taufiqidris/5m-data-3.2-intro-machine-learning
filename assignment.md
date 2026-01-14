# Assignment

## Instructions

Complete the following exercises using Python.

1. Linear Regression Exercise:
   Using the California Housing dataset from scikit-learn, create a linear regression model to predict house prices.
   Evaluate the performance of Linear Regression on test set.

   ```python
   from sklearn.datasets import fetch_california_housing

   # Load dataset
   housing = fetch_california_housing()

   import numpy as np
   from sklearn.datasets import fetch_california_housing
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import mean_squared_error, r2_score

   # Load dataset
   housing = fetch_california_housing()
   X = housing.data
   y = housing.target

   # Train-test split
   X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
   )

   # Train Linear Regression model
   lr = LinearRegression()
   lr.fit(X_train, y_train)

   # Predictions
   y_pred = lr.predict(X_test)

   # Evaluation metrics
   mse = mean_squared_error(y_test, y_pred)
   rmse = np.sqrt(mse)
   r2 = r2_score(y_test, y_pred)

   print("Linear Regression Performance (Test Set)")
   print(f"MSE:  {mse:.4f}")
   print(f"RMSE: {rmse:.4f}")
   print(f"R^2:  {r2:.4f}")
   ```

### Linear Regression Results

The Linear Regression model was evaluated on the test set using mean squared error (MSE), root mean squared error (RMSE), and R-squared. The model achieved an RMSE of 0.7456, indicating that the predicted house prices differ from the actual values by approximately 0.75 units on average.

The R-squared value of 0.5758 shows that about 57.6% of the variance in house prices is explained by the model. This suggests that Linear Regression captures the general relationship between the features and house prices, but a substantial proportion of the variance remains unexplained. More complex models may further improve predictive performance.


2. Classification Exercise:
   Using the breast cancer dataset from scikit-learn, build classification models to predict malignant vs benign tumors.
   Compare Logistic Regression and KNN performance on test set.

   ```python
   from sklearn.datasets import load_breast_cancer

   # Load dataset
   cancer = load_breast_cancer()

   from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

   ```
### Classification Results Comparison

Both Logistic Regression and K-Nearest Neighbors (KNN) achieved high accuracy on the breast cancer classification task. Logistic Regression achieved an accuracy of 97.37%, outperforming KNN, which achieved an accuracy of 94.74%.

Logistic Regression demonstrates strong and balanced performance across both classes, with high precision and recall values. In particular, it achieves a recall of 0.99 for malignant tumors, which is important in a medical context where minimizing false negatives is critical.

While KNN also performs well, its overall precision and recall are slightly lower than those of Logistic Regression. Additionally, KNN is more sensitive to feature scaling and the choice of k. Overall, Logistic Regression provides better performance and greater interpretability for this classification task.

## Submission

- Submit the URL of the GitHub Repository that contains your work to NTU black board.
- Should you reference the work of your classmate(s) or online resources, give them credit by adding either the name of your classmate or URL.
