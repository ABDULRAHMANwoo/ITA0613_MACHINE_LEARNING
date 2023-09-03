import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data with a polynomial relationship
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = 2 * X**2 - 3 * X + 1 + np.random.randn(100, 1)

# Plot the data
plt.scatter(X, y, s=10)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Data with Polynomial Relationship')
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test, y_pred_linear)
print(f"Linear Regression Mean Squared Error: {linear_mse:.2f}")

# Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)
poly_mse = mean_squared_error(y_test, y_pred_poly)
print(f"Polynomial Regression Mean Squared Error: {poly_mse:.2f}")

# Plot the results
plt.scatter(X, y, s=10, label='Data')
plt.plot(X_test, y_pred_linear, color='red', label='Linear Regression')
plt.plot(X_test, y_pred_poly, color='green', label='Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear vs Polynomial Regression')
plt.show()
