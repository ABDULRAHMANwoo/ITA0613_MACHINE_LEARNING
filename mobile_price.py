import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the dataset
dataset = pd.read_csv("C:/Users/12202/Downloads/HousePricePrediction.csv.xls")

# Explore the dataset
print(dataset.head())
print(dataset.describe())

# Preprocess the data
dataset.fillna(0, inplace=True)
dataset = dataset.dropna()

# Choose a machine learning algorithm
model = LogisticRegression()

# Train the model
model.fit(dataset.drop('price_range', axis=1), dataset['price_range'])

# Evaluate the model
predictions = model.predict(dataset.drop('price_range', axis=1))
print(accuracy_score(predictions, dataset['price_range']))

# Deploy the model to production
model.save('mobile_price_prediction_model.pkl')
