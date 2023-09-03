import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report

# Read the Iris dataset
file_path = "C:/IRIS.csv"
iris = pd.read_csv("C:/Users/12202/Downloads/IRIS.csv.xls")

# Display the first few rows of the dataset
print(iris.head())

# Prepare the feature matrix x and target vector y
x = iris.drop("species", axis=1)
y = iris["species"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create instances of the classifiers
decisiontree = DecisionTreeClassifier()
logisticregression = LogisticRegression()
knearestclassifier = KNeighborsClassifier()
bernoulli_naiveBayes = BernoulliNB()
passiveAggressive = PassiveAggressiveClassifier()

# Fit each classifier on the training data
knearestclassifier.fit(x_train, y_train)
decisiontree.fit(x_train, y_train)
logisticregression.fit(x_train, y_train)
passiveAggressive.fit(x_train, y_train)

# Create a dictionary containing the names of the classifiers and their corresponding accuracy scores
data1 = {
    "Classification Algorithms": [
        "KNN Classifier",
        "Decision Tree Classifier",
        "Logistic Regression",
        "Passive Aggressive Classifier"
    ],
    "Score": [
        knearestclassifier.score(x_test, y_test),
        decisiontree.score(x_test, y_test),
        logisticregression.score(x_test, y_test),
        passiveAggressive.score(x_test, y_test)
    ]
}

# Create a pandas DataFrame from the data dictionary
score = pd.DataFrame(data1)

# Print the DataFrame with the accuracy scores
print(score)
