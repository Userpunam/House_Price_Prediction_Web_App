from django.db import models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Create your models here.

# Load dataset
data = pd.read_csv('path_to_your_dataset.csv')

# Select features and target variable
X = data[['feature1', 'feature2', 'feature3']]  # replace with actual features
y = data['target']  # replace with the actual target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Save the model
with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file)
