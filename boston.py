import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Create a synthetic dataset
np.random.seed(42)
n_samples = 1000
age = np.random.randint(1, 15, size=n_samples)
mileage = np.random.randint(5000, 200000, size=n_samples)
horsepower = np.random.randint(70, 250, size=n_samples)
price = 20000 - (age * 1000) - (mileage * 0.02) + (horsepower * 100) + np.random.randint(-2000, 2000, size=n_samples)

# Create a DataFrame
df = pd.DataFrame({
    'Age': age,
    'Mileage': mileage,
    'Horsepower': horsepower,
    'Price': price
})


# Define features and target
X = df.drop('Price', axis=1)
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create the model
model = DecisionTreeRegressor()

# Train the model
model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")




# Create the model with tuned hyperparameters
model = DecisionTreeRegressor(max_depth=10, min_samples_split=5)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error after tuning: {mae}")
