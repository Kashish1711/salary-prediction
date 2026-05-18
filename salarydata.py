# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 2. Load dataset
df = pd.read_csv("Salary_dataset.csv")

print(df.head())

# 3. Split data
X = df[['YearsExperience']]   # independent variable
y = df['Salary']              # dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Model parameters
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# 7. Evaluation metrics
print("\n--- Model Evaluation ---")
print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score:", metrics.r2_score(y_test, y_pred))

# 8. Visualization (Training Data)
plt.scatter(X_train, y_train)
plt.plot(X_train, model.predict(X_train))
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# 9. Visualization (Test Data)
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# # Import libraries
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# # Load dataset
# data = pd.read_csv("Salary_dataset.csv")

# # Check missing values
# print(data.isnull().sum())

# # Handle missing values (fill with mean)
# data['YearsExperience'].fillna(data['YearsExperience'].mean(), inplace=True)
# data['Salary'].fillna(data['Salary'].mean(), inplace=True)

# # Separate input and output
# X = data[['YearsExperience']]
# y = data['Salary']

# # Train-test split (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Create model
# model = LinearRegression()

# # Train model
# model.fit(X_train, y_train)

# # Predict on test data
# y_pred = model.predict(X_test)

# # Print equation
# print("Slope (m):", model.coef_[0])
# print("Intercept (c):", model.intercept_)

# # Plot graph (Test Data)
# plt.scatter(X_test, y_test)
# plt.plot(X_test, y_pred)
# plt.xlabel("Experience")
# plt.ylabel("Salary")
# plt.title("Salary Prediction (Test Data)")
# plt.show()


