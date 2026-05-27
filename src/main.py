import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/superstore.csv")

# Convert order date into date format
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

# Sort values by date
df = df.sort_values("Order Date")

# Select required columns
df = df[["Order Date", "Sales"]]

# Group sales by date
df = df.groupby("Order Date").sum()

# Convert daily sales into monthly sales
monthly_sales = df.resample("ME").sum()

# Reset index
monthly_sales = monthly_sales.reset_index()

# Create month number
monthly_sales["Month_Number"] = range(len(monthly_sales))

# Prepare data
X = monthly_sales[["Month_Number"]]
y = monthly_sales["Sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict sales
predicted_sales = model.predict(X)

# Plot graph
plt.figure(figsize=(12,6))

plt.plot(monthly_sales["Order Date"], y, label="Actual Sales")
plt.plot(monthly_sales["Order Date"], predicted_sales, label="Predicted Sales")

plt.title("Sales Forecasting")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()

plt.show()