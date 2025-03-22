import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
    "Size": [1500, 1800, 2400, 3000, 3500, 4000, 4500],
    "Bedrooms": [3, 4, 4, 5, 5, 6, 7],
    "Age": [10, 8, 5, 3, 2, 1, 1],
    "Price": [300000, 350000, 400000, 450000, 500000, 600000, 650000]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Size", "Bedrooms", "Age"]]
y = df["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "house_price_model.pkl")

print("✅ Model saved successfully as 'house_price_model.pkl'")
