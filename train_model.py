import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("data/tips.csv")

# Encode categorical features
data["sex"] = data["sex"].map({"Male": 1, "Female": 0})
data["smoker"] = data["smoker"].map({"Yes": 1, "No": 0})
data["time"] = data["time"].map({"Dinner": 1, "Lunch": 0})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})

# Features and target
X = data[["total_bill", "sex", "smoker", "day", "time", "size"]]
y = data["tip"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("model/tip_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved successfully!")
