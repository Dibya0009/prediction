# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv(r"C:\Users\dibya\Downloads\tips.csv")

st.title("Interactive Tip Prediction")

total_bill = st.number_input("Total bill", 0.0, 500.0, 20.0)
sex = st.selectbox("Sex", ["Female", "Male"])
smoker = st.selectbox("Smoker", ["No", "Yes"])
day = st.selectbox("Day", ["Thur","Fri","Sat","Sun"])
time = st.selectbox("Time", ["Lunch","Dinner"])
size = st.number_input("Party size", 1, 20, 2)

# Convert to numeric
sex_val = 1 if sex=="Male" else 0
smoker_val = 1 if smoker=="Yes" else 0
day_val = ["Thur","Fri","Sat","Sun"].index(day)
time_val = 1 if time=="Dinner" else 0

features = np.array([[total_bill, sex_val, smoker_val, day_val, time_val, size]])

# Train dummy model (replace with your trained model)
model = LinearRegression()
X_train = np.array([[24.5,1,0,0,1,4],[30,0,1,1,0,2]])
y_train = np.array([5,6.5])
model.fit(X_train, y_train)

predicted_tip = model.predict(features)[0]

st.write(f"Predicted tip: {predicted_tip:.2f}")
