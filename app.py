import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load trained model and features
# -------------------------------
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# ✅ Correct reverse mapping (NUMBER → LABEL)
reverse_workout_map = {
    1: "Yoga",
    2: "HIIT",
    3: "Cardio",
    4: "Strength"
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Workout Prediction", page_icon="🏋️")
st.title("🏋️ Workout Type Prediction App")
st.write("Enter gym member details to predict workout type")

# -------------------------------
# User Inputs
# -------------------------------
with st.form("prediction_form"):
    age = st.number_input("Age", 18, 60)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", 140, 200)
    weight = st.number_input("Weight (kg)", 40, 120)
    bmi = st.number_input("BMI", 15.0, 40.0)
    fat = st.number_input("Fat Percentage", 5.0, 40.0)
    frequency = st.number_input("Workout Frequency (days/week)", 1, 7)
    experience = st.selectbox("Experience Level", [0, 1, 2])
    calories = st.number_input("Calories Burned", 100, 1000)
    heart_rate = st.number_input("Heart Rate (bpm)", 60, 200)
    duration = st.number_input("Session Duration (hours)", 0.5, 3.0)

    submitted = st.form_submit_button("Predict Workout Type")


# -------------------------------
# Manual Encoding
# -------------------------------
gender_encoded = 1 if gender == "Male" else 0

# -------------------------------
# Create Input DataFrame
# -------------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "Gender_encoded": gender_encoded,
    "Height (cm)": height,
    "Weight (kg)": weight,
    "BMI": bmi,
    "Fat_Percentage": fat,
    "Workout_Frequency (days/week)": frequency,
    "Experience_Level": experience,
    "Calories_Burned": calories,
    "Heart_Rate (bpm)": heart_rate,
    "Session_Duration (hours)": duration
}])

# -------------------------------
# Align features (CRITICAL)
# -------------------------------
input_data = input_data.reindex(columns=features, fill_value=0)

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    try:
        pred = model.predict(input_data.values)[0]
        workout = reverse_workout_map.get(pred, "Unknown")

        st.success(f"💪 Predicted Workout Type: **{workout}**")
        st.write("Sweat is your body's way of showing progress💪💪")
        st.write("🔢 Model output:", pred)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
