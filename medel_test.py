import joblib
import pandas as pd

# Load the model
model = joblib.load("LightGBM_best_model.pkl")
print(model)


# Print model parameters
print("Model Parameters:")
print(model.get_params())

# Print feature importance (if available)
if hasattr(model, "feature_importances_"):
    print("Feature Importances:")
    print(model.feature_importances_)
else:
    print("No feature importance found. The model might not be trained.")


# Create a test dataframe with different values
test_data = pd.DataFrame([{
  "avg_temperature_avg": 5.0,
  "avg_temperature_max": 2.2,
  "avg_temperature_min": 1.5,
  "average_No_2": 20.0,
  "avg_pressure": 705,
  "avg_wind": 17.0,
  "year": 2025,
  "average_So_2": 3.5,
  "indice": 30,
  "average_Co": 0.6,
  "SYM3": 10,
  "avg_precipitation": 4.2,
  "SYM34": 10,
  "SYM23": 5,
  "average_IQA_global": 82,
  "SYM8": 41,
  "SYM6": 32,
  "week": 5,
  "average_Pm_2_5": 15.0,
  "SYM19": 52,
  "SYM39": 35,
  "SYM26": 30,
  "SYM22": 29,
  "SYM68": 48,
  "average_Pm_10": 25.5,
  "SYM32": 33,
  "SYM5": 38,
  "SYM24": 47
}])

# Get prediction
prediction = model.predict(test_data)
print(f"Prediction: {prediction[0]}")




# Generate different test inputs
test_data1 = pd.DataFrame([{
    "avg_temperature_avg": 5.0, "avg_temperature_max": 10.2, "avg_temperature_min": 2.5,
    "average_No_2": 20.0, "avg_pressure": 1005, "avg_wind": 12.0, "year": 2025,
    "average_So_2": 3.5, "indice": 60, "average_Co": 0.6, "SYM3": 50, "avg_precipitation": 4.2,
    "SYM34": 40, "SYM23": 55, "average_IQA_global": 82, "SYM8": 41, "SYM6": 32, "week": 5,
    "average_Pm_2_5": 15.0, "SYM19": 52, "SYM39": 35, "SYM26": 30, "SYM22": 29,
    "SYM68": 48, "average_Pm_10": 25.5, "SYM32": 33, "SYM5": 38, "SYM24": 47
}])

test_data2 = pd.DataFrame([{
    "avg_temperature_avg": 15.0, "avg_temperature_max": 20.5, "avg_temperature_min": 10.8,
    "average_No_2": 10.0, "avg_pressure": 1020, "avg_wind": 5.0, "year": 2024,
    "average_So_2": 1.5, "indice": 40, "average_Co": 0.4, "SYM3": 30, "avg_precipitation": 2.1,
    "SYM34": 25, "SYM23": 35, "average_IQA_global": 60, "SYM8": 20, "SYM6": 15, "week": 15,
    "average_Pm_2_5": 10.0, "SYM19": 30, "SYM39": 20, "SYM26": 15, "SYM22": 18,
    "SYM68": 30, "average_Pm_10": 15.2, "SYM32": 20, "SYM5": 22, "SYM24": 30
}])

# Get predictions
pred1 = model.predict(test_data1)
pred2 = model.predict(test_data2)

print(f"Prediction 1: {pred1[0]}")
print(f"Prediction 2: {pred2[0]}")