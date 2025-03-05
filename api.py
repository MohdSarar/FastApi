from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# Load the ML model
model = joblib.load("LightGBM_best_model.pkl")

# Create FastAPI instance
app = FastAPI()

# Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route
@app.get("/")
def home():
    return {"message": "API de prédiction des admissions IRA"}

# Prediction route
@app.post("/predict/")
def predict(data: dict):
    try:
        print("🔍 Received Data:", data)  # Debugging

        # Convert input data into DataFrame
        df = pd.DataFrame([data])
        print("📝 DataFrame for Prediction:\n", df)

        # Make prediction
        prediction = model.predict(df)
        print("✅ Prediction Result:", prediction[0])

        return {"prediction_ira": int(prediction[0])}
   
    except Exception as e:
        print("❌ Error in Prediction:", str(e))
        return {"error": str(e)}
