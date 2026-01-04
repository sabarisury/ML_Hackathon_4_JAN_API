from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# ---------- Load model + threshold ----------
with open("best_xgb_model_with_threshold.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    threshold = data["threshold"]

# ---------- FastAPI app ----------
app = FastAPI(
    title="HR Promotion Prediction API",
    description="Predict employee promotion using XGBoost",
    version="1.0"
)

# ---------- Input schema ----------
class EmployeeInput(BaseModel):
    department: str
    region: str
    education: str
    gender: str
    recruitment_channel: str
    no_of_trainings: int
    age: int
    previous_year_rating: int
    length_of_service: int
    KPIs_met_gt_80: int
    awards_won: int
    avg_training_score: int


# ---------- Health check ----------
@app.get("/")
def health():
    return {"status": "API is running"}


# ---------- Prediction endpoint ----------
@app.post("/predict")
def predict_promotion(data: EmployeeInput):

    input_df = pd.DataFrame([{
        "department": data.department,
        "region": data.region,
        "education": data.education,
        "gender": data.gender,
        "recruitment_channel": data.recruitment_channel,
        "no_of_trainings": data.no_of_trainings,
        "age": data.age,
        "previous_year_rating": data.previous_year_rating,
        "length_of_service": data.length_of_service,
        "KPIs_met >80%": data.KPIs_met_gt_80,
        "awards_won?": data.awards_won,
        "avg_training_score": data.avg_training_score
    }])

    prob = model.predict_proba(input_df)[:, 1][0]
    prediction = int(prob >= threshold)

    return {
        "promotion_probability": round(float(prob), 4),
        "threshold_used": threshold,
        "is_promoted": prediction
    }