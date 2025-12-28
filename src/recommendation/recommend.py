import joblib
import pandas as pd
from src.recommendation.risk_rules import assign_risk

model = joblib.load("models/random_forest.pkl")

def recommend_action(input_data):
    df = pd.DataFrame([input_data])
    prob = model.predict_proba(df)[0][1]

    risk, action = assign_risk(prob)

    return {
        "default_probability": round(prob, 3),
        "risk_level": risk,
        "recommended_action": action
    }
