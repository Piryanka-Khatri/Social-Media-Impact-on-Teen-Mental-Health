from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

app = FastAPI(title="Mental Health Prediction API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (use specific origin in production)
    allow_credentials=True,
    allow_methods=["*"],        # Allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],        # Allow Content-Type, Authorization, etc.
)


class InputData(BaseModel):
    age: int
    daily_social_media_hours: float
    sleep_hours: float
    screen_time_before_sleep: float
    academic_performance: float
    physical_activity: float
    gender: str                      # male / female
    platform_usage: str              # Instagram / TikTok / both
    social_interaction_level: str    # low / medium / high


def encode_gender(gender):
    return 1 if gender.lower() == "male" else 0


def encode_platform(platform):
    platform = platform.lower()
    if platform == "instagram":
        return 1, 0
    elif platform == "tiktok":
        return 0, 1
    elif platform == "both":
        return 1, 1
    else:
        raise ValueError("Invalid platform_usage")


def encode_social(level):
    level = level.lower()
    if level == "low":
        return 1, 0
    elif level == "medium":
        return 0, 1
    elif level == "high":
        return 0, 0
    else:
        raise ValueError("Invalid social_interaction_level")


@app.get("/")
def home():
    return {
        "message": "API is running 🚀",
        "go_to": "/docs to test the model"
    }


@app.post("/predict")
def predict(data: InputData):
    try:
        gender_male = encode_gender(data.gender)
        insta, tiktok = encode_platform(data.platform_usage)
        low, medium = encode_social(data.social_interaction_level)

        # input_list = [
        #     data.age,
        #     data.daily_social_media_hours,
        #     data.sleep_hours,
        #     data.screen_time_before_sleep,
        #     data.academic_performance,
        #     data.physical_activity,
        #     gender_male,
        #     insta,
        #     tiktok,
        #     low,
        #     medium
        # ]

        input_dict = {
            "age": data.age,
            "daily_social_media_hours": data.daily_social_media_hours,
            "sleep_hours": data.sleep_hours,
            "screen_time_before_sleep": data.screen_time_before_sleep,
            "academic_performance": data.academic_performance,
            "physical_activity": data.physical_activity,
            "gender_male": gender_male,
            "platform_usage_Instagram": insta,
            "platform_usage_TikTok": tiktok,
            "social_interaction_level_low": low,
            "social_interaction_level_medium": medium
        }

        # x = np.array(input_list).reshape(1, -1)
        x = np.array([input_dict[col] for col in features]).reshape(1, -1)
        x_scaled = scaler.transform(x)

        pred = model.predict(x_scaled)[0]
        prob = model.predict_proba(x_scaled)[0].tolist()

        return {
            "prediction": int(pred),
            "probability": prob
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
