from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== FILE PATHS ==================
PATH_MEALS = "fitmeal_meals_ready_tags.csv"
PATH_SNACKS = "fitmeal_snacks_ready_v2.csv"
PATH_DRINKS = "fitmeal_drinks_ready_v2.csv"

# ================== INPUT MODEL ==================
class UserInput(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    activity: str
    meals_per_day: int

# ================== FUNCTIONS ==================
def calculate_tdee(age, gender, height_cm, weight_kg, activity):
    g = gender.lower()
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if g in ["male", "m"] else -161)
    factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }
    return bmr * factors.get(activity.lower(), 1.55)

def infer_goal_from_bmi(bmi):
    if bmi < 18.5:
        return "Muscle_Gain"
    elif bmi >= 25:
        return "Weight_Loss"
    return "Maintenance"

def macro_targets(tdee, goal, weight):
    if goal == "Weight_Loss":
        protein = 1.8 * weight
    elif goal == "Muscle_Gain":
        protein = 1.8 * weight
    else:
        protein = 1.6 * weight

    return {
        "protein_g_target": round(protein, 1),
        "fat_g_target": round((0.3 * tdee) / 9, 1),
        "carb_g_target": round((0.4 * tdee) / 4, 1),
    }

def load_meals():
    return pd.read_csv(PATH_MEALS)

def load_snacks():
    return pd.read_csv(PATH_SNACKS)

def load_drinks():
    return pd.read_csv(PATH_DRINKS)

def simple_plan(meals_df):
    return meals_df.head(7)

# ================== API ==================
@app.get("/")
def home():
    return {"message": "FitMeal API working ✅"}

@app.post("/plan")
def get_plan(data: UserInput):

    bmi = data.weight / ((data.height / 100) ** 2)
    goal = infer_goal_from_bmi(bmi)

    tdee = calculate_tdee(
        data.age,
        data.gender,
        data.height,
        data.weight,
        data.activity
    )

    targets = macro_targets(tdee, goal, data.weight)

    meals_df = load_meals()
    snacks_df = load_snacks()
    drinks_df = load_drinks()

    plan = simple_plan(meals_df)

    return {
        "bmi": round(bmi, 2),
        "goal": goal,
        "tdee": round(tdee, 1),

        "user_daily_needs": {
            "Age": data.age,
            "Gender": data.gender,
            "Weight_kg": data.weight,
            "Height_cm": data.height,
            "BMI": round(bmi, 2),
            "Activity": data.activity,
            "Goal": goal,
            "TDEE_kcal": round(tdee, 1),
            "Protein_g_target": targets["protein_g_target"],
            "Fat_g_target": targets["fat_g_target"],
            "Carb_g_target": targets["carb_g_target"],
            "Meals_per_day": data.meals_per_day
        },

        "weekly_plan": plan.to_dict(orient="records"),

        "drinks": [
            "Water (2-3L/day)",
            "Green Tea",
            "Laban"
        ]
    }