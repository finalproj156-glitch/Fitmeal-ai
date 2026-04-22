from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل ملف الوجبات
with open("weekly_plan.pkl", "rb") as f:
    weekly_plan = pickle.load(f)

if not isinstance(weekly_plan, pd.DataFrame):
    try:
        weekly_plan = pd.DataFrame(weekly_plan)
    except Exception:
        weekly_plan = pd.DataFrame()


class UserInput(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    activity: str
    meals_per_day: int = 3
    goal: str | None = None


def calculate_bmi(weight: float, height_cm: float) -> float:
    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)
    return round(bmi, 2)


def calculate_tdee(weight: float, height: float, age: int, gender: str, activity: str) -> float:
    gender = gender.lower().strip()
    activity = activity.lower().strip()

    if gender == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
        "very active": 1.9
    }

    factor = activity_factors.get(activity, 1.55)
    tdee = bmr * factor
    return round(tdee, 2)


def determine_goal(user_goal: str | None, bmi: float) -> str:
    if user_goal:
        g = user_goal.lower().strip()
        if g in ["muscle_gain", "gain", "weight_gain", "bulk"]:
            return "Muscle_Gain"
        if g in ["fat_loss", "loss", "weight_loss", "cut"]:
            return "Fat_Loss"
        if g in ["maintenance", "maintain"]:
            return "Maintenance"

    if bmi < 18.5:
        return "Muscle_Gain"
    elif bmi >= 25:
        return "Fat_Loss"
    else:
        return "Maintenance"


def macro_targets(tdee: float, goal: str) -> dict:
    if goal == "Muscle_Gain":
        calories = tdee + 300
        protein = round((calories * 0.30) / 4, 1)
        fats = round((calories * 0.30) / 9, 1)
        carbs = round((calories * 0.40) / 4, 1)
    elif goal == "Fat_Loss":
        calories = tdee - 400
        protein = round((calories * 0.35) / 4, 1)
        fats = round((calories * 0.25) / 9, 1)
        carbs = round((calories * 0.40) / 4, 1)
    else:
        calories = tdee
        protein = round((calories * 0.30) / 4, 1)
        fats = round((calories * 0.25) / 9, 1)
        carbs = round((calories * 0.45) / 4, 1)

    return {
        "target_kcal": round(calories, 1),
        "protein_g_target": protein,
        "fat_g_target": fats,
        "carb_g_target": carbs
    }


def build_weekly_response(df: pd.DataFrame, meals_per_day: int) -> list:
    if df.empty:
        return []

    result = []
    working_df = df.copy()

    column_map = {}
    for col in working_df.columns:
        c = col.strip().lower()
        if c == "day":
            column_map[col] = "Day"
        elif c == "meal_type":
            column_map[col] = "Meal_Type"
        elif c == "meal_name":
            column_map[col] = "Meal_Name"
        elif c == "servings":
            column_map[col] = "Servings"
        elif c == "target_kcal":
            column_map[col] = "Target_kcal"
        elif c == "est_kcal":
            column_map[col] = "Est_kcal"

    working_df = working_df.rename(columns=column_map)

    if "Day" in working_df.columns:
        working_df = working_df.sort_values(by="Day")

    if "Meal_Type" in working_df.columns:
        if meals_per_day == 3:
            allowed = ["Breakfast", "Lunch", "Dinner"]
        else:
            allowed = ["Breakfast", "Snack", "Lunch", "Dinner"]
        working_df = working_df[working_df["Meal_Type"].isin(allowed)]

    for _, row in working_df.iterrows():
        item = {}
        for col in working_df.columns:
            value = row[col]
            if pd.isna(value):
                item[col] = None
            else:
                item[col] = value.item() if hasattr(value, "item") else value
        result.append(item)

    return result


@app.get("/")
def home():
    return {"message": "FitMeal API working ✅"}


@app.post("/plan")
def get_plan(data: UserInput):
    bmi = calculate_bmi(data.weight, data.height)
    tdee = calculate_tdee(data.weight, data.height, data.age, data.gender, data.activity)
    goal = determine_goal(data.goal, bmi)
    targets = macro_targets(tdee, goal)

    response = {
        "tdee": tdee,
        "goal": goal,
        "target_calories": targets["target_kcal"],
        "protein_g_target": targets["protein_g_target"],
        "fat_g_target": targets["fat_g_target"],
        "carb_g_target": targets["carb_g_target"],
        "bmi": bmi,
        "user_daily_needs": {
            "Age": data.age,
            "Gender": data.gender,
            "Weight_kg": data.weight,
            "Height_cm": data.height,
            "BMI": bmi,
            "Activity": data.activity,
            "Goal": goal,
            "TDEE_kcal": tdee,
            "Protein_g_target": targets["protein_g_target"],
            "Fat_g_target": targets["fat_g_target"],
            "Carb_g_target": targets["carb_g_target"],
            "Meals_per_day": data.meals_per_day
        },
        "weekly_plan": build_weekly_response(weekly_plan, data.meals_per_day),
        "drinks": [
            "Water (aim for 2-3L/day)",
            "Green Tea (optional, no sugar)",
            "Laban (good post-meal)"
        ]
    }

    return response