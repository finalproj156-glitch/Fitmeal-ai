import joblib
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
MODEL_PATH = "fitmeal_best_energy_model.pkl"

# ================== LOAD MODEL ==================
model_package = joblib.load(MODEL_PATH)
energy_model = model_package["model"]
model_name = model_package.get("model_name", "Unknown Model")
feature_columns = model_package.get("feature_columns", None)

# ================== INPUT MODEL ==================
class UserInput(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    activity: str
    meals_per_day: int

# ================== BASIC FUNCTIONS ==================
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


def safe_number(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

# ================== MODEL PREDICTION ==================
def predict_energy_balance(data, bmi, tdee, targets):
    """
    This function tries to use the saved ML model.
    If the model needs different feature columns, it will not stop the API.
    """

    try:
        base_features = {
            "age": data.age,
            "weight": data.weight,
            "weight_kg": data.weight,
            "height": data.height,
            "height_cm": data.height,
            "bmi": bmi,
            "BMI": bmi,
            "tdee": tdee,
            "TDEE": tdee,
            "TDEE_kcal": tdee,
            "protein_g_target": targets["protein_g_target"],
            "fat_g_target": targets["fat_g_target"],
            "carb_g_target": targets["carb_g_target"],
            "meals_per_day": data.meals_per_day,
            "gender_male": 1 if data.gender.lower() in ["male", "m"] else 0,
            "gender_female": 1 if data.gender.lower() in ["female", "f"] else 0,
            "activity_sedentary": 1 if data.activity.lower() == "sedentary" else 0,
            "activity_light": 1 if data.activity.lower() == "light" else 0,
            "activity_moderate": 1 if data.activity.lower() == "moderate" else 0,
            "activity_active": 1 if data.activity.lower() == "active" else 0,
            "activity_very_active": 1 if data.activity.lower() == "very_active" else 0,
        }

        if feature_columns:
            input_df = pd.DataFrame([{
                col: base_features.get(col, 0)
                for col in feature_columns
            }])
        else:
            input_df = pd.DataFrame([base_features])

        prediction = energy_model.predict(input_df)

        if isinstance(prediction, np.ndarray):
            prediction = prediction.ravel()[0]

        probabilities = None

        if hasattr(energy_model, "predict_proba"):
            try:
                proba = energy_model.predict_proba(input_df)[0]
                classes = list(energy_model.classes_) if hasattr(energy_model, "classes_") else []
                probabilities = {
                    str(classes[i]): round(float(proba[i]), 4)
                    for i in range(len(proba))
                }
            except Exception:
                probabilities = None

        return str(prediction), probabilities, None

    except Exception as e:
        return "Model prediction unavailable", None, str(e)

# ================== MEAL PLAN FUNCTIONS ==================
def pick_meal(df, goal, meal_type, target_kcal, used_names):
    x = df.copy()

    if "Goal" in x.columns:
        goal_filtered = x[x["Goal"].astype(str).str.lower() == goal.lower()]
        if not goal_filtered.empty:
            x = goal_filtered

    if "Meal_Type" in x.columns:
        type_filtered = x[x["Meal_Type"].astype(str).str.lower() == meal_type.lower()]
        if not type_filtered.empty:
            x = type_filtered

    if "Meal_Name" in x.columns:
        available = x[~x["Meal_Name"].astype(str).isin(used_names)]
        if not available.empty:
            x = available

    x["Calories"] = pd.to_numeric(x["Calories"], errors="coerce").fillna(0)
    x["calorie_diff"] = (x["Calories"] - target_kcal).abs()

    if x.empty:
        return None

    pick = x.sort_values("calorie_diff").iloc[0]
    used_names.add(str(pick.get("Meal_Name", "")))
    return pick


def pick_snack(snacks_df, goal, target_kcal, used_snacks):
    x = snacks_df.copy()

    if "Goal" in x.columns:
        goal_filtered = x[x["Goal"].astype(str).str.lower() == goal.lower()]
        if not goal_filtered.empty:
            x = goal_filtered

    snack_name_col = "Snack_Name" if "Snack_Name" in x.columns else "Meal_Name"

    if snack_name_col in x.columns:
        available = x[~x[snack_name_col].astype(str).isin(used_snacks)]
        if not available.empty:
            x = available

    x["Calories"] = pd.to_numeric(x["Calories"], errors="coerce").fillna(0)
    x["calorie_diff"] = (x["Calories"] - target_kcal).abs()

    if x.empty:
        return None

    pick = x.sort_values("calorie_diff").iloc[0]
    used_snacks.add(str(pick.get(snack_name_col, "")))
    return pick


def build_weekly_plan(meals_df, snacks_df, goal, tdee, meals_per_day):
    if meals_per_day == 4:
        meal_order = ["Breakfast", "Lunch", "Dinner", "Snack"]
        meal_shares = {
            "Breakfast": 0.25,
            "Lunch": 0.35,
            "Dinner": 0.30,
            "Snack": 0.10,
        }
    else:
        meal_order = ["Breakfast", "Lunch", "Dinner"]
        meal_shares = {
            "Breakfast": 0.30,
            "Lunch": 0.40,
            "Dinner": 0.30,
        }

    rows = []
    used_names = set()
    used_snacks = set()

    for day in range(1, 8):
        for meal_type in meal_order:
            target_kcal = tdee * meal_shares[meal_type]

            if meal_type == "Snack":
                pick = pick_snack(snacks_df, goal, target_kcal, used_snacks)
                if pick is None:
                    continue
                meal_name = pick.get("Snack_Name", pick.get("Meal_Name", "Snack"))
            else:
                pick = pick_meal(meals_df, goal, meal_type, target_kcal, used_names)
                if pick is None:
                    continue
                meal_name = pick.get("Meal_Name", "Meal")

            rows.append({
                "Day": day,
                "Meal_Type": meal_type,
                "Meal_Name": str(meal_name),
                "Calories": round(safe_number(pick.get("Calories")), 1),
                "Protein_g": round(safe_number(pick.get("Protein_g")), 1),
                "Carbs_g": round(safe_number(pick.get("Carbs_g")), 1),
                "Fat_g": round(safe_number(pick.get("Fat_g")), 1),
            })

    return rows

# ================== API ==================
@app.get("/")
def home():
    return {
        "message": "FitMeal API working ✅",
        "model_loaded": model_name
    }


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

    predicted_energy_balance, class_probabilities, model_error = predict_energy_balance(
        data=data,
        bmi=bmi,
        tdee=tdee,
        targets=targets
    )

    meals_df = load_meals()
    snacks_df = load_snacks()
    drinks_df = load_drinks()

    plan = build_weekly_plan(
        meals_df=meals_df,
        snacks_df=snacks_df,
        goal=goal,
        tdee=tdee,
        meals_per_day=data.meals_per_day
    )

    return {
        "bmi": round(bmi, 2),
        "goal": goal,
        "tdee": round(tdee, 1),

        "model_used": model_name,
        "predicted_energy_balance": predicted_energy_balance,
        "class_probabilities": class_probabilities,
        "model_error": model_error,

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

        "weekly_plan": plan,

        "drinks": [
            "Water (2-3L/day)",
            "Green Tea",
            "Laban"
        ]
    }