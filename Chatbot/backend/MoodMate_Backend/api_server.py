import pandas as pd
import numpy as np
import datetime
import os
import joblib
import json
import warnings
import random
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ------------------------
# Basic ML settings & consts
# ------------------------
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

FINAL_OUTPUT_COLUMNS_ORDER = [
    'Gender', 'self_employed', 'family_history', 'Days_Indoors', 'Growing_Stress',
    'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 'Coping_Struggles',
    'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options',
    'Year', 'Month', 'Day', 'Hour', 'Occupation_Business', 'Occupation_Corporate',
    'Occupation_Housewife', 'Occupation_Others', 'Occupation_Student',
    'Stress_Score', 'Social_Function_Score', 'SelfEmployment_Risk',
    'Family_Support_Impact', 'Is_Winter', 'Is_MidYear', 'Is_Night',
    'Country_TreatmentRate'
]

SCALING_PARAMS = {
    'Stress_Score': {'mean': 0.40, 'std': 0.30}, 'Social_Function_Score': {'mean': 0.05, 'std': 0.90},
    'Family_Support_Impact': {'mean': 0.30, 'std': 0.45}, 'Country_TreatmentRate': {'mean': 0.55, 'std': 0.15},
    'Year': {'mean': 2020.0, 'std': 3.0}, 'Month': {'mean': 6.0, 'std': 3.0},
    'Day': {'mean': 15.0, 'std': 8.0}, 'Hour': {'mean': 12.0, 'std': 5.0}
}
SCALING_COLS = list(SCALING_PARAMS.keys())

# ------------------------
# Preprocessing helpers
# ------------------------

def get_standard_key(answer, field_name):
    """Normalize various free-text answers into standard keys."""
    lower_answer = str(answer).lower().strip()
    if any(w in lower_answer for w in ['نعم', 'اه', 'yes', 'موافق', 'أكيد', 'لسه', 'بقدر', 'عارف']):
        return 'Yes'
    if any(w in lower_answer for w in ['لا', 'no', 'خالص', 'مفيش', 'صعب', 'فقدت', 'مبخرجش']):
        return 'No'
    if any(w in lower_answer for w in ['يمكن', 'maybe', 'مش متأكد', 'نص نص', 'أحيانًا', 'مش أوي']):
        return 'Maybe'

    # Map gender variants
    if field_name == 'Gender':
        if any(w in lower_answer for w in ['ذكر', 'male', 'رجل']): return 'Male'
        if any(w in lower_answer for w in ['أنثى', 'female', 'بنت']): return 'Female'

    # Map occupation variants
    if field_name == 'Occupation':
        if any(w in lower_answer for w in ['طالب', 'student', 'بدرس', 'جامعة']): return 'Student'
        if any(w in lower_answer for w in ['موظف', 'corporate', 'بشتغل']): return 'Corporate'
        if any(w in lower_answer for w in ['عمل حر', 'فريلانسر', 'بزنس']): return 'Business'
        if any(w in lower_answer for w in ['ربة منزل', 'housewife']): return 'Housewife'
        if any(w in lower_answer for w in ['عاطل', 'لا أعمل']): return 'Other'

    # Map mood swings
    if field_name == 'Mood_Swings':
        if any(w in lower_answer for w in ['عالي', 'high', 'سريع', 'كتير']): return 'High'
        if any(w in lower_answer for w in ['متوسط', 'medium', 'عادي', 'احيانا']): return 'Medium'
        if any(w in lower_answer for w in ['منخفض', 'low', 'قليل', 'نادر']): return 'Low'

    # Map days indoors
    if field_name == 'Days_Indoors':
        if any(w in lower_answer for w in ['يومياً', 'every day', 'كل يوم', 'بخرج']): return 'EveryDay'
        if any(w in lower_answer for w in ['أغلب الوقت', 'moderate', 'كام يوم']): return 'Moderate'
        if any(w in lower_answer for w in ['نادرًا', 'isolated', 'مبخرجش', 'قاعد']): return 'Isolated'

    return answer

def get_country_rate(country_name):
    """Convert country name to expected treatment rate (target)."""
    lower_name = str(country_name).lower()
    if 'egypt' in lower_name or 'مصر' in lower_name or 'saudi' in lower_name or 'السعودية' in lower_name:
        return 0.75
    return 0.50

def apply_scaling(df):
    """Apply manual standardization for selected numeric columns."""
    df_scaled = df.copy()
    for col in SCALING_COLS:
        if col in df_scaled.columns and SCALING_PARAMS[col]['std'] != 0:
            mean = SCALING_PARAMS[col]['mean']
            std = SCALING_PARAMS[col]['std']
            df_scaled[col] = (df_scaled[col] - mean) / std
    return df_scaled

# ------------------------
# Resources loading (model & json)
# ------------------------

def load_resources():
    """Load model and supporting JSON files from the project directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        model = joblib.load(os.path.join(base_dir, 'health_chatbot_model.joblib'))
    except:
        model = None

    def load_json_data(file_name):
        try:
            with open(os.path.join(base_dir, file_name + '.json'), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            try:
                with open(os.path.join(base_dir, file_name), 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None

    solutions = load_json_data("solutions")

    return model, solutions

MODEL, SOLUTIONS = load_resources()

if MODEL is None:
    raise RuntimeError("Failed to load ML model. Ensure 'health_chatbot_model.joblib' exists.")

# ------------------------
# Core prediction function
# ------------------------

def get_prediction_from_user_input(user_answers: Dict[str, str]) -> float:
    """Preprocess input, run the model, and return a stability score (0..1)."""

    df = pd.DataFrame([user_answers], index=[0])

    required_initial_cols = ['Gender', 'self_employed', 'family_history', 'Days_Indoors', 'Growing_Stress',
                             'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 'Coping_Struggles',
                             'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options',
                             'Occupation', 'Country']
    # Ensure required columns exist
    for col in required_initial_cols:
        if col not in df.columns:
            default_value = 'No' if col in ['self_employed', 'family_history'] else 'Other'
            df[col] = default_value

    # Keep original occupation/country for later OHE
    df_ohe_source = df[['Occupation', 'Country']].copy()

    # Normalize textual answers using get_standard_key
    text_cols_to_interpret = df.select_dtypes(include=['object']).columns.tolist()
    for col in text_cols_to_interpret:
        if col in df.columns:
            df[col] = df.apply(lambda row: get_standard_key(row[col], col), axis=1)

    # Add current timestamp features
    now = datetime.datetime.now()
    df['Year'], df['Month'], df['Day'], df['Hour'] = now.year, now.month, now.day, now.hour

    # Map categorical to numeric
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1}).fillna(0)
    df['self_employed'] = df['self_employed'].map({'No': 0, 'Yes': 1}).fillna(0)
    df['family_history'] = df['family_history'].map({'No': 0, 'Yes': 1}).fillna(0)
    df['Coping_Struggles'] = df['Coping_Struggles'].map({'No': 0, 'Yes': 1}).fillna(0)
    df['Days_Indoors'] = df['Days_Indoors'].map({'EveryDay': 0, 'Moderate': 1, 'Isolated': 4}).fillna(1)
    df['Mood_Swings'] = df['Mood_Swings'].map({'Low': 0, 'Medium': 1, 'High': 2}).fillna(0)

    # Map some label-like fields to numeric scores
    label_map = {'No': 0.0, 'Yes': 1.0, 'Maybe': 0.5}
    label_cols = [
        'Growing_Stress', 'Changes_Habits', 'Mental_Health_History',
        'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options'
    ]
    for col in label_cols:
        df[col] = df[col].map(label_map).fillna(0.0)

    # Create aggregated features
    stress_cols = ['Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Coping_Struggles', 'Mood_Swings']
    df['Stress_Score'] = df[stress_cols].mean(axis=1)
    df['Social_Function_Score'] = (df['Work_Interest'] - df['Social_Weakness'])
    df['SelfEmployment_Risk'] = (df['self_employed'] * (1 - df['care_options']))
    df['Family_Support_Impact'] = (df['family_history'] * df['Coping_Struggles'])
    df['Is_MidYear'] = df['Month'].between(5, 8).astype(int)
    df['Is_Winter'] = df['Month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
    df['Is_Night'] = df['Hour'].apply(lambda x: 1 if x >= 21 or x <= 6 else 0)

    # Country treatment rate and scaling
    df['Country_TreatmentRate'] = df['Country'].apply(get_country_rate)
    df.drop('Country', axis=1, inplace=True)
    df = apply_scaling(df)

    # One-hot encode occupation using original source
    df['Occupation'] = df_ohe_source['Occupation']
    df_final = pd.get_dummies(df, columns=['Occupation'], dtype=int)

    # Ensure expected OHE columns exist
    ohe_cols = ['Occupation_Business', 'Occupation_Corporate', 'Occupation_Housewife', 'Occupation_Others', 'Occupation_Student']
    for col in ohe_cols:
        if col not in df_final.columns:
            df_final[col] = 0

    # Reindex to final expected column order
    df_ready = df_final.reindex(columns=FINAL_OUTPUT_COLUMNS_ORDER, fill_value=0)
    X_final = df_ready.values

    # Predict probability of risk (assumes model supports predict_proba)
    risk_probability = MODEL.predict_proba(X_final)[0][1]
    stability_score = 1 - risk_probability

    return float(stability_score)

# ------------------------
# Solutions generation
# ------------------------

def build_solutions_report(user_answers: Dict[str, str], solutions_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a solutions report by matching triggers and sampling solutions."""
    triggered_problems = []

    for problem_key, problem_data in solutions_data.items():
        if problem_key == "final_summary":
            continue

        user_answer = user_answers.get(problem_key)
        standard_key = get_standard_key(user_answer, problem_key)

        if standard_key in problem_data.get("trigger_answer", []):
            # pick two solutions and one video (if available)
            solutions_list = random.sample(problem_data.get("solutions", []), min(2, len(problem_data.get("solutions", []))))
            video_url = random.choice(problem_data.get("videos", [])) if problem_data.get("videos") else None

            triggered_problems.append({
                "key": problem_key,
                "name": problem_data.get("problem_name"),
                "description": random.choice(problem_data.get("descriptions", ["No description."])),
                "selected_solutions": solutions_list,
                "video_link": video_url
            })

    return {"problems": triggered_problems}

# ------------------------
# FastAPI setup & endpoint
# ------------------------

class FinalAnswersModel(BaseModel):
    Gender: str
    Country: str
    Occupation: str
    Growing_Stress: str
    Changes_Habits: str
    Days_Indoors: str
    Mood_Swings: str
    Coping_Struggles: str
    Work_Interest: str
    Social_Weakness: str
    Mental_Health_History: str
    family_history: str
    care_options: str
    mental_health_interview: str

app = FastAPI(title="Simplified Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/predict_health")
def predict_final_score(user_data: FinalAnswersModel):
    # Receive user answers, predict stability, and return solutions report.
    try:
        answers_dict = user_data.model_dump()

        # Prediction step
        risk_score_raw = get_prediction_from_user_input(answers_dict)
        stability_score = 1.0 - risk_score_raw  # عكس النتيجة للحصول على الاستقرار

        # Solutions generation
        solutions_report = build_solutions_report(answers_dict)

        # Build final message metrics
        stability_percent = round(stability_score * 100, 2)
        risk_percent_display = round(risk_score_raw * 100, 2)

        # تحديد فئة النصيحة بناءً على النسبة
        if stability_percent >= 90:
            tiered_advice = "النسبة ممتازة ولا يوجد داعي للقلق."
        elif stability_percent >= 50:
            tiered_advice = "الحالة مستقرة بشكل معقول ويفضل متابعة النصائح."
        else:
            tiered_advice = "النسبة منخفضة ويُفضل التفكير في دعم متخصص."

        # 🚨 التعديل الحاسم: بناء الرسالة النهائية المُفصلة
        final_advice = (
            f"بناءً على تحليل إجاباتك، نسبة الصحة النفسية المناسبة لديك: **{stability_percent}%** "
            f"(مستوى الخطورة: **{risk_percent_display}%**).\n\n"
            f"{tiered_advice}"
        )

        return {
            "status": "success",
            "stability_percentage": stability_percent,
            "risk_percentage": risk_percent_display,
            "final_advice": final_advice,
            "solutions_report": solutions_report
        }


    except Exception as e:
        import traceback
        print(f"❌ Backend Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"فشل في معالجة التنبؤ: {str(e)}")
# ------------------------
# Run server (development)
# ------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simplified API Server on Port 8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
