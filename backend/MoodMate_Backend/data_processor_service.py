# data_processor_sequential.py - Final corrected code for solving the 0.69% issue

import pandas as pd
import numpy as np
import datetime
import os
import random 
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------
# Core Constants and Scaling Parameters
# ---------------------------------------------

# File paths for testing
INPUT_FILE_PATH = 'raw_answers.csv'
OUTPUT_FILE_PATH = 'final_features_ready.csv'

# Final 30 columns in exact model order
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

# Fixed scaling parameters (mean & standard deviation)
# These values transform raw scores into expected scaled values
SCALING_PARAMS = {
    'Stress_Score': {'mean': 0.40, 'std': 0.30},
    'Social_Function_Score': {'mean': 0.05, 'std': 0.90},
    'Family_Support_Impact': {'mean': 0.30, 'std': 0.45},
    'Country_TreatmentRate': {'mean': 0.55, 'std': 0.15},
    'Year': {'mean': 2020.0, 'std': 3.0}, 
    'Month': {'mean': 6.0, 'std': 3.0},
    'Day': {'mean': 15.0, 'std': 8.0},
    'Hour': {'mean': 12.0, 'std': 5.0}
}
SCALING_COLS = list(SCALING_PARAMS.keys())


# ---------------------------------------------
# Interpretation and Standardization Functions
# ---------------------------------------------

def get_standard_key(answer, field_name):
    """Convert Arabic/colloquial answers to standardized English keys"""
    lower_answer = str(answer).lower().strip()
    
    if any(w in lower_answer for w in ['نعم', 'اه', 'yes', 'موافق', 'أكيد', 'لسه', 'بقدر', 'عارف']): return 'Yes'
    if any(w in lower_answer for w in ['لا', 'no', 'خالص', 'مفيش', 'صعب', 'فقدت', 'مبخرجش']): return 'No'
    if any(w in lower_answer for w in ['يمكن', 'maybe', 'مش متأكد', 'نص نص', 'أحيانًا', 'مش أوي']): return 'Maybe'
    
    if field_name == 'Gender':
        if any(w in lower_answer for w in ['ذكر', 'male', 'رجل']): return 'Male'
        if any(w in lower_answer for w in ['أنثى', 'female', 'بنت']): return 'Female'
    
    if field_name == 'Occupation':
        if any(w in lower_answer for w in ['طالب', 'student', 'بدرس', 'جامعة']): return 'Student'
        if any(w in lower_answer for w in ['موظف', 'corporate', 'بشتغل']): return 'Corporate'
        if any(w in lower_answer for w in ['عمل حر', 'فريلانسر', 'بزنس']): return 'Business'
        if any(w in lower_answer for w in ['ربة منزل', 'housewife']): return 'Housewife'
        if any(w in lower_answer for w in ['عاطل', 'لا أعمل']): return 'Other'
    
    if field_name == 'Mood_Swings':
        if any(w in lower_answer for w in ['عالي', 'high', 'سريع']): return 'High'
        if any(w in lower_answer for w in ['متوسط', 'medium', 'عادي']): return 'Medium'
        if any(w in lower_answer for w in ['منخفض', 'low', 'قليل']): return 'Low'
    
    if field_name == 'Days_Indoors':
        if any(w in lower_answer for w in ['يومياً', 'every day', 'كل يوم']): return 'EveryDay'
        if any(w in lower_answer for w in ['أغلب الوقت', 'moderate', 'كام يوم']): return 'Moderate'
        if any(w in lower_answer for w in ['نادرًا', 'isolated', 'مبخرجش']): return 'Isolated'

    return answer

def get_country_rate(country_name):
    """Convert country name to expected treatment rate (Target Encoding)"""
    lower_name = str(country_name).lower()
    if 'egypt' in lower_name or 'مصر' in lower_name or 'saudi' in lower_name:
        return 0.75  # High treatment rate
    return 0.50  # Default rate

def apply_scaling(df):
    """Apply standardization (StandardScaler) to specified columns"""
    
    df_scaled = df.copy()
    
    for col in SCALING_COLS:
        if col in df_scaled.columns and SCALING_PARAMS[col]['std'] != 0:
            mean = SCALING_PARAMS[col]['mean']
            std = SCALING_PARAMS[col]['std']
            
            # Standard scaling formula: (x - mean) / std
            df_scaled[col] = (df_scaled[col] - mean) / std
            
    return df_scaled


# ---------------------------------------------
# Sequential Processing Execution
# ---------------------------------------------

if __name__ == "__main__":
    
    print("--- 🏭 Starting data processing ---")
    
    try:
        # Step 1: Read raw data
        df = pd.read_csv(INPUT_FILE_PATH)
        print(f"✅ Input file loaded: {INPUT_FILE_PATH}")
        
        # Step 1.1: Preserve original text columns for OHE
        df_ohe_source = df[['Occupation', 'Country']].copy() 
        
        # Step 2: Apply linguistic interpretation and standardization
        text_cols_to_interpret = df.select_dtypes(include=['object']).columns.tolist()

        for col in text_cols_to_interpret:
            if col in df.columns:
                df[col] = df.apply(lambda row: get_standard_key(row[col], col), axis=1)
        
        print("✅ Text answers standardized.")

        # Step 3: Extract time features
        now = datetime.datetime.now()
        df['Year'] = now.year
        df['Month'] = now.month
        df['Day'] = now.day
        df['Hour'] = now.hour
        if 'self_employed' not in df.columns:
          df['self_employed'] = 'No'
        
        # Step 4: Apply binary and ordinal encoding
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1}).fillna(0)
        df['self_employed'] = df['self_employed'].map({'No': 0, 'Yes': 1}).fillna(0)
        df['family_history'] = df['family_history'].map({'No': 0, 'Yes': 1}).fillna(0)
        df['Coping_Struggles'] = df['Coping_Struggles'].map({'No': 0, 'Yes': 1}).fillna(0)
        
        df['Days_Indoors'] = df['Days_Indoors'].map({'EveryDay': 0, 'Moderate': 1, 'Isolated': 4}).fillna(1)
        df['Mood_Swings'] = df['Mood_Swings'].map({'Low': 0, 'Medium': 1, 'High': 2}).fillna(0)
        
        # Apply label encoding for Yes/No/Maybe columns
        label_map = {'No': 0.0, 'Yes': 1.0, 'Maybe': 0.5}
        label_cols = [
            'Growing_Stress', 'Changes_Habits', 'Mental_Health_History',
            'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options'
        ]
        for col in label_cols:
            df[col] = df[col].map(label_map).fillna(0.0)
            
        # Step 5: Engineer score features
        stress_cols = ['Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Coping_Struggles', 'Mood_Swings']
        df['Stress_Score'] = df[stress_cols].mean(axis=1)
        df['Social_Function_Score'] = (df['Work_Interest'] - df['Social_Weakness'])
        df['SelfEmployment_Risk'] = (df['self_employed'] * (1 - df['care_options']))
        df['Family_Support_Impact'] = (df['family_history'] * df['Coping_Struggles'])
        df['Is_MidYear'] = df['Month'].between(5, 8).astype(int)
        df['Is_Winter'] = df['Month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
        df['Is_Night'] = df['Hour'].apply(lambda x: 1 if x >= 21 or x <= 6 else 0)



        # Step 6: Critical step - Target Encoding and OHE for Occupation only
        
        # Step 6.1: Apply Target Encoding (create Country_TreatmentRate)
        df['Country_TreatmentRate'] = df['Country'].apply(get_country_rate)
        # Drop original text column to avoid conflicts
        df.drop('Country', axis=1, inplace=True) 
        
        # Step 6.2: Apply scaling to score features before OHE
        df = apply_scaling(df)

        # Step 6.3: Restore text column for Occupation OHE
        # Use preserved text column from Step 1
        df['Occupation'] = df_ohe_source['Occupation']
        
        # Step 6.4: Apply one-hot encoding for Occupation only
        df_final = pd.get_dummies(df, columns=['Occupation'], dtype=int)
        
        # Step 7: Enforce final column order (Re-indexing)
        # Use reindex to ensure exact 30-column order and fill missing with 0
        df_ready = df_final.reindex(columns=FINAL_OUTPUT_COLUMNS_ORDER, fill_value=0)
        
        # Step 8: Save output to new CSV file
        df_ready.to_csv(OUTPUT_FILE_PATH, index=False)
        print(f"✅ Success! Model-ready output file created: {OUTPUT_FILE_PATH}")
        print(df_ready)

    except FileNotFoundError:
        print(f"❌ Error: Input file '{INPUT_FILE_PATH}' not found. Please create it first.")
    except Exception as e:
        print(f"❌ Processing failed: {e}. Please check column consistency.")