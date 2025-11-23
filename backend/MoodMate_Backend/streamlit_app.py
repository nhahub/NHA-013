import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import joblib
import warnings
import random
import json
import time 

# -------------------------------------------------------------
# Section 1: ML Logic and Preprocessing (In-Memory)
# -------------------------------------------------------------

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Final 30 columns for model (fixed order)
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

# Fixed scaling parameters (mean & std dev)
SCALING_PARAMS = {
    'Stress_Score': {'mean': 0.40, 'std': 0.30}, 'Social_Function_Score': {'mean': 0.05, 'std': 0.90},
    'Family_Support_Impact': {'mean': 0.30, 'std': 0.45}, 'Country_TreatmentRate': {'mean': 0.55, 'std': 0.15},
    'Year': {'mean': 2020.0, 'std': 3.0}, 'Month': {'mean': 6.0, 'std': 3.0},
    'Day': {'mean': 15.0, 'std': 8.0}, 'Hour': {'mean': 12.0, 'std': 5.0}
}
SCALING_COLS = list(SCALING_PARAMS.keys())

# Preprocessing functions (must match training pipeline)

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
        if any(w in lower_answer for w in ['عالي', 'high', 'سريع', 'كتير']): return 'High'
        if any(w in lower_answer for w in ['متوسط', 'medium', 'عادي', 'احيانا']): return 'Medium'
        if any(w in lower_answer for w in ['منخفض', 'low', 'قليل', 'نادر']): return 'Low'
    if field_name == 'Days_Indoors':
        if any(w in lower_answer for w in ['يومياً', 'every day', 'كل يوم', 'بخرج']): return 'EveryDay'
        if any(w in lower_answer for w in ['أغلب الوقت', 'moderate', 'كام يوم']): return 'Moderate'
        if any(w in lower_answer for w in ['نادرًا', 'isolated', 'مبخرجش', 'قاعد']): return 'Isolated'
    return answer

def get_country_rate(country_name):
    """Map country to treatment rate"""
    lower_name = str(country_name).lower()
    if 'egypt' in lower_name or 'مصر' in lower_name or 'saudi' in lower_name or 'السعودية' in lower_name:
        return 0.75 
    return 0.50 

def apply_scaling(df):
    """Apply standardization to numerical features"""
    df_scaled = df.copy()
    for col in SCALING_COLS:
        if col in df_scaled.columns and SCALING_PARAMS[col]['std'] != 0:
            mean = SCALING_PARAMS[col]['mean']
            std = SCALING_PARAMS[col]['std']
            df_scaled[col] = (df_scaled[col] - mean) / std
    return df_scaled

# ---------------------------------------------
# Load core resources (model and JSON files)
# ---------------------------------------------

@st.cache_resource
def load_resources():
    """Load model and JSON data files (responses & solutions)"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    
    # Load model
    try:
        model = joblib.load(os.path.join(base_dir, 'health_chatbot_model.joblib'))
    except FileNotFoundError:
        st.error(f"❌ خطأ: ملف الموديل ('health_chatbot_model.joblib') غير موجود في مسار التطبيق.")
        model = None
        
    # Load JSON data files
    def load_json_data(file_name):
        full_path = os.path.join(base_dir, file_name)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"❌ خطأ أثناء تحميل الملف '{file_name}': تأكد من أنه بصيغة JSON سليمة وموجود في مسار التطبيق.")
            return None

    responses = load_json_data("responses.json")
    solutions = load_json_data("solutions.json")
    
    return model, responses, solutions

MODEL, RESPONSES, SOLUTIONS = load_resources()

# Stop app if resources failed to load
if MODEL is None or RESPONSES is None or SOLUTIONS is None:
    st.error("⚠️ فشل في تحميل الموارد الأساسية (الموديل أو ملفات JSON). يرجى التأكد من الأسماء والمسارات.")
    st.stop() 


def get_prediction_from_user_input(user_answers: dict) -> float:
    """Apply full preprocessing pipeline and generate prediction"""
    
    if MODEL is None: return 0.0

    df = pd.DataFrame([user_answers])
    
    # Ensure initial columns exist
    required_initial_cols = ['Gender', 'self_employed', 'family_history', 'Days_Indoors', 'Growing_Stress', 
                             'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 'Coping_Struggles', 
                             'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options', 
                             'Occupation', 'Country']
    for col in required_initial_cols:
        if col not in df.columns:
            default_value = 'No' if col in ['self_employed', 'family_history'] else 'Other' 
            df[col] = default_value

    df_ohe_source = df[['Occupation', 'Country']].copy() 

    # Apply linguistic standardization
    text_cols_to_interpret = df.select_dtypes(include=['object']).columns.tolist()
    for col in text_cols_to_interpret:
        if col in df.columns:
            df[col] = df.apply(lambda row: get_standard_key(row[col], col), axis=1)

    # Extract time features
    now = datetime.datetime.now()
    df['Year'], df['Month'], df['Day'], df['Hour'] = now.year, now.month, now.day, now.hour

    # Apply binary and ordinal encoding
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
        
    # Create engineered features
    stress_cols = ['Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Coping_Struggles', 'Mood_Swings']
    df['Stress_Score'] = df[stress_cols].mean(axis=1)
    df['Social_Function_Score'] = (df['Work_Interest'] - df['Social_Weakness'])
    df['SelfEmployment_Risk'] = (df['self_employed'] * (1 - df['care_options']))
    df['Family_Support_Impact'] = (df['family_history'] * df['Coping_Struggles'])
    df['Is_MidYear'] = df['Month'].between(5, 8).astype(int)
    df['Is_Winter'] = df['Month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
    df['Is_Night'] = df['Hour'].apply(lambda x: 1 if x >= 21 or x <= 6 else 0)

    # Apply target encoding and drop Country column
    df['Country_TreatmentRate'] = df['Country'].apply(get_country_rate)
    df.drop('Country', axis=1, inplace=True) 
    
    # Apply scaling before OHE
    df = apply_scaling(df)

    # Restore Occupation for OHE
    df['Occupation'] = df_ohe_source['Occupation']
    
    # Apply one-hot encoding
    df_final = pd.get_dummies(df, columns=['Occupation'], dtype=int)
    
    # Ensure all OHE columns exist
    ohe_cols = ['Occupation_Business', 'Occupation_Corporate', 'Occupation_Housewife', 'Occupation_Others', 'Occupation_Student']
    for col in ohe_cols:
        if col not in df_final.columns:
            df_final[col] = 0

    # Enforce final column order
    df_ready = df_final.reindex(columns=FINAL_OUTPUT_COLUMNS_ORDER, fill_value=0)
    
    X_final = df_ready.values
    
    # Get prediction probability
    prediction_proba = MODEL.predict_proba(X_final)[0][1] 
    
    return float(prediction_proba)

# -------------------------------------------------------------
# Section 2: Conversation Analysis Functions
# -------------------------------------------------------------

def get_sentiment_score(text):
    """Simple sentiment analysis without TextBlob dependency"""
    if any(word in text for word in ["زعلان", "وحش", "تعبان", "ضيق", "حزين", "مكتئب"]): return -0.5
    if any(word in text for word in ["سعيد", "ممتاز", "كويس", "فرحان", "جميل"]): return 0.5
    return 0.0

def get_empathetic_reply_and_key(user_text, question_config):
    """Match user input to predefined answer keys and return empathetic reply"""
    user_text_lower = user_text.lower()
    replies_config = question_config.get("answer_replies", {})
    
    # Try keyword matching
    for std_key, data in replies_config.items():
        if std_key != "Other":
            for keyword in data.get("keywords", []):
                if keyword in user_text_lower:
                    reply = random.choice(data.get("bot_reply", ["تمام."]))
                    return reply, std_key 
    
    # Try "Other" fallback
    if "Other" in replies_config:
        reply = random.choice(replies_config["Other"].get("bot_reply", ["تمام، سجلت ده."]))
        if question_config.get("field") == "Country":
            return reply, user_text 
        return reply, "Other"
        
    # No match found
    return None, user_text 

def check_mood_keywords(user_text):
    """Check if user message contains mood keywords"""
    if not RESPONSES or "mood_keywords" not in RESPONSES: return None 
    user_text_lower = user_text.lower()
    for mood in ["مبضون", "وحش", "تعبان", "زعلان", "سيء"]:
        if mood in RESPONSES["mood_keywords"]:
            for keyword in RESPONSES["mood_keywords"][mood]:
                if keyword in user_text_lower:
                    return mood 
    for mood in ["ممتاز", "كويس"]:
        if mood in RESPONSES["mood_keywords"]:
            for keyword in RESPONSES["mood_keywords"][mood]:
                if keyword in user_text_lower:
                    return mood 
    return None 

def build_solutions_menu(collected_data):
    """Build list of problems based on collected answers"""
    problem_list = []
    if not SOLUTIONS: return []
    for problem_key, problem_data in SOLUTIONS.items():
        if problem_key == "final_summary": continue
        
        user_answer = collected_data.get(problem_key, "") 
        standard_key = get_standard_key(user_answer, problem_key)
        
        if standard_key in problem_data.get("trigger_answer", []):
            problem_list.append(problem_key) 
    return problem_list

def format_solution(problem_key):
    """Build complete solution message list for a given problem"""
    if not SOLUTIONS or problem_key not in SOLUTIONS: return ["آسف، مش لاقي حلول للمشكلة دي."] 
    data = SOLUTIONS[problem_key]
    response_list = [] 
    
    # Add problem intro and description
    if "problem_intro" in data and data["problem_intro"]: response_list.append(data['problem_intro'])
    else: response_list.append(f"تمام، خلينا نتكلم عن **{data['problem_name']}**.")
      
    if data.get("descriptions") and data["descriptions"]:
        response_list.append(f"**ملخص المشكلة:**\n{random.choice(data['descriptions'])}")
    
    # Add practical solutions (pick 2 random)
    if data.get("solutions") and data["solutions"]:
        sol_text = "**طيب، إيه حلين عمليين مقترحين؟**\n"
        k = min(len(data["solutions"]), 2) 
        chosen_solutions = random.sample(data["solutions"], k)
        for i, sol in enumerate(chosen_solutions):
            sol_text += f"\n**{i+1}.** {sol}"
        response_list.append(sol_text) 
    
    # Add video and podcast resources
    if data.get("videos") and data["videos"]: 
        response_list.append(f"\n{data.get('video_intro', 'فيديوهات مقترحة:')}\n- [شاهد الفيديو]({random.choice(data['videos'])})")
            
    if data.get("podcasts") and data["podcasts"]: 
        response_list.append(f"\n{data.get('podcast_intro', 'بودكاست مقترح:')}\n- [استمع للبودكاست]({random.choice(data['podcasts'])})")
            
    return response_list 


def reset_session():
    """Reset session state to initial values"""
    st.session_state.convo_state = {
        "mode": "greeting", 
        "current_question_index": 0,
        "collected_data": {},
        "problem_list": [] 
    }

# -------------------------------------------------------------
# Section 3: Main Streamlit Interface and Flow Logic
# -------------------------------------------------------------

st.set_page_config(page_title="MoodMate", page_icon="🤖")
st.title("🤖 MoodMate")
st.caption("أنا صديقك النفسي، وموجود هنا عشان أسمعك.")

# Apply RTL for Arabic
st.markdown("""
<style>
div[data-testid="chat-message-container"] {direction: rtl; text-align: right;}
div[data-testid="stTextInput"] > div > div > input {direction: rtl; text-align: right;}
</style>
""", unsafe_allow_html=True)

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_greeting = random.choice(RESPONSES["greetings"]["عام"]) + " عامل إيه النهارده؟"
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

# Initialize conversation state
if "convo_state" not in st.session_state:
    reset_session() 

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat logic

if user_prompt := st.chat_input("اكتب رسالتك هنا..."):
    # Display user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Begin bot logic
    user_text = user_prompt.strip()
    state = st.session_state.convo_state 
    
    # Check for keywords
    is_farewell = any(keyword in user_text.lower() for keyword in RESPONSES.get("farewell_keywords", []))
    mood_key_check = check_mood_keywords(user_text)
    is_negative_trigger = mood_key_check in ["وحش", "تعبان", "مبضون", "زعلان", "سيء"]
    is_greeting = any(keyword in user_text.lower() for keyword in RESPONSES["greetings_keywords"]["عام"]) and len(user_text.split()) < 4
    
    # Fix stuck memory if user triggers reset keywords
    if (state["mode"] != "greeting") and (is_farewell or is_negative_trigger or is_greeting):
          reset_session()
          state = st.session_state.convo_state
        
    # State machine logic (ordered by priority)

    # State: farewell
    if is_farewell:
        bot_response = random.choice(RESPONSES.get('farewells'))
        with st.chat_message("assistant"): st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        reset_session()
        
    # State: awaiting confirmation
    elif state["mode"] == "awaiting_confirmation":
        conf_keywords = RESPONSES["interview_intro"]["confirmation_keywords"]
        if any(keyword in user_text.lower() for keyword in conf_keywords):
            state["mode"] = "in_interview"
            first_question = RESPONSES["interview_questions"][0]
            state["current_question_index"] = 0
            bot_response = first_question["question"]
        else:
            state["mode"] = "greeting" 
            bot_response = "تمام، براحتك جدًا. لو حبيت نبدأ في أي وقت، قولي بس إنك متضايق أو زهقان."
        with st.chat_message("assistant"): st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
    # State: inside interview
    elif state["mode"] == "in_interview":
        last_q_index = state["current_question_index"]
        last_q_config = RESPONSES["interview_questions"][last_q_index]

        empathetic_reply, stored_key = get_empathetic_reply_and_key(user_text, last_q_config)
        
        # Handle failed match (unclear response)
        if empathetic_reply is None:
            bot_response = random.choice(RESPONSES.get("unclear_responses"))
            with st.chat_message("assistant"): st.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.rerun() 
            
        # Successful match - store answer
        state["collected_data"][last_q_config["field"]] = stored_key
        
        with st.chat_message("assistant"):
            with st.spinner("..."): time.sleep(0.5)
            st.markdown(empathetic_reply)
        st.session_state.messages.append({"role": "assistant", "content": empathetic_reply})

        # Check if more questions remain
        next_q_index = last_q_index + 1
        if next_q_index < len(RESPONSES["interview_questions"]):
            next_question = RESPONSES["interview_questions"][next_q_index]
            state["current_question_index"] = next_q_index
            bot_response = next_question["question"] 
            with st.chat_message("assistant"):
                with st.spinner("..."): time.sleep(0.5)
                st.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
        else:
            # Interview complete
            # Send completion message
            with st.chat_message("assistant"):
                with st.spinner("..."): time.sleep(1.0)
                st.markdown(RESPONSES["interview_end"])
            st.session_state.messages.append({"role": "assistant", "content": RESPONSES["interview_end"]})
            
            # Generate prediction
            prediction_result = get_prediction_from_user_input(state["collected_data"]) 
            
            # Calculate and display stability score
            risk_percentage = prediction_result * 100
            stability_score = 100 - risk_percentage
            
            # Determine appropriate advice message
            if stability_score >= 95:
                advice = "✅ هذا يعني أن الموديل لا يرى حاليًا ضرورة ملحة للحصول على رعاية متخصصة."
            elif stability_score < 50:
                advice = "🚨 النسبة منخفضة، يرجى التفكير جدياً في الخيارات المتاحة للمساعدة."
            else:
                advice = "⚠️ النسبة جيدة، لكن يفضل متابعة الحلول التي سنعرضها."
                
            # Display prediction message
            with st.chat_message("assistant"):
                with st.spinner("..."): time.sleep(1.0)
                
                bot_response_prediction_main = (
                    f"بناءً على تحليل إجاباتك، يُظهر نموذجنا الإحصائي أن نسبة **الصحة النفسية المناسبة** لديك "
                    f"هي: **{stability_score:.2f}%** تقريبًا."
                )
                
                st.markdown(bot_response_prediction_main)
            st.session_state.messages.append({"role": "assistant", "content": bot_response_prediction_main})
            
            # Display advice message
            with st.chat_message("assistant"):
                 with st.spinner("..."): time.sleep(0.5)
                 st.markdown(advice)
            st.session_state.messages.append({"role": "assistant", "content": advice})
            
            # Build and display solutions menu
            problem_list = build_solutions_menu(state["collected_data"])
            
            if not problem_list:
                bot_response = "بصراحة، من إجاباتك النفسية، أنا شايف إنك في حالة كويسة ومش محتاج أي حلول. لو حابب تتكلم في أي حاجة تانية أنا موجود!"
                state["mode"] = "final_summary"
            else:
                state["problem_list"] = problem_list
                state["mode"] = "solutions_menu"
                menu_text = "ودلوقتي، خلينا نتكلم في (الحلول النفسية) للمشاكل اللي إنت ذكرتها. أنا لاحظت إننا ممكن نتكلم في النقط دي:\n\n"
                for i, problem_key in enumerate(problem_list):
                    problem_name = SOLUTIONS[problem_key].get("problem_name", problem_key)
                    menu_text += f"**{i+1}. {problem_name}**\n"
                menu_text += "\nتحب نبدأ بأنهي واحدة فيهم؟ (اكتب الرقم أو الاسم)"
                bot_response = menu_text
            
            with st.chat_message("assistant"):
                with st.spinner("..."): time.sleep(1.0)
                st.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
    # State: solutions menu
    elif state["mode"] == "solutions_menu":
        chosen_problem = None
        if SOLUTIONS and "problem_list" in state: 
            for i, problem_key in enumerate(state["problem_list"]):
                problem_name = SOLUTIONS.get(problem_key, {}).get("problem_name", "")
                if (str(i+1) == user_text) or (problem_name and problem_name.lower() in user_text.lower()) or (problem_key.lower() in user_text.lower()):
                    chosen_problem = problem_key
                    break
            
        if chosen_problem:
            solution_responses = format_solution(chosen_problem)
            state["problem_list"].remove(chosen_problem)
            
            # Display all solution messages
            for response in solution_responses:
                with st.chat_message("assistant"):
                    with st.spinner("..."): time.sleep(0.5) 
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            # Check if more problems remain
            with st.chat_message("assistant"):
                with st.spinner("..."): time.sleep(0.5)
                if state["problem_list"]:
                    bot_response = "\n\n--- (فاصل) ---\nتحب نكمل في النقط الباقية؟ (قول 'كمل' أو 'لا')"
                    state["mode"] = "solutions_flow"
                else:
                    state["mode"] = "final_summary"
                    bot_response = "✅ خلصنا كل الحلول! جاري إرسال الخاتمة النهائية."
                
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
        else:
            bot_response = "آسف مش فاهم. ممكن تختار رقم أو اسم المشكلة من القايمة؟"
            with st.chat_message("assistant"): st.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
    # State: solutions flow continuation
    elif state["mode"] == "solutions_flow":
        if any(keyword in user_text.lower() for keyword in ["نعم", "اه", "ماشي", "تمام", "كمل"]):
            state["mode"] = "solutions_menu"
            menu_text = "تمام. دي النقط الباقية اللي ممكن نتكلم فيها:\n\n"
            for i, problem_key in enumerate(state["problem_list"]):
                problem_name = SOLUTIONS[problem_key]["problem_name"]
                menu_text += f"**{i+1}. {problem_name}**\n"
            menu_text += "\nتحب نختار أنهي واحدة؟"
            bot_response = menu_text
        else:
            state["mode"] = "final_summary" 
            summary_messages = SOLUTIONS.get("final_summary", {}).get("messages", ["شكرًا لوقتك. لو احتجت أي حاجة تانية، أنا موجود!"])
            bot_response = "\n".join(summary_messages)
            
        with st.chat_message("assistant"): st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
# State: final summary
    elif state["mode"] == "final_summary":
        summary_messages = SOLUTIONS.get("final_summary", {}).get("messages", ["شكرًا لوقتك. لو احتجت أي حاجة تانية، أنا موجود!"])
        
        # Display all summary messages
        for msg in summary_messages:
            with st.chat_message("assistant"):
                with st.spinner("..."): time.sleep(0.5) 
                st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            
        reset_session() 

    # State: greeting/normal conversation
    elif state["mode"] == "greeting":
        
        mood_key_check = check_mood_keywords(user_text)
        
        # Handle greeting
        if is_greeting: 
            bot_response = f"{random.choice(RESPONSES['greetings']['عام'])} عامل إيه النهارده؟"
        # Handle negative mood trigger
        elif mood_key_check in ["وحش", "تعبان", "مبضون", "زعلان", "سيء"]:
            bot_response = RESPONSES["interview_intro"]["speech"]
            state["mode"] = "awaiting_confirmation"
        # Handle positive mood
        elif mood_key_check in ["ممتاز", "كويس"]:
            bot_response = random.choice(RESPONSES["mood_responses"][mood_key_check]["responses"])
        # Sentiment-based fallback
        else:
            sentiment_score = get_sentiment_score(user_text)
            if sentiment_score < -0.2: 
                bot_response = RESPONSES["interview_intro"]["speech"]
                state["mode"] = "awaiting_confirmation" 
            elif sentiment_score > 0.3: 
                bot_response = random.choice(RESPONSES["mood_responses"]["ممتاز"]["responses"])
            else:
                bot_response = random.choice(RESPONSES.get("unclear_responses"))
            
        with st.chat_message("assistant"): st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
    # Save updated state to session
    st.session_state.convo_state = state
        
    st.rerun()