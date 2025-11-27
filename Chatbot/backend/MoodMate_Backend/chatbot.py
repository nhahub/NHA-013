import json
import random
from textblob import TextBlob
import os
import time
import sqlite3
import datetime

# Load responses JSON file
def load_responses(file_path='responses.json'):
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading responses: {e}")
        return None

# Get sentiment polarity using TextBlob
def get_sentiment_score(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Detect keyword in user response and return reply + stored key
def get_empathetic_reply_and_key(user_text, question_config):
    user_text_lower = user_text.lower()
    replies_config = question_config.get("answer_replies", {})

    # Search for predefined keywords
    for std_key, data in replies_config.items():
        if std_key != "Other":
            for keyword in data.get("keywords", []):
                if keyword in user_text_lower:
                    reply = random.choice(data.get("bot_reply", ["Ok."]))
                    return reply, std_key
    
    # Default fallback (Other)
    if "Other" in replies_config:
        reply = random.choice(
            replies_config["Other"].get("bot_reply", ["Noted."])
        )
        if question_config.get("field") == "Country":
            return reply, user_text
        return reply, "Other"
    
    return None, user_text

# Check mood keywords from JSON
def check_mood_keywords(user_text):
    if "mood_keywords" not in RESPONSES:
        return None
    
    user_text_lower = user_text.lower()

    for mood in ["مبضون", "وحش", "تعبان"]:
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

# Setup SQLite DB and ensure required columns exist
def setup_database():
    conn = sqlite3.connect('moodmate.db')
    c = conn.cursor()

    # Create interviews table if not exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            Gender TEXT,
            Country TEXT,
            Occupation TEXT,
            Growing_Stress TEXT,
            Changes_Habits TEXT,
            Days_Indoors TEXT,
            Mood_Swings TEXT,
            Coping_Struggles TEXT,
            Work_Interest TEXT,
            Social_Weakness TEXT,
            Mental_Health_History TEXT,
            family_history TEXT,
            care_options TEXT,
            mental_health_interview TEXT
        )
    ''')

    # Ensure newer columns exist in older DB versions
    existing_columns = [col[1] for col in c.execute("PRAGMA table_info(interviews)")]
    new_columns = {
        "Gender": "TEXT",
        "Country": "TEXT",
        "care_options": "TEXT",
        "mental_health_interview": "TEXT"
    }

    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            try:
                c.execute(f"ALTER TABLE interviews ADD COLUMN {col_name} {col_type}")
            except Exception as e:
                print(f"Column update error for {col_name}: {e}")

    conn.commit()
    conn.close()

# Save collected interview data dynamically
def save_interview(data):
    conn = sqlite3.connect('moodmate.db')
    c = conn.cursor()

    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?'] * len(data))
    values = list(data.values())

    columns += ', timestamp'
    placeholders += ', ?'
    values.append(datetime.datetime.now())

    try:
        query = f"INSERT INTO interviews ({columns}) VALUES ({placeholders})"
        c.execute(query, values)
        conn.commit()
    except Exception as e:
        print(f"DB save error: {e}")
    finally:
        conn.close()

# Load responses once
RESPONSES = load_responses()

if __name__ == "__main__":
    
    # Ensure DB exists and is configured
    setup_database()
    
    if not RESPONSES:
        print("Critical error: responses.json missing.")
    elif "mood_keywords" not in RESPONSES:
        print("Critical error: mood_keywords missing in JSON.")
    else:
        print("🤖 MoodMate: Ready. How are you today?")
        
        conversation_state = {
            "mode": "greeting",
            "current_question_index": 0,
            "collected_data": {}
        }

        while True:
            try:
                user_text = input("You: ").strip()
                bot_response = ""

                # Check for exit keywords
                if any(keyword in user_text.lower() for keyword in RESPONSES.get("farewell_keywords", [])):
                    print(f"🤖 MoodMate: {random.choice(RESPONSES.get('farewells'))}")
                    break

                # If waiting confirmation to start interview
                if conversation_state["mode"] == "awaiting_confirmation":
                    if any(keyword in user_text.lower() for keyword in RESPONSES["interview_intro"]["confirmation_keywords"]):
                        conversation_state["mode"] = "in_interview"
                        first_question = RESPONSES["interview_questions"][0]
                        conversation_state["current_question_index"] = 0
                        bot_response = first_question["question"]
                    else:
                        conversation_state["mode"] = "greeting"
                        bot_response = "Alright, whenever you’re ready."

                # If user is inside interview
                elif conversation_state["mode"] == "in_interview":
                    last_q_index = conversation_state["current_question_index"]
                    last_q_config = RESPONSES["interview_questions"][last_q_index]
                    last_q_field = last_q_config["field"]

                    empathetic_reply, stored_key = get_empathetic_reply_and_key(
                        user_text, last_q_config
                    )

                    conversation_state["collected_data"][last_q_field] = stored_key

                    if empathetic_reply:
                        print(f"🤖 MoodMate: {empathetic_reply}")
                        time.sleep(1.2)

                    next_q_index = last_q_index + 1
                    if next_q_index < len(RESPONSES["interview_questions"]):
                        next_question = RESPONSES["interview_questions"][next_q_index]
                        conversation_state["current_question_index"] = next_q_index
                        bot_response = next_question["question"]
                    else:
                        bot_response = RESPONSES["interview_end"]
                        save_interview(conversation_state["collected_data"])

                        conversation_state = {
                            "mode": "greeting",
                            "current_question_index": 0,
                            "collected_data": {}
                        }

                # Default greeting mode
                elif conversation_state["mode"] == "greeting":
                    if any(keyword in user_text.lower() for keyword in RESPONSES["greetings_keywords"]["عام"]) and len(user_text.split()) < 4:
                        bot_response = f"{random.choice(RESPONSES['greetings']['عام'])} كيف حالك اليوم؟"
                    else:
                        mood_key = check_mood_keywords(user_text)

                        if mood_key in ["وحش", "تعبان", "مبضون"]:
                            bot_response = RESPONSES["interview_intro"]["speech"]
                            conversation_state["mode"] = "awaiting_confirmation"

                        elif mood_key in ["ممتاز", "كويس"]:
                            bot_response = random.choice(
                                RESPONSES["mood_responses"][mood_key]["responses"]
                            )
                        else:
                            sentiment_score = get_sentiment_score(user_text)
                            if sentiment_score < -0.2:
                                bot_response = RESPONSES["interview_intro"]["speech"]
                                conversation_state["mode"] = "awaiting_confirmation"
                            elif sentiment_score > 0.3:
                                bot_response = random.choice(
                                    RESPONSES["mood_responses"]["ممتاز"]["responses"]
                                )
                            else:
                                bot_response = random.choice(
                                    RESPONSES.get("unclear_responses")
                                )

                print(f"🤖 MoodMate: {bot_response}")

            except EOFError:
                break
            except Exception as e:
                print(f"Critical runtime error: {e}")
