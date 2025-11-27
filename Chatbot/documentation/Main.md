# 💬 MoodMate Chatbot Documentation

---

## 🤖 Project Overview: Three Chatbots

In this project, **three chatbots** were created:

1.  The first one was a Python code chatbot that we deployed on **Streamlit**.
2.  The second one had a **front-end** and a **back-end**.
3.  The third one is when I took the second code and uploaded it to a **server**. This meant we had to **change the code structure**.

---

## 🛠️ The Basics

### 1. The Virtual Environment (`venv`)

We use **`venv`** to create a **separate and isolated space** for each project.
* **Goal:** To prevent library conflicts and keep dependencies clean.
* **How it works:** All required Python libraries are installed only inside the project's `venv` folder, guaranteeing the project runs smoothly without affecting other applications on the machine.
***(we will create venv in backend file)***

### 2. Response and Rule Files

We created **two files**:
* A **response file** that contains all the questions and answers the bot will use. This helps the bot respond easily to user input.
* This file also shows the questions about the most important features we got from the model we used.
* The bot will ask the user a question, take the answer, and then reply to that answer.
* This setup makes the chatbot conversation **more engaging and natural** (gives the chatbot "liveliness in the conversation").

### 3. The Solutions File (Advice and Resources)

This is an **important file**. Its job is to act after the model finishes its prediction and gives you a prediction number.
The file will:
* Show you the **problems** you were facing.
* Provide a detailed **description**, **symptoms**, and **solutions** for the issue.
* Finally, it will recommend a video or a podcast.
* We also added the conversation closing statement (the conclusion) here.

### 4. Saving the Model and Preprocessor 💾

We need to make sure we have **copies** of both the model and the preprocessing steps. We save them in two separate files. We use the following Python code in the main file to save them:

* **Saving the final model (e.g., XGBoost):**
    ```
    joblib.dump(final_XGB, 'health_chatbot_model.joblib')
    ```
* **Saving the data preprocessor:**
    ```
    joblib.dump(preprocessor, 'data_preprocessor.joblib')
    ```
This ensures the chatbot can **load and use** the trained model and the exact preprocessing steps later (e.g., when deployed on a server).

---

## 🐍 Chatbot 1: Python Console Version

This code is the core of the chatbot. It handles **loading rules**, **talking to the user**, and **saving the data**.

### 1. Setup and Loading Functions (I/O)

These functions get the chatbot ready by importing tools and reading the rules.

| Function | What it does (Simple) |
| :--- | :--- |
| `import ...` | **Brings in tools** (like `json`, `sqlite3`, and `TextBlob`) that the code needs to work. |
| `load_responses()` | **Reads the main rules file** (`responses.json`). This file contains all questions, keywords, and replies. |
| `RESPONSES = load_responses()` | **Saves the rules** into a variable so the whole program can use them easily. |

### 2. Conversation Processing Functions (The Brain)

These functions look at what the user types and figure out the best response.

| Function | What it does (Simple) |
| :--- | :--- |
| `get_sentiment_score()` | **Checks the user's mood** in the text (is it sad, happy, or neutral?) using the TextBlob tool. |
| `get_empathetic_reply_and_key()` | **Finds the right answer** during the interview. It searches the user's reply for a keyword and sends a caring response, then saves the keyword's meaning (e.g., saves "Yes" if the user replied with "I agree"). |
| `check_mood_keywords()` | **Looks for specific mood words** ("good," "bad," "tired") at the start of the chat to decide if the interview should begin. |

### 3. Database Functions (Storage)

These functions manage saving all the information the user gives us.

| Function | What it does (Simple) |
| :--- | :--- |
| `setup_database()` | **Prepares the database** (`moodmate.db`). It creates the main table (`interviews`) and makes sure all columns for saving data are ready. |
| `save_interview()` | **Writes the collected data** (answers from the user) into the database table, along with the current time. |

### 4. The Main Chat Loop (Execution)

This is the part that runs the actual conversation.
* `conversation_state`: This is a storage box that tracks where the user is in the conversation (e.g., greeting, in the middle of a question, or finished).
* `while True`: This loop keeps the chatbot running forever until the user types an exit word.
* `Mode Checks`: This logic decides what to do next. It checks the `conversation_state`:
    * If it's in greeting mode, it checks the mood to start the interview.
    * If it's in interview mode, it processes the user's answer, saves the data, and moves to the next question.
*(After a while, I can run the code on the terminal, and the chatbot will start and do everything needed (or: perform all its required tasks).)*

---

## ✨ Chatbot 1: Streamlit Deployment Summary

This code combines the **Chatbot Logic**, the **Machine Learning Model**, and the **Streamlit Web Interface** into one application.

### 1. ⚙️ ML Logic: Data Preparation & Prediction

The main idea here is to make the user's text answers ready for the ML model.

* `load_resources()`: This function is the most important. It loads the trained model and the rules/solutions files (`.json`) only once when the app starts, which makes the app fast.
* `get_standard_key()`: Translates user answers (like "نعم" or "أكيد") into one standardized key ('Yes') so the model can understand it easily.
* `get_prediction_from_user_input()`: This is the full prediction pipeline. It takes the user's collected answers, converts them into 30 fixed numerical features (columns) that the model expects, and then asks the model for a prediction percentage (the risk score).

### 2. 💬 Conversation Logic: Flow and Solutions

This section manages the chat experience and the final results.
* `st.session_state`: This is the app's memory. It stores the current chat history, the user's answers, and the bot's current state (`mode`) between messages.
* **State Machine Logic**: This is the core flow. It guides the user step-by-step through different stages:
    * **Greeting** (checking mood).
    * **Interview** (asking questions and saving answers).
    * **Displaying** the Prediction Score (the percentage).
    * **Solutions Menu** (finding the problems based on answers and offering advice).
* `build_solutions_menu()` & `format_solution()`: These functions look at the user's answers, identify the problems they might be facing, and present organized solutions, descriptions, and resource links (videos/podcasts).

*(Now, we can run the code on the terminal inside our virtual environment (`venv`). This version is a better example of how the chatbot will converse. It's a simple chat deployment example to describe the recipe (or: to describe how to make something).)*

---

## 💻 Chatbot 2: Front-End & Back-End Split

### Structure

The chatbot interface is divided into **three main parts**:
* **Front-End**: Includes HTML, CSS, and **JavaScript (JS)**.
* **Back-End**.

The **two most important files** for us are the **JavaScript** file and the **Back-End** file.

### 🌐 Front-End vs. 🧠 Back-End Comparison (Deployment)

| Feature | JavaScript File (Front-End) | Python File (Back-End/API) |
| :--- | :--- | :--- |
| **Primary Role** | **Manages the User Interface (UI)**, conversation flow, and displays messages/solutions. | **Manages the ML Model** (Prediction & Data Processing), and business logic. |
| **Execution Location** | The **user's web browser** (Client-side). | The **web server** (must be constantly running). |
| **Key Data Handled** | **Conversation Rules** (Questions/Replies) and **Display Format** (HTML/CSS). | The **Trained ML Model** (`.joblib`) and complex numerical transformation logic. |
| **Crucial Action** | Sends user answers to the API (`sendDataToAPI`) and waits for the result. | Receives data, executes the prediction pipeline, and returns the **Stability Score** and the **Solutions Report**. |
| **File** | `app.js` | `api_server`|

---

### 2. 🌐 Front-End Code (JavaScript/HTML) Details

This code manages everything the user sees and interacts with in the web browser.

| Function/Concept | Simple Summary |
| :--- | :--- |
| **Embedded Data** | **Contains all the text** (greetings, questions, farewells, solutions) the bot will use. |
| `addUserMessage()` / `addBotMessage()` | **Displays messages** on the screen, showing them as User or Bot. |
| `parseMarkdown()` | **Converts text features** (like **bold** text and URL links) into styled web elements (like buttons). |
| `getStoredKey()` | **Matches user replies** to specific keywords to save the standardized answer (e.g., matching "نعم" to 'Yes'). |
| `sendDataToAPI()` | **Communicates with the Back-End** (Python server) by sending all collected answers to get the prediction result. |
| **Event Listeners** | **Handles user actions**, primarily managing the chat input box and the button clicks. |

### 📥 Primary Data Fields (Required Order)

The developer needs the following data fields, collected from the user's answers, to be maintained in this **exact order** when sent to the API/Back-End for processing:

1.  **Gender**
2.  **self_employed**
3.  **family_history**
4.  **Days_Indoors**
5.  **Growing_Stress**
6.  **Changes_Habits**
7.  **Mental_Health_History**
8.  **Mood_Swings**
9.  **Coping_Struggles**
10. **Work_Interest**
11. **Social_Weakness**
12. **mental_health_interview**
13. **care_options**
14. **Occupation**
15. **Country**

---

### 3. 🧠 Back-End Code (Python/FastAPI - API Server) Details

This code is the core logic that handles the prediction and is exposed as a service (API) for the Front-End to use.

| Function/Concept | Simple Summary |
| :--- | :--- |
| `load_resources()` | **Loads the trained ML model** (`.joblib`) and the `solutions.json` file into the server's memory. |
| `get_standard_key()` | **Cleans and standardizes** the user's answers (e.g., 'طالب' to 'Student') to prepare them for the model. |
| `get_prediction_from_user_input()` | **The main ML pipeline.** It transforms all user answers into the exact 30 numerical features required by the model and runs the prediction. |
| `build_solutions_report()` | **Selects relevant advice** by matching the user's answers to the problem triggers in the `solutions.json` file. |
| `@app.post("/predict_health")` | **The API endpoint.** It is the specific URL the JavaScript calls to send the data and receive the final stability score and advice. |

---

### To test this chatbot, you need to go to the HTML code and select **"Open with Live Server"**

---

## 🚀 Chatbot Version 3: Web Deployment

The third chatbot was created by modifying the JavaScript and Back-End code to successfully **deploy the project on a web server** and turn it into a live website. We split the components for efficient hosting:

### Hugging Face Spaces 😊

To prepare the code for deployment on **Hugging Face Spaces**, we needed a few adjustments. We created a new file, separate from the main project, and copied the **Back-End** file into it, naming it **`app.py`**. We also needed to include our model, **`health_chatbot_model.joblib`**. To successfully deploy the Back-End, we had to create some necessary support files like **`requirements.txt`**, a **`Dockerfile`**, a **`space-start.sh`** script and **`.gitattributes`**.

#### ⚙️ Deployment Setup Files

* **`requirements.txt`** -> **Dependencies List**: Lists all the Python libraries (e.g., `fastapi`, `uvicorn`, `pandas`, `joblib`, `numpy`) that the Back-End code needs to run successfully on the server.
* **`Dockerfile`** -> **Environment Configuration**: Specifies the base operating system image and the sequential steps needed to set up the environment, install dependencies, and prepare the application.
* **`space-start.sh`** -> **Startup Command**: A shell script that tells the Hugging Face server the exact command to start the FastAPI application, ensuring it runs on the correct host and port (e.g., `uvicorn app:app --host 0.0.0.0 --port 7860`).

#### 🧠 Final Back-End Code (FastAPI) Summary

This Python code acts as the central **API service** for the chatbot. Its main job is to receive user data, perform the complex Machine Learning prediction, and prepare the customized solution report.

##### 1. Key Functions (What it Does)

| Function/Concept | Simple Summary |
| :--- | :--- |
| **Model Loading** | **Loads the trained ML Model** (`.joblib`) into the server's memory once at startup, ready for instant use. |
| `get_prediction_from_user_input()` | **ML Prediction Pipeline.** Takes user answers, performs all 8 steps of data processing (encoding, scaling, feature engineering), and returns the **final stability score percentage**. |
| `build_solutions_report()` | **Custom Advice Generator.** Matches user answers to problems and selects random descriptions, solutions, and resource links (videos/podcasts) to send to the Front-End. |
| `@app.post("/predict_health")` | **API Endpoint.** This is the specific URL the JavaScript (Front-End) uses to send the final survey data and receive the results. |
| **HTML Button Generation** | **Sends styled HTML** for video and podcast links directly to the Front-End, reducing the complexity on the JavaScript side. |

##### 2. Major Changes and Rationale (Why it changed)

The final version of this code was designed for reliable deployment on platforms like Hugging Face Spaces:

* **Integrated Data (Self-Contained):**
    * **Change:** The **Solutions** and **Responses** data (previously in external `.json` files) were **added directly into the Python code**.
    * **Reason:** To make the API **self-reliant** (self-contained). This ensures the code always has its rules and advice, simplifying deployment and avoiding errors when trying to read external files on a server.
* **Full ML Pipeline Activation:**
    * **Change:** The dummy prediction logic was replaced with the **full 8-step preprocessing pipeline** and the actual call to the loaded model.
    * **Reason:** To allow the API to perform its **actual job**, which is calculating the statistical prediction based on the user's answers, instead of returning a placeholder score.

### 🌐 Vercel Deployment Setup (Front-End)

To prepare the Front-End code for deployment on **Vercel**, we followed these steps:

* **File Preparation** -> We created a **new file** (e.g., `index.html`) and copied the **entire Front-End code** into it.
* **JavaScript Modification** -> The JavaScript code was slightly modified to ensure it **remains self-contained** and uses the data already embedded within it (HTML/JS file), rather than making external calls for rules.
* **API Connection** -> The key step was linking the JavaScript to the Back-End by updating the **`API_URL`** to use the new service address obtained from the **Hugging Face Spaces deployment**.

---

| Component | Hosting Platform | Project Link |
| :--- | :--- | :--- |
| **Back-End (The API / Python Logic)** | Hugging Face Spaces | [Hugging Face Space Link](https://huggingface.co/spaces/Adhamelmalhy/Chatbot?logs=container) |
| **Front-End (The User Interface / HTML, CSS, JS)** | Vercel | [Vercel Project Link](https://chatbot-adham-mahmouds-projects.vercel.app/) |

This separation ensures **scalability** and efficient handling of both the ML prediction service and the user interface.

---

## 🔗 Final Live Project Link

### 🌐 The Deployed Chatbot is Live! 🎉

You can access and test the final deployed Front-End here:

**[LIVE CHATBOT LINK (Vercel)](https://chatbot-psi-dusky.vercel.app/)**

*Note: This link relies on the Back-End API deployed on Hugging Face Spaces being active.*

---

## 💡 Future Improvements and Challenges

This list outlines the suggested enhancements that can be implemented to develop the project in future stages:

* **Persistent Database (PostgreSQL):** Integrate a PostgreSQL database for permanently storing conversation logs, user answers, and predictions.
* **Multilingual Support:** Add support for other languages (such as English) to make the chatbot available to a wider segment of users.
* **Front-End Features:** Incorporate new and enhanced interactive features in the Front-End to improve the user experience.
* **Multi-Model State Challenge:** Integrate an additional machine learning model with another dataset. The goal is to evolve the chatbot to operate in **two different states** (e.g., one state for psychological consultation and one for general information/education), instead of relying on a single state.