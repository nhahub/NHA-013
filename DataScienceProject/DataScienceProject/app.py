import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Mental Health Dashboard", layout="wide")

st.title("🧠 Mental Health Interactive Dashboard")

# -------------------------------------------------
# OPTION B: Load CSV automatically from your device
# -------------------------------------------------
try:
    df = pd.read_csv("df_encoded_data_deploy.csv")
    st.success("Dataset loaded successfully! (encoded_dataset.csv)")
except:
    st.error("❗ Could not find 'encoded_dataset.csv'. Make sure it is in the same folder as app.py.")
    st.stop()

st.write("### Preview of Your Encoded Dataset")
st.dataframe(df.head())

# ===================================================
# ----- SIDEBAR FILTERS -----
# ===================================================
st.sidebar.header("🔍 Filters")

# Gender filter (0 = Male, 1 = Female)
gender_filter = st.sidebar.selectbox(
    "Select Gender",
    options=["All", 0, 1],
    format_func=lambda x: "All" if x == "All" else ("Male" if x == 0 else "Female")
)

# Stress level slider
stress_min, stress_max = st.sidebar.slider(
    "Growing Stress Level",
    min_value=int(df["Growing_Stress"].min()),
    max_value=int(df["Growing_Stress"].max()),
    value=(int(df["Growing_Stress"].min()), int(df["Growing_Stress"].max()))
)

# Mood swings slider
mood_min, mood_max = st.sidebar.slider(
    "Mood Swings (0=Low,1=Medium,2=High)",
    min_value=0, max_value=2, value=(0, 2)
)

# Days Indoors selector
days_filter = st.sidebar.multiselect(
    "Select Days Indoors",
    options=sorted(df["Days_Indoors"].unique()),
    default=sorted(df["Days_Indoors"].unique())
)

# Apply filters
filtered_df = df.copy()

if gender_filter != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == gender_filter]

filtered_df = filtered_df[
    (filtered_df["Growing_Stress"] >= stress_min) &
    (filtered_df["Growing_Stress"] <= stress_max) &
    (filtered_df["Mood_Swings"] >= mood_min) &
    (filtered_df["Mood_Swings"] <= mood_max) &
    (filtered_df["Days_Indoors"].isin(days_filter))
]

# ===================================================
# ----- GRAPH CHOOSER -----
# ===================================================
st.sidebar.header("📊 Choose a Graph")

graph_choice = st.sidebar.selectbox(
    "Graph Type",
    [   "Treatment",
        "Country vs Treatment",
        "Stress vs treatment",
        "Mood Swings vs Treatment",
        "History vs Treatment",
        "Stress by Gender",
        "Days Indoors",
        "Days Indoors vs Treatment"
        
    ]
)

# ===================================================
# ----- GRAPHS -----
# ===================================================
st.subheader(f"📌 {graph_choice}")

if graph_choice == "Stress vs treatment":
    stress_map = {
        0: "YES",
        1: "NO",
        2:"Maybe"
   # adjust based on your actual categories
    }
    filtered_df["Stress_Label"] = filtered_df["Growing_Stress"].map(stress_map)
    fig = px.histogram(
        filtered_df,
        x="Stress_Label",
        color="treatment",
       barmode="group"
    )

    fig.update_layout(
        xaxis_title="Growing Stress Level",
        yaxis_title="Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)
elif graph_choice == "Mood Swings vs Treatment":
    
    # Map the encoded values to readable labels
    mood_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    filtered_df["Mood_Label"] = filtered_df["Mood_Swings"].map(mood_map)

    fig = px.histogram(
        filtered_df,
        x="Mood_Label",
        color="treatment",
        barmode="group",
        category_orders={"Mood_Label": ["Low", "Medium", "High"]}  # correct order
    )

    fig.update_layout(
        xaxis_title="Mood Swings",
        yaxis_title="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

elif graph_choice == "History vs Treatment":
    
    # Map encoded values to human-friendly labels
    history_map = {
        0: "Yes",
        1: "NO",
        2: "Maybe"
    }

    filtered_df["History_Label"] = filtered_df["Mental_Health_History"].map(history_map)

    fig = px.histogram(
        filtered_df,
        x="History_Label",
        color="treatment",
        barmode="group",
    )

    fig.update_layout(
        xaxis_title="Mental Health History",
        yaxis_title="Count"
    )

    st.plotly_chart(fig, use_container_width=True)


elif graph_choice == "Stress by Gender":

    # Map Growing_Stress encoded values → readable labels
    stress_map = {
        0: "YES",
        1: "NO",
        2:"Maybe"
   # adjust based on your actual categories
    }
    filtered_df["Stress_Label"] = filtered_df["Growing_Stress"].map(stress_map)

    # Map Gender encoded values → readable labels
    gender_map = {
        0: "Male",
        1: "Female"
    }
    filtered_df["Gender_Label"] = filtered_df["Gender"].map(gender_map)

    fig = px.histogram(
        filtered_df,
        x="Stress_Label",
        color="Gender_Label",
        category_orders={"Stress_Label": list(stress_map.values())}
    )

    fig.update_layout(
        xaxis_title="Growing Stress Level",
        yaxis_title="Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)


elif graph_choice == "Days Indoors":
    
    # Map numeric codes → human-friendly labels
    days_labels = {
        0: "Go out every day",
        1: "1–14 days",
        2: "15–30 days",
        3: "31–60 days",
        4: "More than 2 months"
    }

    # Replace codes with labels in a temporary column
    filtered_df["Days_Label"] = filtered_df["Days_Indoors"].map(days_labels)

    # Define custom color for each category
    colors = {
        "Go out every day": "#1f77b4",
        "1–14 days": "#ff7f0e",
        "15–30 days": "#2ca02c",
        "31–60 days": "#d62728",
        "More than 2 months": "#9467bd"
    }

    fig = px.histogram(
        filtered_df,
        x="Days_Label",
        category_orders={"Days_Label": list(colors.keys())},  # force fixed order
        color="Days_Label",
        color_discrete_map=colors
    )



    st.plotly_chart(fig, use_container_width=True)
elif graph_choice == "Days Indoors vs Treatment":
    days_labels = {
        0: "Go out every day",
        1: "1–14 days",
        2: "15–30 days",
        3: "31–60 days",
        4: "More than 2 months"
    }

    # Replace codes with labels in a temporary column
    filtered_df["Days_Label"] = filtered_df["Days_Indoors"].map(days_labels)
    fig = px.histogram(
        filtered_df,
        x="Days_Label",
        color="treatment",
    )
    st.plotly_chart(fig, use_container_width=True)
elif graph_choice == "Treatment":
    days_labels = {
        0: "No Treatment",
        1: "Treatment",

    }
    filtered_df["treatment"] = filtered_df["treatment"].map(days_labels)
   
    fig = px.histogram(
        filtered_df,
        x="treatment",

    )
    st.plotly_chart(fig, use_container_width=True)
elif graph_choice == "Country vs Treatment":
    days_labels = {
        0: "No Treatment",
        1: "Treatment",

    }
    filtered_df["treatment"] = filtered_df["treatment"].map(days_labels)

    fig = px.histogram(
        filtered_df,
        x="Country",
        color="treatment"

    )
    st.plotly_chart(fig, use_container_width=True)
