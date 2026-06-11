import streamlit as st
import pickle
import pandas as pd
import base64
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Smart Taxi Fare AI",
    page_icon="🚕",
    layout="wide"
)

# =========================================================
# LOAD MODEL & SCALER (CACHED FOR SPEED)
# =========================================================
@st.cache_resource
def load_assets():
    model = pickle.load(open("taxi_price_model.save", "rb"))
    scaler = pickle.load(open("taxi_price_scaler.save", "rb"))
    le_time = pickle.load(open("le_time_of_day.save", "rb"))
    le_day = pickle.load(open("le_day_of_week.save", "rb"))
    le_traffic = pickle.load(open("le_traffic_conditions.save", "rb"))
    le_weather = pickle.load(open("le_weather.save", "rb"))
    return model, scaler, le_time, le_day, le_traffic, le_weather

model, scaler, le_time, le_day, le_traffic, le_weather = load_assets()

# =========================================================
# SESSION STATE
# =========================================================
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False

# =========================================================
# BACKGROUND IMAGE PROCESSING
# =========================================================
def get_base64(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

bg_img = get_base64("bg.jpg")
bg_style = f'url("data:image/jpg;base64,{bg_img}")' if bg_img else "#0f0f12"

# =========================================================
# PREMIUM CUSTOM CSS
# =========================================================
st.markdown(f"""
<style>
/* Global App Background overrides */
.stApp {{
    background-image: linear-gradient(rgba(10, 10, 15, 0.88), rgba(10, 10, 15, 0.93)), {bg_style};
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #e0e0e6;
}}

/* Clean up header elements */
#MainMenu {{visibility:hidden;}}
footer {{visibility:hidden;}}
header {{visibility:hidden;}}

/* Typography Styles */
.main-title {{
    text-align: left;
    font-size: 52px;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #00F5FF, #FF00FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}}

.subtitle {{
    text-align: left;
    color: #a0a0b0;
    font-size: 18px;
    margin-bottom: 40px;
}}

/* Glassmorphic Input & Section Cards */
div[data-testid="stVerticalBlock"] > div {{
    background: rgba(255, 255, 255, 0.02);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 5px;
}}

/* Premium CTA Glowing Button Style */
.stButton > button {{
    width: 100%;
    height: 54px;
    border: none !important;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 700;
    background: linear-gradient(90deg, #00f5ff, #3333ff) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(0, 245, 255, 0.2);
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 245, 255, 0.4);
    background: linear-gradient(90deg, #ff00cc, #3333ff) !important;
}}

/* Secondary Action / Home Button Override */
div.home-btn-container .stButton > button {{
    background: rgba(255, 255, 255, 0.07) !important;
    color: #e0e0e6 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: none;
}}

div.home-btn-container .stButton > button:hover {{
    background: rgba(255, 255, 255, 0.15) !important;
    box-shadow: none;
    transform: translateY(-1px);
}}

/* Prediction Result Display Wrapper */
.result-card {{
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
    background: linear-gradient(135deg, rgba(0, 255, 153, 0.1), rgba(0, 50, 30, 0.4)) !important;
    border: 2px solid #00ff99 !important;
    box-shadow: 0 0 30px rgba(0, 255, 153, 0.25);
    margin-top: 25px;
    margin-bottom: 25px;
    animation: pulse 2s infinite alternate;
}}

/* Clean metric blocks */
div[data-testid="stMetricValue"] {{
    font-size: 28px;
    font-weight: 700;
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HOME PAGE
# =========================================================
if not st.session_state.show_prediction:

    # Top Section Hero Bar
    title_col, action_col = st.columns([5, 2])
    
    with title_col:
        st.markdown("<div class='main-title'>🚕 SMART TAXI FARE AI</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Advanced Machine Learning Based Taxi Fare Prediction</div>", unsafe_allow_html=True)
        
    with action_col:
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Predict Trip Fare"):
            st.session_state.show_prediction = True
            st.rerun()

    # Creative Content Presentation
    body_left, body_right = st.columns([4, 3])
    
    with body_left:
        st.markdown("""
        ### 🔮 Intelligent Fare Forecasting
        Avoid price-gouging and hidden surcharges. Our predictive system interprets live structural criteria to determine baseline taxi expenditures before execution.
        
        #### 🎯 Input Telemetry Analyzed
        * **Spatial Metrics:** Track exact spatial distance trends.
        * **Environmental Overhead:** Real-time calculation of atmospheric weather conditions.
        * **Temporal & Congestion Impact:** Peak traffic variations mapped alongside chronological hours of operations.
        
        ### 🚀 Integrated Technology Stack
        The underlying computing framework integrates **Scikit-Learn StandardScalers** with pipeline regressors served seamlessly via state-aware multi-threaded python execution contexts.
        """)
        
    with body_right:
        if os.path.exists("taxi.jpg"):
            st.image("taxi.jpg", use_container_width=True)
        else:
            # Fallback visually clean element if target image asset isn't local
            st.info("💡 Pro Tip: Fill out trip variables precisely to evaluate optimal baseline metrics accurately.")

    st.markdown("---")

# =========================================================
# PREDICTION PAGE
# =========================================================
else:

    # Header Row with Back Button on the Far Right
    title_col, action_col = st.columns([5, 2])
    
    with title_col:
        st.markdown("<div class='main-title'>🔮 PREDICT TRIP PRICE</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Configure specific trip parameters below</div>", unsafe_allow_html=True)
        
    with action_col:
        st.markdown("<div style='margin-top: 25px;' class='home-btn-container'></div>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<div class='home-btn-container'>", unsafe_allow_html=True)
            if st.button("⬅ Return Home"):
                st.session_state.show_prediction = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Main Layout Form Matrix
    input_left, input_right = st.columns(2)

    with input_left:
        with st.container():
            st.markdown("<h4 style='color:#00F5FF;'>📏 Distance & Rates</h4>", unsafe_allow_html=True)
            Trip_Distance_km = st.number_input("Trip Distance (km)", min_value=0.0, step=0.1 )
            Base_Fare = st.number_input("Base Fare (₹)", min_value=0.0)
            Per_Km_Rate = st.number_input("Per Km Rate (₹)", min_value=0.0)
            Per_Minute_Rate = st.number_input("Per Minute Rate (₹)", min_value=0.0)

    with input_right:
        with st.container():
            st.markdown("<h4 style='color:#FF00FF;'>⏱ Timing & Conditions</h4>", unsafe_allow_html=True)
            Trip_Duration_Minutes = st.number_input("Trip Duration (Minutes)", min_value=0)
            Passenger_Count = st.number_input("Passenger Count", min_value=1,max_value=7, step=1)
            Time_of_Day = st.selectbox("Time of Day", list(le_time.classes_))
            Day_of_Week = st.selectbox("Day of Week", list(le_day.classes_))
            Traffic_Conditions = st.selectbox("Traffic Overhead", list(le_traffic.classes_))
            Weather = st.selectbox("Weather Condition", list(le_weather.classes_))

    st.markdown("<br>", unsafe_allow_html=True)

    # Sticky Action Bar Bottom Area
    bottom_col1, bottom_col2, bottom_col3 = st.columns([2, 3, 2])
    
    with bottom_col2:
        predict_clicked = st.button("🚕 Compute Accurate Fare")

    if predict_clicked:
        # Build base dictionary array matching exactly the initial features map
        input_data = pd.DataFrame([{
            "Trip_Distance_km": Trip_Distance_km,
            "Time_of_Day": le_time.transform([Time_of_Day])[0],
            "Day_of_Week": le_day.transform([Day_of_Week])[0],
            "Passenger_Count": Passenger_Count,
            "Traffic_Conditions": le_traffic.transform([Traffic_Conditions])[0],
            "Weather": le_weather.transform([Weather])[0],
            "Base_Fare": Base_Fare,
            "Per_Km_Rate": Per_Km_Rate,
            "Per_Minute_Rate": Per_Minute_Rate,
            "Trip_Duration_Minutes": Trip_Duration_Minutes
        }])

        # Process standard transformation safely keeping original column headers intact
        scaled_array = scaler.transform(input_data)
        input_scaled_df = pd.DataFrame(scaled_array, columns=input_data.columns)

        # Execute safe runtime calculation
        prediction = model.predict(input_scaled_df)
        fare = max(0.0, round(prediction[0], 2))

        # Output UI Element display
        st.markdown(f"""
        <div class='result-card'>
            💰 Est. Total Fare: ₹ {fare:,}
        </div>
        """, unsafe_allow_html=True)

      