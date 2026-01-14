import streamlit as st
import pandas as pd
import joblib
import pytesseract
from PIL import Image
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


stop_names = {
    1: "Sitabuldi",
    2: "Railway Station",
    3: "Dharampeth",
    4: "Ajni",
    5: "MIHAN",
    6: "Civil Lines",
    7: "Manewada",
    8: "Sadar",
    9: "Medical Square",
    10: "Wardhaman Nagar"
}

# Passenger flow profile (Nagpur)
# Negative = people usually board here
# Positive = people usually get down here
stop_flow = {
    "Sitabuldi": -3,
    "Railway Station": 5,
    "Dharampeth": -1,
    "Ajni": -2,
    "MIHAN": 6,
    "Civil Lines": -1,
    "Manewada": 2,
    "Sadar": -1,
    "Medical Square": 3,
    "Wardhaman Nagar": 4,
    "Zero Mile" : -2
}

route_sequences = {
    "Sitabuldi → Railway Station": ["Sitabuldi", "Zero Mile", "Civil Lines", "Railway Station"],
    "Sitabuldi → MIHAN": ["Sitabuldi", "Zero Mile", "Ajni", "MIHAN"],
    "MIHAN → Sitabuldi": ["MIHAN", "Ajni", "Zero Mile", "Sitabuldi"],
    "Civil Lines → Sadar": ["Civil Lines", "Zero Mile", "Sadar"],
    "Manewada → Wardhaman Nagar": ["Manewada", "Medical Square", "Wardhaman Nagar"]
}



model = joblib.load("final_nagpur_crowd_model.pkl")

st.set_page_config(page_title="Nagpur Bus Crowd Prediction", layout="centered")

st.title("🚍 Nagpur Public Transport Crowd Prediction")
st.write("Real-time overcrowding prediction using ML")

st.divider()


st.subheader("Enter Live Bus Details")

vehicle_id = st.number_input("Vehicle ID", value=102)
boarding_stop = st.selectbox("Boarding Stop", list(stop_names.values()))
dropping_stop = st.selectbox("Dropping Stop", list(stop_names.values()))


if boarding_stop == dropping_stop:
    st.error("❌ Boarding and Dropping stop cannot be the same")
    st.stop()


stop_id = [k for k, v in stop_names.items() if v == boarding_stop][0]

route_name = f"{boarding_stop} → {dropping_stop}"
route_id = abs(hash(route_name)) % 1000

st.success(f"🛣️ Route: {route_name}  |  Route ID: {route_id}")


stop_name = stop_names.get(stop_id, "Unknown Stop")


capacity = st.number_input("Bus Capacity", value=50)
ticket_mode = st.radio("Ticket Input Mode", ["Manual", "OCR Camera"])

ticket_count = None

if ticket_mode == "Manual":
    ticket_count = st.slider("Tickets scanned (last 5 min)", 0, capacity, min(10, capacity))

else:
    uploaded = st.file_uploader("Upload Ticket Image", type=["jpg","png","jpeg"])
    if uploaded is None:
        st.stop()

    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Ticket")

    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    config = "--psm 6"
    text = pytesseract.image_to_string(gray, config=config)

    import re
    numbers = re.findall(r'\b[1-5]\b', text)

    if not numbers:
        st.error("❌ Could not detect passenger number")
        st.stop()

    ticket_count = int(numbers[0])
    st.success(f"🎫 Passenger Count Detected: {ticket_count}")


st.subheader("Time & Day")

day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])

boarding_time = st.time_input("Boarding Time")
hour = boarding_time.hour   # what ML uses

weekday = st.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

day_type_val = 0 if day_type == "Weekday" else 1
weekday_val = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(weekday)



if st.button("Predict Crowd Level"):
    st.caption("Crowd level is based on historical patterns, not just current occupancy")

    live_input = {
        "vehicle_id": vehicle_id,
        "route_id": route_id,
        "stop_id": stop_id,
        "ticket_count": ticket_count,
        "capacity": capacity,
        "day_type": day_type_val,
        "hour": hour,
        "weekday": weekday_val
    }

    df_live = pd.DataFrame([live_input])
    pred = model.predict(df_live)[0]


    crowd_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    predicted_crowd = crowd_map[pred]

    route_key = f"{boarding_stop} → {dropping_stop}"
    path = route_sequences.get(route_key)

    if path:
        start = path.index(boarding_stop)
        end = path.index(dropping_stop)

        trend = 0
        for stop in path[start+1:end+1]:
            trend += stop_flow.get(stop, 0)

        if trend > 0:
            journey_insight = f"🔽 Crowd will reduce as you approach {dropping_stop}"
        elif trend < 0:
            journey_insight = f"🔼 Crowd will increase as you approach {dropping_stop}"
        else:
            journey_insight = "➡️ Crowd will remain stable during this journey"
    else:
        journey_insight = "⚠️ No crowd-flow data for this route"



    raw_occupancy = (ticket_count / capacity) * 100
    occupancy = round(min(raw_occupancy, 100), 2)


    if occupancy < 20 and predicted_crowd == "MEDIUM":
        predicted_crowd = "LOW"


    if raw_occupancy > 100:
        alert = "🔴 Overcapacity detected (immediate action required)"
    elif raw_occupancy >= 100 and predicted_crowd == "LOW":
        alert = "🔴 Bus is calm but fully occupied. Safety action required."
    elif predicted_crowd == "HIGH" and raw_occupancy > 80:
        alert = "🔴 Overcrowding risk detected"
    elif predicted_crowd == "MEDIUM":
        alert = "🟡 Moderate crowd, monitor closely"
    else:
        alert = "🟢 Crowd level normal"


    st.divider()
    st.subheader("Prediction Result")
    st.write(f"🚌 Route: {route_name}")
    st.write(f"📍 Stop: {stop_name}")
    st.write(f"📍 Boarding Stop: {boarding_stop}")
    st.write(f"🏁 Dropping Stop: {dropping_stop}")



    
    if "🔴" in alert:
        st.error(alert)
    elif "🟡" in alert:
        st.warning(alert)
    else:
        st.success(alert)

    st.metric("Occupancy (%)", occupancy)

   
    st.markdown(f"**Expected Crowd Pattern:** {predicted_crowd}")
    st.info(journey_insight)

