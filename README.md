# Real-Time Public Transportation Overcrowding Prediction (Nagpur)

## Overview
This project predicts real-time crowd levels in Nagpur city buses using
Machine Learning and OCR-based ticket analysis. It helps passengers and
transport authorities understand overcrowding patterns and take proactive actions.

## Key Features
- OCR-based passenger count detection from bus tickets
- Manual and camera-based ticket input
- Boarding and dropping stop selection
- Route-based journey crowd insights
- Time and day-based crowd prediction
- ML-based classification (LOW / MEDIUM / HIGH)

## Technology Stack
- Python
- Streamlit
- OpenCV
- Pytesseract OCR
- Scikit-learn (Random Forest)
- Pandas, NumPy

## How It Works
1. Passenger uploads ticket image or enters ticket count manually
2. OCR extracts passenger count
3. Route and stop data are processed
4. ML model predicts crowd level
5. Journey insight shows whether crowd will increase or decrease

## Dataset
- Synthetic Nagpur public transport dataset
- Includes route, stop, time, capacity, and ticket data

## Output
- Crowd level prediction
- Occupancy percentage
- Journey crowd trend
- Safety alerts

## Future Enhancements
- Live CCTV-based passenger counting
- GPS-based real-time bus tracking
- Deep learning OCR for better accuracy
- Mobile app integration
- Multi-city scalability

## Journey Insights Feature
-This update adds journey-level crowd trend prediction.
-The system estimates whether crowd will increase or decrease
-between boarding and dropping stops using stop-flow logic.


## Author
Meet Sarda
