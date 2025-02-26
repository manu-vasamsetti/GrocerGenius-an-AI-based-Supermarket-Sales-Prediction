# Grocery Sales Prediction App

## Overview
This project is a **Grocery Sales Prediction App** built using **Streamlit**. The app allows users to input product and store details to predict the sales of a grocery item using a pre-trained **XGBoost** model.

## Features
- User-friendly web interface powered by **Streamlit**
- Predicts sales based on multiple product and store attributes
- Handles data preprocessing, including outlier handling and encoding
- Uses **XGBoost** for accurate predictions

## Requirements
Ensure you have the following dependencies installed before running the app:

```sh
pip install streamlit pandas numpy joblib xgboost
```

## How to Run
To launch the application, navigate to the project directory and run:

```sh
python -m streamlit run app.py
```

## Files Included
- `app.py` - Main Streamlit application
- `xgb_model.pkl` - Trained XGBoost model
- `ohe_encoder.pkl` - One-Hot Encoder
- `loo_encoder.pkl` - Leave-One-Out Encoder
- `ordinal_encoder.pkl` - Ordinal Encoder

## Usage
1. Open the app in a browser after running the command.
2. Enter the required product and store details.
3. Click on **Predict Sales** to get the prediction.

## Notes
- This project is for educational and demonstration purposes.
- The prediction results are based on a trained machine learning model and may not be 100% accurate.

## License
This project is open-source and available for personal and educational use.

