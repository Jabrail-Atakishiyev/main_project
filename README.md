# 🏡 Airbnb Price and License Prediction App

This project allows users to predict the **estimated price** and **license status** of Airbnb listings. The app is built using **Streamlit** and uses **XGBoost** for regression and **LightGBM** for classification tasks.

## 📊 Data Source

The data used in this project is sourced from the [Inside Airbnb](https://insideairbnb.com/get-the-data/) platform.

## ⚙️ Technologies Used

- Python  
- Streamlit  
- Pandas  
- Scikit-learn  
- XGBoost  
- LightGBM  

## 🚀 How to Run

1. Install the required packages:

```bash
pip install -r requirements.txt
````

2. Run the Streamlit app:

```bash
streamlit run airbnb_streamlit.py
```

## 📌 Features

✅ Based on the Airbnb listing input, the app can:

### Price Prediction

Estimate the nightly price using features such as city, neighborhood, room type, minimum nights, availability, and host listing count.

### License Prediction

Predict whether a listing is licensed (YES / NO) based on given properties using a classification model.
