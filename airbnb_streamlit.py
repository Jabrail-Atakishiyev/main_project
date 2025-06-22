import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier

# --- Faylları yüklə (caching ilə) ---
@st.cache_data
def load_data():
    df_cleaned = pd.read_csv("cleaned_airbnb_2.csv")
    df_original = pd.read_csv("airbnb_data.csv")
    return df_cleaned, df_original

df_cleaned, df_original = load_data()

# --- Lazım olan sütunlar ---
categorical_cols = ['city', 'neighbourhood', 'room_type']
numerical_cols = ['minimum_nights', 'availability_365', 'calculated_host_listings_count', 'number_of_reviews_ltm']
target_cols = ['price', 'license']

# --- Dataframe-ləri birləşdir ---
categorical_data = df_original[categorical_cols].reset_index(drop=True)
numerical_data = df_cleaned[numerical_cols + target_cols].reset_index(drop=True)
df_merged = pd.concat([categorical_data, numerical_data], axis=1)

# --- Başlıq ---
st.title("Airbnb Qiymət və Lisenziya Proqnozu")

# --- Seçimlər ---
unique_cities = sorted(df_original['city'].dropna().unique().tolist())
room_types = sorted(df_original['room_type'].dropna().unique().tolist())

# --- Model təlim funksiyaları ---
@st.cache_resource
def train_price_model(X, y):
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['city', 'neighbourhood', 'room_type'])],
        remainder='passthrough'
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(enable_categorical=False, verbosity=0))
    ])
    model.fit(X, y)
    return model

@st.cache_resource
def train_license_model(X, y):
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['city', 'neighbourhood', 'room_type'])],
        remainder='passthrough'
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBRegressor())
    ])
    model.fit(X, y)
    return model

# --- Proqnoz növü ---
option = st.radio("Proqnoz növünü seç:", ["Price Proqnozu", "License Proqnozu"])

# --------------------------------
# PRICE PROQNOZU
# --------------------------------
if option == "Price Proqnozu":
    st.header("Qiymət Proqnozu")

    city = st.selectbox("Şəhər seçin", unique_cities, key="city_price")
    filtered_neighs = df_original[df_original['city'] == city]['neighbourhood'].dropna().unique().tolist()
    neighbourhood = st.selectbox("Küçə seçin", sorted(filtered_neighs), key="neigh_price")
    room_type = st.selectbox("Otaq növü seçin", room_types, key="room_price")
    min_nights = st.slider("Minimum gecə sayı", 1, 30, value=2)
    availability = st.slider("İldə neçə gün əlçatandır?", 1, 365, value=180)
    host_listings = st.number_input("Ev sahibinin elan sayı", min_value=1, max_value=1000, step=1, value=1)

    input_df = pd.DataFrame({
        'city': [city],
        'neighbourhood': [neighbourhood],
        'room_type': [room_type],
        'minimum_nights': [min_nights],
        'availability_365': [availability],
        'calculated_host_listings_count': [host_listings]
    })

    feature_cols = ['city', 'neighbourhood', 'room_type', 'minimum_nights', 'availability_365', 'calculated_host_listings_count']
    df_price = df_merged.dropna(subset=feature_cols + ['price'])
    X = df_price[feature_cols]
    y = df_price['price']

    model = train_price_model(X, y)

    if st.button("Qiyməti Hesabla"):
        prediction = model.predict(input_df)
        st.success(f"Təxmini qiymət: €{prediction[0]:.2f}")

# --------------------------------
# LICENSE PROQNOZU
# --------------------------------
else:
    st.header("Lisenziya Proqnozu")

    city = st.selectbox("Şəhər seçin", unique_cities, key="city_license")
    filtered_neighs = df_original[df_original['city'] == city]['neighbourhood'].dropna().unique().tolist()
    neighbourhood = st.selectbox("Küçə seçin", sorted(filtered_neighs), key="neigh_license")
    room_type = st.selectbox("Otaq növü seçin", room_types, key="room_license")
    availability = st.slider("İldə neçə gün əlçatandır?", 1, 365, key="avail_license")
    number_of_reviews_ltm = st.number_input("Son 12 ayda olan rəy sayı", min_value=0, max_value=1000, step=1)
    host_listings = st.number_input("Ev sahibinin elan sayı", min_value=1, max_value=1000, step=1)

    input_df = pd.DataFrame({
        'city': [city],
        'neighbourhood': [neighbourhood],
        'room_type': [room_type],
        'availability_365': [availability],
        'number_of_reviews_ltm': [number_of_reviews_ltm],
        'calculated_host_listings_count': [host_listings]
    })

    feature_cols = ['city', 'neighbourhood', 'room_type', 'availability_365', 'number_of_reviews_ltm', 'calculated_host_listings_count']
    df_license = df_merged.dropna(subset=feature_cols + ['license'])
    X = df_license[feature_cols]
    y = df_license['license']

    model = train_license_model(X, y)

    if st.button("Lisenziya Proqnozlaşdır"):
        prediction = model.predict(input_df)
        result = "VAR" if prediction[0] == 1 else "YOXDUR"
        st.success(f"Təxmini lisenziya vəziyyəti: {result}")
