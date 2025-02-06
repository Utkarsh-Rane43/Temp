import streamlit as st
import joblib
import pandas as pd

# Load the trained CatBoost model
model_path = "catboost_model.joblib"  # Ensure this file is in the same directory
model = joblib.load(model_path)

# Define categorical and numerical features
categorical_cols = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
numerical_cols = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']

def predict_price(input_data):
    df = pd.DataFrame([input_data])
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    prediction = model.predict(df)[0]
    return round(prediction, 2)

# Streamlit UI
def main():
    st.title("Used Car Price Prediction")
    st.write("Enter the car details to get an estimated selling price.")
    
    brand = st.selectbox("Brand", ["Maruti", "Hyundai", "Ford", "Kia", "Toyota", "Honda", "Tata", "Other"])
    model_name = st.text_input("Model")
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission_type = st.selectbox("Transmission", ["Manual", "Automatic"])
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, step=1)
    km_driven = st.number_input("Kilometers Driven", min_value=0, step=500)
    mileage = st.number_input("Mileage (km/l)", min_value=0.0, step=0.1)
    engine = st.number_input("Engine Capacity (cc)", min_value=500, step=100)
    max_power = st.number_input("Max Power (bhp)", min_value=30.0, step=5.0)
    seats = st.number_input("Number of Seats", min_value=2, max_value=10, step=1)
    
    if st.button("Predict Price"):
        input_data = {
            "brand": brand, "model": model_name, "seller_type": seller_type,
            "fuel_type": fuel_type, "transmission_type": transmission_type,
            "vehicle_age": vehicle_age, "km_driven": km_driven, "mileage": mileage,
            "engine": engine, "max_power": max_power, "seats": seats
        }
        price = predict_price(input_data)
        st.success(f"Estimated Selling Price: ₹{price}")

if __name__ == "__main__":
    main()
