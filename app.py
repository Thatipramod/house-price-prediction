import streamlit as st
import joblib
import numpy as np
#load model
model=joblib.load("house_price_model.pkl")
st.title("🏠 House Price Prediction App")
st.write("Enter the details below to predict house price:")
size=st.number_input("Number of BHK",min_value=1,max_value=10)
total_sqft=st.number_input("Total square feets",min_value=100)
bath=st.number_input("Number of Bathrooms:",min_value=1,max_value=10)
balcony=st.number_input("Number of Balconiess:",min_value=0,max_value=5)
if st.button("predict price"):
    input_data=np.array([[size,total_sqft,bath,balcony]])
    predict=model.predict(input_data)
    st.success(f"Estimated House Price: ₹ {predict[0]:,.2f}")

