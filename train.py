import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
#load datasets
data=pd.read_csv("hyderabad_house_data.csv")
#data cleaning
data['total_sqft'] = pd.to_numeric(data['total_sqft'], errors='coerce')
data=data.dropna()
#convert size 2-BHK into to 2
data['size']=data['size'].str.replace(" BHK","").astype(int)
data=data.dropna()
#x,y
x=data[['size','total_sqft','bath','balcony']]
y=data['price']
#split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#train
model=LinearRegression()
model.fit(x_train,y_train)
print('model trained successfully')
#predit
y_pred=model.predict(x_test)
#accuracy 
print("r2:",r2_score(y_test,y_pred))
#save
joblib.dump(model,"house_price_model.pkl")
print("Model saved")
#print(model.predict(np.array([[2,1100,2,1]])))