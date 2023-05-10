!pip install streamlit 
!pip install streamlit-option-menu
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np 
import pandas as pd 
from PIL import Image
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("smoke_detection_iot.csv") 
df.drop(columns='CNT',inplace = True)
x = df.iloc[:,2:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


model = LogisticRegression(max_iter = 100000)
model.fit(X_train,y_train)
y_predicted = model.predict(X_test)


# web title
st.set_page_config(
    page_title="Aplikasi Pendeteksi Asap",
)

# navigation/option
with st.sidebar:
   selected = option_menu(
        menu_title="Main Menu",  
        options=["Beranda", "Prediksi"], 
        icons=["house", "record-circle"],  
        menu_icon="cast",  # optional
        default_index=0,  # optional         
)

# option : Home
if selected == "Beranda":
    st.write("# Aplikasi Pendeteksi Asap")
   

    image1 = Image.open('download.jpeg')
    
    st.image(image1)

    st.caption("Dibuat Oleh **Patwan Saputra**")

# option : Demo 
if selected == "Prediksi":
    st.title("Aplikasi Pendeteksi Asap")
    st.write("Isi data di bawah ini :")

    
    temperature = st.number_input("Temperature[c]")
    humidity = st.number_input("Humidity[%]")
    tvoc = st.number_input("TVOC[ppb]")
    eco2 = st.number_input("eCO2[ppm]")
    rawh2 = st.number_input("Raw H2")
    rawethanol = st.number_input("Raw Ethanol")
    pressure = st.number_input("Pressure[hPa]")
    pm10 = st.number_input("PM1.0")
    pm25 = st.number_input("PM2.5")
    nc05 = st.number_input("NC0.5")
    nc10 = st.number_input("NC1.0")
    nc25 = st.number_input("NC2.5")
    
    

    ok = st.button ("Prediksi")

    if ok:
      x_new = [[temperature, humidity, tvoc, eco2, rawh2,
       rawethanol, pressure, pm10, pm25, nc05, nc10,
       nc25]]
      result = model.predict(x_new)
      if result == 0:
        st.subheader("Tidak Ada Asap")
      if result == 1:
        st.subheader("Ada Asap")
