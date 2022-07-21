#Description: Detect if someone has diabetes using machine learning and python!

#Import Packages
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Title and Sub-Title
st.write("""
# Diabetes Detection
Detect if someone has diabetes using machine learning and python!
""")

#Open and Display Image
image = Image.open('diabetes_detection_web_app.png')

st.image(image, caption='ML', use_column_width=True)

#Get the Data
data = pd.read_csv('diabetes.csv')

#Set a Subheader
st.subheader('Data Information:')

#Show the Data as a Table
st.dataframe(data)

#Show Statistics on the Data
st.write(data.describe())

#Show the Data as a Chart
chart = st.bar_chart(data)

#Split the Data Into Independent 'X' and Dependent 'Y' Variables
X = data.iloc[:, 0:8].values
Y = data.iloc[:, - 1].values

#Split the Data Set Into 75% Training and 25% Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#Get the Feature Input from the User
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 	72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    bmi = st.sidebar.slider('bmi', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('dpf', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)
    
    #Store a Dictionary Into a Variable
    user_data = {'pregnancies': pregnancies,'glucose': glucose,'blood_pressure': blood_pressure,'skin_thickness': skin_thickness,'insulin': insulin,'bmi': bmi,'dpf': dpf,'age': age}

    #Transform the Data Into a Data Frame
    features = pd.DataFrame(user_data, index = [0])
    return features

#Store the User Input Into a Variable
user_input = get_user_input()

#Set a Subheader and Display the Users Input
st.subheader('User Input:')
st.write(user_input)

#Create and Train the Model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Show the Model's Metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))*100)+'%')

#Store the Model's Predictions in a Variable
prediction = RandomForestClassifier.predict(user_input)

#Set a Subheader and Display the Classification
st.subheader('Classification: ')
st.write(prediction)
