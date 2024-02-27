
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
df=pd.read_csv('diabetes.csv')
st.title("Health Report(based on Diabetes)")
st.sidebar.header("Patient report Data")
st.subheader("Description of data")
st.write(df.describe())
st.subheader("Visualisation of data")
st.bar_chart(df)
def patientreport():
  pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  bp = st.sidebar.slider('Blood Pressure', 0,122, 70 )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  bmi = st.sidebar.slider('BMI', 0,67, 20 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  age = st.sidebar.slider('Age', 21,88, 33 )
  patientreportdata = {
    'Pregnancies':pregnancies,
    'Glucose':glucose,
    'BloodPressure':bp,
    'SkinThickness':skinthickness,
    'Insulin':insulin,
    'BMI':bmi,
    'DiabetesPedigreeFunction':dpf,
    'Age':age
}
  reportdata = pd.DataFrame(patientreportdata, index=[0])
  return reportdata
patientdata = patientreport()
st.subheader('Report Data')
st.write(patientdata)
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
lg=LogisticRegression()
lg.fit(x_train, y_train)
patientresult = lg.predict(patientdata)

st.subheader('Your Report: ')
output=''
if patientresult[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
st.subheader(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, lg.predict(x_test))*100)+'%')
