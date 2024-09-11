import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
st.title('Machine Learning App: Diabetes Predict')
st.info('Hello')

with st.expander('Dataset'):
  st.write('**Raw Data**')
  st.write('This Dataset from : https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset')
  df = pd.read_csv('diabetes_prediction_dataset.csv')
  df


#with st.expander('EDA'):
    

with st.sidebar:
  st.header('Input Features')
  gender = st.selectbox('Gender',{'Male','Female'})
  age = st.slider('Age',0,100,30)
  hypertension = st.selectbox('Hypertension',{'Yes','No'})
  heart_disease = st.selectbox('Heart Disease',{'Yes','No'})
  smoking_history = st.selectbox('Smoking History',{'Never','Current','Former'})
  bmi = st.slider('BMI',0.0,50.0,25.0)
  HbA1c_level = st.slider('HbA1c Level',0.0,10.0,5.0)
  blood_glucose_level = st.slider('Blood Glucose Level',50.0,400.0,100.0)

  new_data = {'gender':gender,
              'age':age,
              'hypertension':hypertension,
              'heart_disease':heart_disease,
              'smoking_history':smoking_history,
              'bmi':bmi,
              'HbA1c_level':HbA1c_level,
              'blood_glucose_level':blood_glucose_level}
  
  input_df = pd.DataFrame(new_data,index=[0])
  #merge_df = pd.concat([input_df,X_raw],axis=0)
  
          
              
with st.expander('Input Features'):
  st.write('**Your input**')
  input_df

#Data Preprocessing
#Clean Category Column
gender = pd.get_dummies(df['gender'],drop_first=True)
smoking_history = pd.get_dummies(df['smoking_history'],drop_first=True)
df = pd.concat([df,gender,smoking_history],axis=1)
df.drop(['gender','smoking_history'],axis=1,inplace=True)

#Split Data
X = df.drop('diabetes',axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#scaler
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#Logistic Regression
model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

with st.expander('Model'):
  input_model = st.selectbox('Select Model for Predict',{'Logistic Regression','KNN','Random Forest'})
  st.write('Your model use ',input_model)
  if(input_model == 'Logistic Regression'):
     model = LogisticRegression()
  elif(input_model == 'KNN'):
    model = KNeighborsClassifier(n_neighbors=7)
  else:
    model = RandomForestClassifier(n_estimators=200,max_depth=10) 
  model.fit(X_train,y_train)
  predictions = model.predict(X_test)
  score = model.score(X_train,y_train)
  st.write('Model Score:', score)
  st.write('Confusion Matrix')
  confs = confusion_matrix(y_test,predictions)
  st.write(confs)
  predictions_input = model.predict(input_df)
  st.write(predictions_input)
  
 
  
