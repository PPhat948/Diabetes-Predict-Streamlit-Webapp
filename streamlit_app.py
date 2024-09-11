import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
st.title('Machine Learning App: Diabetes Predict')
st.info('Hello')

with st.expander('Dataset'):
  st.write('**Raw Data**')
  st.write('This Dataset from : https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset')
  df = pd.read_csv('diabetes_prediction_dataset.csv')
  df

  st.write('**X**')
  X_raw = df.drop('diabetes',axis=1)
  X_raw

  st.write('**y**')
  y_raw = df['diabetes']
  y_raw

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
#Split Data
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.30, random_state=101)

#OneHot Encoder
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[object_cols]))

OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

num_X_train = X_train.drop(object_cols, axis=1)
num_X_test = X_test.drop(object_cols, axis=1)

X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

#scaler
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#Logistic Regression
model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

with st.expander('Model Score'):
  st.write('Your model use Logistic Regression')
  score = model.score(X_train,y_train)
  st.write('Model Score:', score)
  confs = confusion_matrix(y_test,predictions)
  st.write(confs)
