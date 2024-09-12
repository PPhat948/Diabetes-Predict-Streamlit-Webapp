import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Title and description
st.title('Machine Learning App: Diabetes Prediction')
st.info('This web app predicts the risk of diabetes based on input features.')

# Load dataset function
@st.cache_data
def load_data():
    url = 'diabetes_prediction_dataset.csv'
    df = pd.read_csv(url)
    return df

# Clean and preprocess data
def clean_data(df):
    df = df.replace({'not current': 'former', 'ever': 'former'})  # Cleaning text values
    df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)  # One-hot encoding
    return df

# Get input from user
def get_user_input():
    gender = st.sidebar.selectbox('Gender', {'Male', 'Female'})
    age = st.sidebar.slider('Age', 0, 100, 30)
    hypertension = st.sidebar.selectbox('Hypertension (1 = Yes, 0 = No)', {'1', '0'})
    heart_disease = st.sidebar.selectbox('Heart Disease (1 = Yes, 0 = No)', {'1', '0'})
    smoking_history = st.sidebar.selectbox('Smoking History', {'Never', 'Current', 'Former'})
    bmi = st.sidebar.slider('BMI', 0.0, 50.0, 25.0)
    HbA1c_level = st.sidebar.slider('HbA1c Level', 0.0, 10.0, 5.0)
    blood_glucose_level = st.sidebar.slider('Blood Glucose Level', 50.0, 400.0, 100.0)
    
    user_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level
    }
    
    return pd.DataFrame(user_data, index=[0])

# Split the data for training and testing
def split_data(df_cleaned, original_df):
    X = df_cleaned.iloc[1:, :]
    y = original_df['diabetes']
    input_data = df_cleaned.iloc[:1, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
    return X_train, X_test, y_train, y_test, input_data

# Train and predict based on model selection
def train_and_predict(X_train, y_train, X_test, input_data, model_choice):
    if model_choice == 'Logistic Regression':
        model = LogisticRegression()
    elif model_choice == 'KNN':
        model = KNeighborsClassifier(n_neighbors=7)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=10)

    model.fit(X_train, y_train)
    prob_input = model.predict_proba(input_data)[:, 1]  # Get the probability for class 1 (diabetes)
    score = model.score(X_train, y_train)
    return prob_input, score

# Display prediction result based on custom thresholds
def display_results(prob_input):
    prob = prob_input[0]
    
    if prob < 0.4:
        st.success(f'You have Low risk of diabetes (Probability: {prob:.2f})')
    elif 0.4 <= prob < 0.7:
        st.warning(f'You have a risk of diabetes. Please consult a doctor. (Probability: {prob:.2f})')
    else:
        st.error(f'You have high risk of diabetes!! Please consult a doctor. (Probability: {prob:.2f})')

# Main app logic
def main():
    # Load and display the dataset
    with st.expander('Dataset'):
        st.subheader('**Raw Data**')
        df = load_data()
        st.write(df)

    # User input features
    input_df = get_user_input()
    
    # Preprocessing and cleaning
    df_cleaned = clean_data(pd.concat([input_df, df.drop('diabetes', axis=1)], axis=0))
    
    # Display user input features
    with st.expander('Input Features'):
        st.write('**Your input**')
        st.write(input_df)
    
    # Model selection and training
    with st.expander('Model'):
        st.subheader('Select Model for Prediction')
        model_choice = st.selectbox('Choose Model', ['Logistic Regression', 'KNN', 'Random Forest'])
        
        if st.button('Predict', type='primary'):
            # Split data for training and testing
            X_train, X_test, y_train, y_test, input_data = split_data(df_cleaned, df)
            
            # Train the model and get prediction probabilities
            with st.spinner('Wait for it...'):
                prob_input, model_score = train_and_predict(X_train, y_train, X_test, input_data, model_choice)
                # Display prediction result based on probability thresholds
                display_results(prob_input)
                # Display model score
                st.write('Model Score:', model_score)

if __name__ == '__main__':
    main()
