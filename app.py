import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle 
import base64 

st.title('Heart Disease Predictor')
tab1,tab2,tab3=st.tabs(['Predict','Bulk Predict','Model Information'])

def get_binary_file_downloader_html(df):
    csv=df.to_csv(index=False)
    b64=base64.b64encode(csv.encode()).decode()
    href=f'<a href="data:file/csv;base64,{b64}"download="predictions.csv">Download Predictions CSV</a>'
    return href

with tab1:
    age=st.number_input("Age(years)",min_value=0,max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    
    # Convert categorical inputs to numeric
    sex = 1 if sex == "Male" else 0
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })
    
    
    # Algorithm and model file names
    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    modelnames = ['DT.pkl', 'LR.pkl', 'RF.pkl', 'SVM.pkl']

    # Function to predict with all models
    def predict_heart_disease(data):
        predictions = []
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction[0])  # take first value
        return predictions

    # Create a submit button to make predictions
    if st.button("Submit"):
        st.subheader("Results....")
        st.markdown("--------------------------------")

        result = predict_heart_disease(input_data)

        # Display results model-wise
        for i in range(len(result)):
            st.subheader(algonames[i])
            if result[i] == 0:
                st.write("🟢 No heart disease detected.")
            else:
                st.write("🔴 Heart disease detected.")
            st.markdown("--------------------------------")
            
# Tab 2: Upload CSV File
with tab2:
    st.title("Upload CSV File")

    st.subheader("Instructions to note before uploading the file:")
    st.info(
        "1. No NaN values allowed.\n"
        "2. Total 11 features in this order: ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').\n"
        "3. Check the spellings of the feature names.\n"
        "4. Feature values conventions:\n\n"
        "   - Age: age of the patient [years]\n"
        "   - Sex: sex of the patient [0: Male, 1: Female]\n"
        "   - ChestPainType: chest pain type [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic]\n"
        "   - RestingBP: resting blood pressure [mm Hg]\n"
        "   - Cholesterol: serum cholesterol [mm/dl]\n"
        "   - FastingBS: fasting blood sugar [1: FastingBS > 120 mg/dl, 0: otherwise]\n"
        "   - RestingECG: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality, 2: showing probable or definite LVH]\n"
        "   - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]\n"
        "   - ExerciseAngina: exercise-induced angina [1: Yes, 0: No]\n"
        "   - Oldpeak: ST [Numeric value measured in depression]\n"
        "   - ST_Slope: the slope of the peak exercise ST segment [0: upsloping, 1: flat, 2: downsloping]\n"
    )
    
   
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV into a DataFrame
        input_data = pd.read_csv(uploaded_file)

        # Load a trained model (example: Logistic Regression)
        model = pickle.load(open("LogisticRegression.pkl", "rb"))

        expected_columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        
        if set(expected_columns).issubset(input_data.columns):
                              
            # Add a new column for predictions
            input_data['Prediction LR'] = ""

            # Loop through each row and make predictions
            for i in range(len(input_data)):
                arr = input_data.iloc[i, :].values.reshape(1, -1)
                input_data['Prediction LR'].iloc[i] = model.predict(arr)[0]

            # Save results into a new CSV
            input_data.to_csv("PredictedHeartLR.csv", index=False)

            # Display predictions on the Streamlit app
            st.subheader("Predictions:")
            st.write(input_data)
            
            st.markdown(get_binary_file_downloader_html(input_data),unsafe_allow_html=True)
        else:
            st.warning("Please make sure the uploaded CSV file has the correct columns.")    
    else:
        st.info("Upload a CSV file to get predictions.")
        
with tab3:
    import plotly.express as px
    data = {
        "Decision Trees": 80.97,
        "Logistic Regression": 85.86,
        "Random Forest": 88.58,
        "Support Vector Machine": 84.22,
    }

    models = list(data.keys())
    accuracies = list(data.values())

    df = pd.DataFrame(list(zip(models, accuracies)), columns=["Models", "Accuracies"])

    fig = px.bar(df, x="Models", y="Accuracies", title="Model Accuracies", text="Accuracies")
    st.plotly_chart(fig)