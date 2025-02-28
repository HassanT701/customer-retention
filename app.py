import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
import os

# Load dataset
df = pd.read_csv('churn.csv')
customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

# Load models
models = {
    "XGBoost Feature Engineered": pickle.load(open('models/xgboost-featureEngineered.pkl', "rb")),
    "XGBoost SMOTE": pickle.load(open('models/xgboost-SMOTE.pkl', "rb")),
    #"Voting Classifier": pickle.load(open('models/voting_clf.pkl', "rb"))
}

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HF_API_KEY = os.getenv("HF_API_KEY")  

# State management
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# UI Setup
st.title("Customer Retention Prediction")
selected_customer = st.selectbox("Select a customer", customers)

if selected_customer:
    selected_customer_id = int(selected_customer.split(" - ")[0])
    customer_data = df[df['CustomerId'] == selected_customer_id].iloc[0]

    # Input Form
    st.subheader("Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=int(customer_data['CreditScore']))
        geography = st.selectbox("Geography", ["Spain", "France", "Germany"], index=["Spain", "France", "Germany"].index(customer_data['Geography']))
        gender = st.radio("Gender", ["Male", "Female"], index=0 if customer_data['Gender'] == "Male" else 1)
        age = st.number_input("Age", min_value=18, max_value=100, value=int(customer_data['Age']))
        tenure = st.number_input("Tenure", min_value=0, max_value=10, value=int(customer_data['Tenure']))

    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=float(customer_data['Balance']))
        num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=int(customer_data['NumOfProducts']))
        has_cr_card = st.checkbox("Has Credit Card", value=bool(customer_data['HasCrCard']))
        is_active_member = st.checkbox("Is Active Member", value=bool(customer_data['IsActiveMember']))
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=float(customer_data['EstimatedSalary']))

    # Prepare input data
    input_data = [
        credit_score,
        1 if geography == "France" else 0,
        1 if geography == "Germany" else 0,
        1 if gender == "Male" else 0,
        age,
        tenure,
        balance,
        num_of_products,
        int(has_cr_card),
        int(is_active_member),
        estimated_salary,
    ]
    input_data += [0] * (18 - len(input_data))

    # Prediction Section
    if st.button("Predict"):
        st.session_state.predictions = {
            model_name: model.predict_proba([input_data])[0][1] * 100
            for model_name, model in models.items()
        }
        st.session_state.prediction_made = True
        st.session_state.explanation = None

    # Display predictions if available
    if st.session_state.predictions:
        voting_prediction = st.session_state.predictions["Voting Classifier"]
        
        # Main prediction display
        st.subheader("Prediction Results")
        st.write(f"Overall Prediction (Voting Classifier): {voting_prediction:.2f}%")
        
        # Visualizations
        progress_color = "red" if voting_prediction > 50 else "green"
        st.markdown(f"""
            <div style="background-color: {progress_color}; padding: 10px; border-radius: 5px;">
                <h3 style="color: white;">{voting_prediction:.2f}%</h3>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            model_names = list(st.session_state.predictions.keys())
            model_values = list(st.session_state.predictions.values())
            ax.bar(model_names, model_values, color=["red" if val > 50 else "green" for val in model_values])
            ax.set_ylim(0, 100)
            st.pyplot(fig)

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=voting_prediction,
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': progress_color}}
            ))
            st.plotly_chart(fig)

    # Explanation Section
    st.subheader("Model Explanation")
    
    if st.session_state.prediction_made and st.button("Generate Explanation"):
        try:
            prompt = f"""
            As a data scientist, provide a technical explanation of why the Voting Classifier predicted {'churn' if st.session_state.predictions['Voting Classifier'] > 50 else 'retention'}.
            Focus on these specific factors:
            
            - Credit Score: {credit_score} ({'low' if credit_score < 650 else 'average' if credit_score < 750 else 'high'})
            - Account Balance: ${balance:,.2f}
            - Tenure: {tenure} years
            - Active Member: {'Yes' if is_active_member else 'No'}
            
            Include both:
            1. Technical aspects of the Voting Classifier (ensemble of XGBoost, Random Forest, SVC)
            2. Feature weight analysis based on the input values
            
            Avoid assumptions about bank programs or unverified details.
            """
            
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 350,
                    "temperature": 0.6,
                    "do_sample": True,
                    "top_k": 40,
                    "top_p": 0.85,
                    "repetition_penalty": 1.6
                }
            }

            response = requests.post(HF_API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                explanation = response.json()[0]['generated_text']
                
                # Clean response
                explanation = explanation.replace("rewards program", "").strip()
                explanation = explanation.replace("BB&T", "").strip()
                explanation = explanation.split("As a data scientist")[0].strip()
                
                explanation += "\n(Additional factors may include product usage patterns and regional competition)"

                # Add technical note about Voting Classifier
                explanation += "\n\nTechnical Note: The Voting Classifier combines predictions from XGBoost, Random Forest, and Support Vector Machine models using weighted averaging."
                
                st.write(f"### Explanation")
                st.write(explanation)
                
            elif "CUDA out of memory" in response.text:
                st.error("System busy - please try generating the explanation again")
                st.button("Retry Explanation")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Explanation Error: {str(e)}")
