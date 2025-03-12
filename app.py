import streamlit as st
import pandas as pd
import pickle as pk
import time
import base64
import os

# 🔹 Load the model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# 🔹 Function to Convert Image to Base64 (with error handling)
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        print(f"⚠️ Error: File not found at {image_path}")
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 🔹 Set Image Path (Make sure the image exists!)
image_path = "C:\CU\cu_ml_loan_prediction\img1.jpeg"  # Use raw string (r"") or forward slashes ("/")
image_base64 = get_base64_image(image_path)

# 🔹 Set Page Configuration
st.set_page_config(page_title="Loan Prediction App", page_icon="🔍", layout="centered")

# 🔹 Apply Custom Styling with Background Image (if image is available)
if image_base64:
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

            * {{
                font-family: 'Poppins', sans-serif;
            }}
            
            /* Background Image */
            body {{
                background-image: url("data:image/jpeg;base64,{image_base64}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            /* Form Container */
           .stApp {{
                background: rgba(255, 255, 255, 0.95);
                padding: 30px;
                border-radius: 20px;
                max-width: 750px;
                margin: auto;
                box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.3);
                border: 3px solid rgba(44, 62, 80, 0.6); /* Added border */
}}

        /* Result Box */
        .result-box {{
            padding: 20px;
            border-radius: 20px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            margin-top: 20px;
            animation: fadeIn 1s ease-in-out;
            border: 2px solid rgba(0, 0, 0, 0.2); /* Added subtle border */
}}

            /* Header */
            h1 {{
                color: #2c3e50;
                text-align: center;
                font-size: 34px;
                font-weight: 700;
            }}

            /* Subtitle with Search Icon */
            .subtext {{
                color: #34495e;
                text-align: center;
                font-size: 18px;
                font-weight: 400;
                margin-bottom: 15px;
            }}
            .subtext::before {{
                content: '🔍 ';  /* Search Icon */
                font-size: 22px;
                font-weight: bold;
            }}

            /* Button Styling */
            div.stButton > button {{
                width: 100%;
                background: linear-gradient(90deg, #27ae60, #2ecc71);
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 10px;
                padding: 14px;
                transition: all 0.3s ease-in-out;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
            }}
            div.stButton > button:hover {{
                background: linear-gradient(90deg, #2ecc71, #27ae60);
                transform: scale(1.08);
                box-shadow: 0px 6px 15px rgba(0,0,0,0.4);
            }}

            .approved {{
                background: #2ecc71;
                color: white;
                box-shadow: 0px 4px 10px rgba(0, 255, 0, 0.4);
            }}
            .rejected {{
                background: #e74c3c;
                color: white;
                box-shadow: 0px 4px 10px rgba(255, 0, 0, 0.4);
            }}
        </style>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Background image not found. Please check the file path.")

# 🔹 Page Header with Search Icon
st.markdown("<h1>💰 Loan Prediction System</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtext">Fill out the form to know whether your loan will be approved or not</p>', unsafe_allow_html=True)

# 🔹 User Inputs
col1, col2 = st.columns(2)

with col1:
    no_of_dep = st.slider('👨‍👩‍👦 No. of Dependents', 0, 5)
    Annual_Income = st.text_input('💰 Annual Income', placeholder="Enter your annual income")
    Loan_Amount = st.text_input('🏦 Loan Amount', placeholder="Enter the loan amount")

with col2:
    grad = st.selectbox('🎓 Education', ['Graduated', 'Not Graduated'])
    self_emp = st.selectbox('💼 Self Employed?', ['Yes', 'No'])
    Loan_Dur = st.slider('📅 Loan Duration (Years)', 0, 30)

Cibil = st.text_input('📊 CIBIL Score', placeholder="Enter your CIBIL score")
Assets = st.text_input('📈 Assets', placeholder="Enter your total assets")

# 🔹 Convert Inputs
grad_s = 0 if grad == 'Graduated' else 1
emp_s = 0 if self_emp == 'No' else 1

# 🔹 Predict Button
if st.button("📊 Predict Loan Eligibility"):
    try:
        # Convert text inputs to numbers
        Annual_Income = float(Annual_Income) if Annual_Income else 0
        Loan_Amount = float(Loan_Amount) if Loan_Amount else 0
        Cibil = int(Cibil) if Cibil else 0
        Assets = float(Assets) if Assets else 0
        
        # Prepare data for prediction
        pred_data = pd.DataFrame([[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
                                 columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets'])
        pred_data = scaler.transform(pred_data)
        
        # 🔹 Show a loading animation
        with st.spinner("⏳ Analyzing your loan eligibility... Please wait..."):
            time.sleep(2)  # Simulate processing time
            predict = model.predict(pred_data)
        
        # 🔹 Display Result
        if predict[0] == 1:
            st.markdown('<div class="result-box approved">✅ Congratulations! Your Loan is Approved! 🎉</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box rejected">❌ Sorry, Your Loan is Rejected! 💔</div>', unsafe_allow_html=True)

    except ValueError:
        st.error("⚠️ Please enter valid numerical values for Income, Loan Amount, Assets, and CIBIL Score.")
