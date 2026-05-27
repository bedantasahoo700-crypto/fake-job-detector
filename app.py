import streamlit as st
import pandas as pd
import pickle
import os

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

st.set_page_config(page_title="Fake Job Detector", layout="centered")

st.title("🚨 Fake Job Posting Detection AI")
st.write("Enter job details below:")

# Inputs
title = st.text_input("Job Title")
location = st.text_input("Location")
department = st.text_input("Department")
salary_range = st.text_input("Salary Range")
company_profile = st.text_area("Company Profile")
description = st.text_area("Description")
requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits")
employment_type = st.text_input("Employment Type")
required_experience = st.text_input("Experience")
required_education = st.text_input("Education")
industry = st.text_input("Industry")
function = st.text_input("Function")

if st.button("Predict 🚀"):

    input_data = pd.DataFrame([[
        title, location, department, salary_range,
        company_profile, description, requirements,
        benefits, employment_type, required_experience,
        required_education, industry, function
    ]],
    columns=[
        "title","location","department","salary_range",
        "company_profile","description","requirements",
        "benefits","employment_type","required_experience",
        "required_education","industry","function"
    ])

    pred = model.predict(input_data)[0]

    st.markdown("---")

    if pred == 1:
        st.error("❌ FAKE JOB DETECTED")
    else:
        st.success("✅ REAL JOB POSTING")