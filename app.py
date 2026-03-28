import streamlit as st
from utils import predict_resume

st.title("Resume Screening AI")

resume_text = st.text_area("Paste your resume here")

if st.button("Analyze Resume"):
    if resume_text:
        result, score, skills = predict_resume(resume_text)

        st.subheader("Result:")
        st.write("Status:", result)
        st.write("Confidence:", str(score) + "%")

        st.subheader("Detected Skills:")
        st.write(skills)
    else:
        st.write("Please enter resume text")
