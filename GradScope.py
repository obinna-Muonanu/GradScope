import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from PIL import Image
import xgboost

#center the contents of the page
st.set_page_config(page_title="GradScope", layout="centered")

#load the model
model = load("GradScope.joblib") 

# open the logo image
logo = Image.open("GradScope Logo.png")

#creates three columns with different widths with col2 taking the most space
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    #create two inner columns with different widths with inner_col2 taking the most space ( 8 times bigger)
    inner_col1, inner_col2 = st.columns([1, 8]) 
    with inner_col1:
         #st.write() was used here to add some space above the logo so it aligns horizontally with the app title
         st.write("") 
         st.write("")
         st.image(logo, width=50)
    with inner_col2:
        st.markdown("# GradScope")

#center the title and subtitle
st.markdown("<h5 style='text-align: center;'>GradScope is a predictive model that accurately predicts your chances of getting admitted based on your academic profile </h5>", unsafe_allow_html=True)

#Later in the code, the user will ask why the app predicted that the user will not get admitted.
if "show_reason" not in st.session_state:
    st.session_state["show_reason"] = False

student_name = st.text_input("Enter your name:")
if student_name:
    st.write(f"Hello {student_name}!, welcome to GradScope!")

#create 2 new columns for feature inputs
col1, col2 = st.columns([1,1])
with col1:
    #input for GRE score
    gre_score = st.number_input("Enter your GRE score:", min_value=260, max_value=340, value=280, step=1)
    #input for TOEFL score
    toefl_score = st.number_input("Enter your TOEFL score:", min_value=0, max_value=120, value=90, step=1)
    #input for university rating
    university_rating = st.number_input("Enter your university rating (1-5):", min_value=1.00, max_value=5.00, value=3.00, step=0.01)
    #input for SOP score
    sop_score = st.number_input("Enter your SOP score (out of 5):", min_value=0.00, max_value=5.00, value=3.00, step=0.01)

with col2:
    #input for LOR score
    lor_score = st.number_input("Enter your LOR score (out of 5):", min_value=0.00, max_value=5.00, value=3.00, step=0.01)
    #input for CGPA score
    cgpa_score = st.number_input("Enter your CGPA (out of 10):", min_value=0.00, max_value=10.00, value=8.00, step=0.01)
    #input for research score yes or no
    Research = st.selectbox("Do you have any research experience?", options=["Yes", "No"])

# Feature engineering
academic_strength = 0.8 * gre_score + 0.2 * toefl_score
Gre_university = gre_score * university_rating
Research_strength = (1 if Research == "Yes" else 0) + (sop_score + lor_score + cgpa_score + gre_score)

#predict button
# Ensure 'prediction_made' and 'admission_probability' states exist
if "prediction_made" not in st.session_state:
    st.session_state["prediction_made"] = False #will be set to True when the user clicks the predict button
    st.session_state["admission_probability"] = 0.0 #will be set to the predicted probability when the user clicks the predict button
    st.session_state["student_name"] = "" #will be set to the name of the user when the user clicks the predict button
    st.session_state["show_reason"] = False #will be set to True when the user clicks the button to see the reasons why the model predicted that the user will not get admitted


st.warning("**Note:** This is just a prediction and should not be taken as a guarantee of admission.")
    
#predict button
if st.button("Predict"):
    with st.spinner("Predicting..."):
        # Simulate a delay of 3s for the prediction process
        import time
        time.sleep(3)
        # provide the input features and engineered features as required by the model
        input_features = np.array([[gre_score, toefl_score, university_rating,
                                sop_score, lor_score, cgpa_score,
                                1 if Research == "Yes" else 0,
                                academic_strength, Gre_university, Research_strength]])
    
    st.session_state["admission_probability"] = model.predict_proba(input_features)[0][1] #replaces False because the button has been clicked
    st.session_state["prediction_made"] = True #replaces False because the button has been clicked
    st.session_state["student_name"] = student_name #replaces "" because the button has been clicked
    st.session_state["show_reason"] = False #still False because the user hasn't requested to know why the model predicted not admitted

# Show prediction results after Predict is clicked
if st.session_state["prediction_made"]:
    admission_probability = st.session_state["admission_probability"]
    student_name = st.session_state["student_name"]
    
    if admission_probability > 0.5:
        st.markdown(f"**Congrats {student_name}!** You have **{admission_probability * 100:.2f}%** chance of getting admitted.")
    else:
        st.warning(f"**{student_name}!** It seems your chances are low as your profile isn't strong enough to get admitted. You have **{admission_probability * 100:.2f}%** chance of getting admitted.")
        
        # Ask if the user wants to know why
        if st.button("Do you wish to know why?"):
            st.session_state["show_reason"] = True #Now the user has clicked the button to see the reasons why the model predicted that the user will not get admitted

if st.session_state["show_reason"]:
    average_values = {
        "gre": 320,
        "toefl": 109,
        "sop": 3.69,
        "lor": 3.73,
        "cgpa": 8.84,
        "research": 1  
    }
    if gre_score < average_values['gre']:
        st.write(f"Your GRE score ({gre_score}) is below average. Consider improving it to at least {average_values['gre']}.")
        st.success("**Tip:** Consider investing time in a structured GRE preparation course or utilizing practice tests to improve your score, especially in areas where you're weakest.")
    if toefl_score < average_values['toefl']:
        st.write(f"Your TOEFL score ({toefl_score}) is below average. Consider improving it to at least {average_values['toefl']}.")
        st.success("**Tip:** Take a TOEFL preparation course or practice with official TOEFL resources to improve your English proficiency, focusing on areas like reading and speaking.")
    if sop_score < average_values['sop']:
        st.write(f"Your SOP score ({sop_score}) is below average. Consider improving it to at least {average_values['sop']}.")
        st.success("**Tip:** Strengthen your SOP by clearly articulating your academic and career goals, showing your motivation, and aligning your interests with the program's strengths .Also seek feedback from professors or mentors to improve your SOP.")
    if lor_score < average_values['lor']:
        st.write(f"Your LOR score ({lor_score}) is below average. Consider improving it to at least {average_values['lor']}.")
        st.success("**Tip:** Build strong relationships with professors or mentors who can provide detailed, personalized recommendations that highlight your academic abilities and character.")
    if cgpa_score < average_values['cgpa']:
        st.write(f"Unfortunately, you can't improve your cgpa as you've already graduated")
        st.success("**Tip:** Take relevant online courses or certifications such as from Coursera, edX, or other platforms to show your commitment to learning and improving.")
    if Research != average_values['research']:
        st.write(f"You stand a better chance if you have some research experience.")
        st.success("**Tip:** Seek out research assistantships or internships related to your field of study, or initiate small research projects to demonstrate your commitment and interest in research.")

    st.markdown("These are some ways you can improve your chances. Consider improving your profile by working on the areas highlighted above. Keep working smart and hard!")

        