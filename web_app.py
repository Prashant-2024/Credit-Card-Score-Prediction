import streamlit as st
import pickle
import pandas as pd
import sklearn

teams=['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

model=['Logistic Regression', 'RainForestClassification', 'XGBoostClassification']

pipe_lr=pickle.load(open('pipe_lr.pkl', 'rb'))
pipe_rfc=pickle.load(open('pipe_rfc.pkl', 'rb'))
pipe_xgb=pickle.load(open('pipe_xgb.pkl', 'rb'))

st.title("IPL Winning Predictor")

col1, col2=st.columns(2)
with col1:
    batting_team=st.selectbox("Select the Batting Team", teams)
with col2:
    bowling_team=st.selectbox("Select the Bowling Team", teams)

selected_city=st.selectbox('Select host city', sorted(cities))

target=st.number_input('Target')

Model=st.selectbox('Model', sorted(model))

col3, col4, col5=st.columns(3)
with col3:
    score=st.number_input('Score')
with col4:
    overs=st.number_input('Overs Completed')
with col5:
    wickets=st.number_input('Wickets out')

if st.button('Predict Probabilty'):
    runs_left=target-score
    balls_left=120-(overs*6)
    wickets=10-wickets
    crr=score/overs 
    rrr=(runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    st.table(input_df)
    if (Model==model[0]):
        result=pipe_lr.predict_proba(input_df)
    elif (Model==model[1]):
        result=pipe_rfc.predict_proba(input_df)
    elif (Model==model[2]):
        result=pipe_xgb.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.text(batting_team+"- "+str(round(win*100))+"%")
    st.text(bowling_team+"- "+str(round(loss*100))+"%")