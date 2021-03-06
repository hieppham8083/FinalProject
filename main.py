import streamlit as st
from predict import *
from predict_party import *
#from fec import *
from epi import *
from Senate import *
#from swing_states import *
#from Swingstates import *
from Georgia import *
#from NN import *
from NeuralNetwork import *
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
import webbrowser
import base64
import os
st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_option('deprecation.showPyplotGlobalUse', False)

page = st.sidebar.selectbox("Please select the options:", ("Predict Senate Wining", "Predicts Party Spending", "EPI", "Senate", "Swing States", "Georgia", "Neural Network Model"))

if page == "Predict Senate Wining":
    show_predict()
elif page == "Predicts Party Spending":
    party_predict()
elif page == "FEC":
    st.title("Federal Election Commission")
    Payee()
    Report()
    Category()
    
elif page == "EPI":
    st.title("Elections Performance Index")
    Absentee()
    Provisional()
    UOCAVA()
    Turnout()
    predict_Turnout()
   
elif page == "Senate":
    st.title("Senate Election Results")
    House_vs__Senate()
    dem_vs_rep()
    party_history()
    image()
    
elif page == "Swing States":
    st.title("Swing States Election Results")
    #slide()
    #swing_states()
    #Race_population()
    #Contributions()
    # Margin()
    exec(open("Swingstates.py").read())
   
elif page == "Georgia":
    st.title("Georgia election results")
    map()
    DEM_vs_REP()
    Metro()
    Race()
    Facts()
    #map2()
   
elif page == "Neural Network Model":
  
    st.write("""### Vote Turnout Prediction Using Deep Learning Neural Network Model!""")
    #st.write("<h3 style='text-align: center; color: white;'>Apply Neural Network Model to analyze Total Votes Of Senate Dataset!-\nline.</h3>", unsafe_allow_html=True)
    st.write("""#### Prediction (y = vep_turnout): Target variable that will be predicted by the input.\n\n""")
    widget1 = st.empty()
    widget1.write("<h3 style='text-align: center; color: red;'>Testing In Progress...</h3>", unsafe_allow_html=True)
    widget2 = st.empty()
    widget2.markdown("")
    result = epi_nn()
    if result < 0.88: 
        widget1.write("<h3 style='text-align: center; color: red;'>r2_score is low! Please increase hidden layer to improve Performance.</h3>", unsafe_allow_html=True)
        widget2.write(f"<h3 style='text-align: center; color: red;'>r2_score = {result}</h3>", unsafe_allow_html=True)
    else:
        widget1.write("<h3 style='text-align: center; color: blue;'>Training Model is done!</h3>", unsafe_allow_html=True)
        widget2.write(f"<h3 style='text-align: center; color: blue;'>r2_score = {result}</h3>", unsafe_allow_html=True)
    st.write("""### Training Model is done!""")
   
    

