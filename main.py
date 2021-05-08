import streamlit as st
from predict import *
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

page = st.sidebar.selectbox("Please select the options:", ("Predict Senate", "EPI", "Senate", "Swing States", "Georgia", "Neural Network Model"))

if page == "Predict Senate":
    show_predict()
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
  
    st.write("""### Apply Deep Learning Neural Network Model To Analyze EPI Dataset!""")
    #st.write("<h3 style='text-align: center; color: white;'>Apply Neural Network Model to analyze Total Votes Of Senate Dataset!-\nline.</h3>", unsafe_allow_html=True)
    st.write("""#### Prediction (y = vep_turnout): Target variable that will be predicted by the input.\n\n""")
    widget1 = st.empty()
    widget1.write("<h3 style='text-align: center; color: red;'>Testing In Progress...</h3>", unsafe_allow_html=True)
    widget2 = st.empty()
    #widget2.markdown("![Testing In Progress](https://hieppham.s3.us-east-2.amazonaws.com/Final_project/test.gif)")
    try:
        result = epi_nn()
        if result:
            widget1.empty()
            widget2 = st.empty()
            st.write("""### Training Model is done!""")
    except:
        pass
    

