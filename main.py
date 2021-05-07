import streamlit as st
from predict import *
#from fec import *
from epi import *
from Senate import *
#from swing_states import *
#from Swingstates import *
from Georgia import *
from NN import *
#from selenium import webdriver
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
import webbrowser
import base64
import os


st.set_option('deprecation.showPyplotGlobalUse', False)

page = st.sidebar.selectbox("Please select the options:", ("Predict Senate", "FEC", "EPI", "Senate", "Swing States", "Georgia", "Neural Network Model"))

if page == "Predict Senate":
    show_predict()
    
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
    from streamlit import caching
    import base64
    caching.clear_cache()
    bl = "https://i.pinimg.com/originals/94/30/e8/9430e85bcee211cda9dfff8400e2fe19.jpg"
    black = "https://image.freepik.com/free-photo/old-black-background-grunge-texture-dark-wallpaper-blackboard-chalkboard-room-wall_1258-28313.jpg"
    blue = "https://img.wallpapersafari.com/desktop/1920/1080/86/77/GSFHEt.jpg"
   
    st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url({bl})
    }}
    </style>
    """,
    unsafe_allow_html=True)

    st.write("""### Apply Deep Learning Neural Network Model To Analyze Senate Dataset!""")
    #st.write("<h3 style='text-align: center; color: white;'>Apply Neural Network Model to analyze Total Votes Of Senate Dataset!-\nline.</h3>", unsafe_allow_html=True)
    st.write("""#### Note: y = df['totalvotes'].values and The training model can be viewed in Terminal.\n""")
    widget1 = st.empty()
    widget1.write("<h3 style='text-align: center; color: red;'>Testing In Progress...</h3>", unsafe_allow_html=True)
    #st.write("""### Testing In Progress...""")
    widget2 = st.empty()
    widget2.markdown("![Testing In Progress](https://i.pinimg.com/originals/aa/8d/93/aa8d93e1023e43a1c1c2d04951854f53.gif)")
    #image2()
    result = nn_2layers()
    if result[1]:
        widget1.empty()
        widget2.empty()
        st.write("""### Training Model is done!""")
    

