import streamlit as st
import pickle
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
#from tabulate import tabulate

def model(state, year, candidatevotes, totalvotes):
    df = pd.read_csv('./Resources/senate.csv')
    df = df.dropna()
    df= df[df['Results'] != 0]
    df.rename(columns = {'party_simplified' : 'winning_party'},inplace = True)  
    df = df.loc[df.state_po == state]   
    #print(red("Typo! Please try again."))
    columns = ['state_po', '_year', 'candidatevotes', 'totalvotes','winning_party']
    df = df[columns]   
    #print(f"Here's {state} DataFrame {df}"'\n')
    #print(df)
     #SECOND PREDICTION
    df.winning_party = df. winning_party.replace({'DEMOCRAT': 1, 'REPUBLICAN': 2, 'LIBERTARIAN': 3, 'OTHER': 4})
    # the last column is our label
    y = df.winning_party.values
    #drop last column of data
    X = df.drop(['winning_party', 'state_po'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1, test_size=0.01) #0.2 means only 20% sample
    #print(X_train.shape)
    model2 = RandomForestRegressor(max_depth=5, random_state=1, n_estimators=1000).fit(X_train, y_train)
    y_pred = model2.predict(X_train)
    #print(len(y_pred))
    #print(blue(f"Predict_Score of {state}: {model2.score(X_train, y_train)}"))
    #print(y_pred)
    a = [year, candidatevotes, totalvotes] 
    test_prediction = model2.predict([a])
    return test_prediction, df

def show_predict():

    st.title("Senate Election Predictions")


    st.write("""### Please input some information to predict the Senate Election""")

    state = (
        "AZ",
        "CA",
        "FL",
        "GA",
        "MI",
        "MN",
        "NV",
        "NH",
        "NC",
        "PA",
        "SC",
        "TX",
    )
   
    state = st.selectbox("State", state)
    year = st.slider("Election Year", 1976, 2024, 2020)
    candidatevotes = st.slider("Candidate Votes", 500000, 10000000, 2000000)
    totalvotes = st.slider("Total Votes", 500000, 20000000, 5000000)
    result = model(state, year, candidatevotes, totalvotes)
    test_prediction = result[0]
    df = result[1]
        
    ok = st.button("Predict who will win")
    if ok:
        if test_prediction < [1.5]:
            #st.subheader(f"Predicts SENATE DEMOCRAT will win in {state}")
            st.write(f"<h2 style='text-align: center; color: blue;'>*Predicts DEMOCRATS PARTY will win in {state}*</h2>", unsafe_allow_html=True)
        elif test_prediction > [1.5]:
            #st.subheader(f"Predicts SENATE PUBLICAN will win in {state}")
            st.write(f"<h2 style='text-align: center; color: red;'>*Predicts REPUBLICANS PARTY will win in {state}*</h2>", unsafe_allow_html=True)
        else:
            st.subheader(f"Predicts Otherwill win in {state}")
        st.write(df)
        st.write("""#### winning_party: '1 is DEMOCRAT', '2 is REPUBLICAN', and "3 is OTHER'""")


            
