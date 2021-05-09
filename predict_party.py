import streamlit as st
import pickle
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import matplotlib.pyplot as plt


def party_predict():
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    st.title("Party Spending Predictions")
    st.write("""### Please input some information to predict the Party Wining""")
    df = pd.read_csv('./Resources/party.csv')
    columns = ['year', 'Democrats', 'Republicans', 'winning_party']
    df = df[columns] 
    df[['Democrats', 'Republicans']] = df[['Democrats', 'Republicans']].replace('[\$,]','',regex=True).astype(float)
    # the last column is our label
    y = df.winning_party.values
    #drop last column of data
    X = df.drop(['winning_party', 'year'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1, test_size=0.01) #0.2 means only 20% sample
    #print(X_train.shape)
    model2 = RandomForestRegressor(max_depth=10, random_state=42, n_estimators=500).fit(X_train, y_train)
    y_pred = model2.predict(X_train)
    dem_spend = st.slider("Democrats Spending in Million", 500, 20000, 7200)
    rep_spend = st.slider("Republicans Spending in Million", 500, 20000, 5000)
    dem_spend = dem_spend * 1000000
    rep_spend = rep_spend * 1000000
    a = [dem_spend, rep_spend] 
    test_prediction = model2.predict([a])
    ok = st.button("Predict who will win")
    if ok:
        if dem_spend > rep_spend * 1.3: 
            st.write(f"<h2 style='text-align: center; color: blue;'>*Predicts DEMOCRATS PARTY will win*</h2>", unsafe_allow_html=True)
        elif rep_spend > dem_spend * 1.3: 
           st.write(f"<h2 style='text-align: center; color: red;'>*Predicts REPUBLICANS PARTY will win*</h2>", unsafe_allow_html=True)
        elif test_prediction < [1.5]:  
            #st.write(test_prediction)
            st.write(f"<h2 style='text-align: center; color: blue;'>*Predicts DEMOCRATS PARTY will win*</h2>", unsafe_allow_html=True)
        elif test_prediction > [1.5]:
            st.write(f"<h2 style='text-align: center; color: red;'>*Predicts REPUBLICANS PARTY will win*</h2>", unsafe_allow_html=True)
            #st.write(test_prediction)
        df = X
        df_length = len(df)
        df.loc[df_length] = a
        df = df.iloc[[0, -1]]
        mylist = year = ["last_Election", "Next_Election"]
        df['year'] = mylist
        df = df[ ['year'] + [ col for col in df.columns if col != 'year' ] ]
        df = df[['Democrats', 'Republicans', 'year']] 
        df = df.sort_values(by='year', ascending=True)
        df.plot(
        x = 'year', 
        kind = 'barh', 
        stacked = True, 
        title = 'Democrats vs. Republican Spending',)
        plt.xlabel('Spending')
        st.pyplot()
#show_predict()
