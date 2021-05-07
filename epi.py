import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import ticker
import seaborn as sns



@st.cache #avoid to load csv again
def load_data():
    df = pd.read_csv("epi.csv")
    return df

df = load_data()

def Absentee():
    fig, ax = plt.subplots()
    table_count = df.groupby(df['state_abbv'])['abs_rej_all_ballots'].sum()
    table_count = table_count.sort_values(ascending=False)[:10]
    payee_index = table_count.index
    payee_val = table_count.values
    ax = sns.barplot(x = payee_val,y=payee_index,orient='h')
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    #plt.annotate("EPI", xy=(0.5, 0.5), fontsize=15, xycoords='axes fraction', bbox=props)
    fmt = '%.2f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    plt.ylabel('State')
    plt.xlabel('Absentee ballot Rejection Rate')
    ax.set_title("Top Ten Absentee Ballot Rejection by State (2008 -2018)")
    st.pyplot(fig)

def Provisional():
    fig, ax = plt.subplots()
    table_count = df.groupby(df['state_abbv'])['prov_rej_all'].sum()
    table_count = table_count.sort_values(ascending=False)[:10]
    payee_index = table_count.index
    payee_val = table_count.values
    ax = sns.barplot(x = payee_val,y=payee_index,orient='h')
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    #plt.annotate("EPI", xy=(0.5, 0.5), fontsize=15, xycoords='axes fraction', bbox=props)
    fmt = '%.2f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    plt.ylabel('State')
    plt.xlabel('Provisional ballot Rejection Rate')
    ax.set_title("Top Ten Provisional Ballot Rejection by State (2008 -2018)")
    st.pyplot(fig)

def UOCAVA():
    fig, ax = plt.subplots()
    table_count = df.groupby(df['state_abbv'])['uocava_rej'].sum()
    table_count = table_count.sort_values(ascending=False)[:10]
    payee_index = table_count.index
    payee_val = table_count.values
    ax = sns.barplot(x = payee_val,y=payee_index,orient='h')
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    #plt.annotate("EPI", xy=(0.5, 0.5), fontsize=15, xycoords='axes fraction', bbox=props)
    fmt = '%.2f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    plt.ylabel('State')
    plt.xlabel('UOCAVA ballot Rejection Rate')
    ax.set_title("Top Ten UOCAVA Ballot Rejection by State (2008 -2018)")
    st.pyplot(fig)

def Turnout():
    fig, ax = plt.subplots()
    table_count = df.groupby(df['state_abbv'])['vep_turnout'].sum()
    table_count = table_count.sort_values(ascending=False)[:10]
    payee_index = table_count.index
    payee_val = table_count.values
    ax = sns.barplot(x = payee_val,y=payee_index,orient='h')
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    #plt.annotate("EPI", xy=(0.5, 0.5), fontsize=15, xycoords='axes fraction', bbox=props)
    fmt = '%.2f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    plt.ylabel('State')
    plt.xlabel('VEP Turnout Rate')
    ax.set_title("Top Ten VEP Turnout By State (2008 - 2018)")
    st.pyplot(fig)

def predict_Turnout():
    #fig, ax = plt.subplots()
    df1 = df.drop(['state_abbv', 'state_fips', 'website_reg_status', 'website_provisional_status', 'online_reg'],axis=1)
    df1 = df1.fillna(0)
    # the last column is our label
    y_train = df1.iloc[:,-1:]
    #drop last column of data
    X_train = df1.iloc[:, :-1]
    #drop first colum of data
    X_test = df1.iloc[:,1:]
    # lets have a look on the shape 
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(max_depth=5, random_state=1, n_estimators=1000).fit(X_train, y_train)
    y_pred = model.predict(X_train)
    prediction = pd.DataFrame({'state':df.state_abbv,'Actual':df1.vep_turnout, 'Prediction': y_pred})
    #st.write(prediction)
    table_count = prediction.groupby(prediction['state'])['Actual', 'Prediction'].sum() 
    table_count = table_count.sort_values(by='Prediction', ascending=True)[:10]
    word = [table_count]
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax = table_count.plot.barh()
    fmt = '%.2f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    plt.ylabel('State')
    plt.xlabel('VEP Turnout Rate')
    ax.set_title("Predict Top Ten VEP Turnout By State")
    st.pyplot()





