#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.colors as mcolors
import pandas as pd 
import plotly as py
import math
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import cycle, islice
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
from simple_colors import *
import colorama
from colorama import Fore
plt.style.use('fivethirtyeight')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import glob
import os.path
from pathlib import Path
import fnmatch
from simple_colors import * # pip install simple-colors
import warnings #fixed any warning in terminal
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import matplotlib.ticker as mtick
from matplotlib import ticker


# In[ ]:


def race_5():
    df = pd.read_csv(csv_file)
    df[['Population']] = df[['Population']] .replace('[\,,]','',regex=True).astype(int)
    years = [2019]
    df = df.loc[df.Year.isin(years)]
    table_count = df.groupby(df['State'])['Population'].sum()
    table_count = table_count.sort_values(ascending=False)
    y = table_count.index
    x = table_count.values
    ax = sns.barplot(x = x,y=y,orient='h')
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    #plt.annotate("EPI", xy=(0.5, 0.5), fontsize=15, xycoords='axes fraction', bbox=props)
    plt.ylabel('State')
    plt.xlabel('Population (million)')
    ax.set_title("2019 Swing States Population")
    plt.show()


# In[ ]:


def race_4():
    while True:
        df = pd.read_csv(csv_file)
        df[['Population']] = df[['Population']] .replace('[\,,]','',regex=True).astype(int)
        try:
            years = [int(x) for x in input("Enter a list of year[2010 - 2019] to test: ").split()]
            df1 = df.loc[df.Year.isin([years[0]])]
            table_0 = df1.groupby(df['Gender'])['Population'].sum()
            table_0 = table_0.sort_values(ascending=True)
            try:
                df2 = df.loc[df.Year.isin([years[1]])]
                table_1 = df2.groupby(df['Gender'])['Population'].sum()
                table_1 = table_1.sort_values(ascending=True)
            except:
                pass
            try:
                df3 = df.loc[df.Year.isin([years[2]])]
                table_2 = df3.groupby(df['Gender'])['Population'].sum()
                table_2 = table_2.sort_values(ascending=True)
            except:
                pass
        except:
            print(red("Typo! Please try again."))
            break
        if len(years) == 1:
            df4 = pd.DataFrame({years[0]: table_0.values.tolist()}, index=table_0.index.tolist())
        elif len(years) == 2:  
            df4 = pd.DataFrame({years[1]: table_1.values.tolist(), years[0]: table_0.values.tolist()}, index=table_1.index.tolist()) 
        elif len(years) == 3:
            df4 = pd.DataFrame({years[2]: table_2.values.tolist(), years[1]: table_1.values.tolist(), years[0]: table_0.values.tolist()}, index=table_2.index.tolist()) 
        df4.plot.barh()
        plt.xlabel('Population (million)')
        plt.ylabel('Race')
        plt.title('Gender By Population')
        plt.show()


# In[ ]:


def race_3():
    while True:
        df = pd.read_csv(csv_file)
        df[['Population']] = df[['Population']] .replace('[\,,]','',regex=True).astype(int)
        try:
            years = [int(x) for x in input("Enter a list of year[2010 - 2019] to test: ").split()]
            df = df.loc[df.Year.isin(years)]
            df1 = df.loc[df.Year.isin([years[0]])]
            df1 = df.loc[df.Year.isin([years[0]])]
            table_0 = df1.groupby(df['Race'])['Population'].sum()
            table_0 = table_0.sort_values(ascending=True)
            try:
                df2 = df.loc[df.Year.isin([years[1]])]
                table_1 = df2.groupby(df['Race'])['Population'].sum()
                table_1 = table_1.sort_values(ascending=True)
            except:
                pass
            try:
                df3 = df.loc[df.Year.isin([years[2]])]
                table_2 = df3.groupby(df['Race'])['Population'].sum()
                table_2 = table_2.sort_values(ascending=True)
            except:
                pass
        except:
            print(red("Typo! Please try again."))
            break
        if len(years) == 1:
            df4 = pd.DataFrame({years[0]: table_0.values.tolist()}, index=table_0.index.tolist())
        elif len(years) == 2:  
            df4 = pd.DataFrame({years[1]: table_1.values.tolist(), years[0]: table_0.values.tolist()}, index=table_1.index.tolist()) 
        elif len(years) == 3:
            df4 = pd.DataFrame({years[2]: table_2.values.tolist(), years[1]: table_1.values.tolist(), years[0]: table_0.values.tolist()}, index=table_2.index.tolist()) 
        df4.plot.barh()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        plt.annotate(f"Total population of all 7 Swing States", xy=(0.3, 0.5), fontsize=11, xycoords='axes fraction', bbox=props)
        plt.xlabel('Population (million)')
        plt.ylabel('Race')
        plt.title('Race By Population')
        plt.show()


# In[ ]:


def race_2():
    while True:
        df = pd.read_csv(csv_file)
        df[['Population']] = df[['Population']] .replace('[\,,]','',regex=True).astype(int)
        try:
            years = [int(x) for x in input("Enter a list of year[2010 - 2019] to test: ").split()]
            df = df.loc[df.Year.isin(years)]
            df1 = df.loc[df.Year.isin([years[0]])]
            table_0 = df1.groupby(df['State'])['Population'].sum()
            table_0 = table_0.sort_values(ascending=True)
            try:
                df2 = df.loc[df.Year.isin([years[1]])]
                table_1 = df2.groupby(df['State'])['Population'].sum()
                table_1 = table_1.sort_values(ascending=True)
            except:
                pass
            try:
                df3 = df.loc[df.Year.isin([years[2]])]
                table_2 = df3.groupby(df['State'])['Population'].sum()
                table_2 = table_2.sort_values(ascending=True)
            except:
                pass
        except:
            print(red("Typo! Please try again."))
            break
        if len(years) == 1:
            df4 = pd.DataFrame({years[0]: table_0.values.tolist()}, index=table_0.index.tolist())
        elif len(years) == 2:  
            df4 = pd.DataFrame({years[1]: table_1.values.tolist(), years[0]: table_0.values.tolist()}, index=table_1.index.tolist()) 
        elif len(years) == 3:
            df4 = pd.DataFrame({years[2]: table_2.values.tolist(), years[1]: table_1.values.tolist(), years[0]: table_2.values.tolist()}, index=table_2.index.tolist()) 
        df4.plot.barh()
        plt.xlabel('Population (million)')
        plt.ylabel('Swing State')
        plt.title('Swing States By Population')
        plt.show()


# In[ ]:


def georgia_6():
    df = pd.read_csv(csv_file)
    columns = ['fact', 'Arizona', 'Georgia', 'South Carolina']
    df1 = df[columns]
    df1 = df1.dropna()
    df1[['Arizona', 'Georgia', 'South Carolina']] = df1[['Arizona', 'Georgia', 'South Carolina']].replace('[\%,]','',regex=True).astype(float)
    df1 = pd.DataFrame({'Georgia': df1['Georgia'].tolist(), 'Arizona': df1['Arizona'].tolist(), 'South Carolina': df1['South Carolina'].tolist()}, index=df1['fact'].tolist())
    df1.plot.barh()
    plt.xlabel('Percentage(%)')
    plt.ylabel('Facts')
    plt.title('Quick Facts Comparison')
    plt.show()


# In[ ]:


def georgia_5():
    df = pd.read_csv(csv_file)
    columns = ['Race', '2016', '2020', '% increase']
    df1 = df[columns]
    df1 = df1.dropna()
    df2 = pd.DataFrame({'2020': df1['2020'].tolist(), '2016': df1['2016'].tolist()}, index=df1['Race'].tolist())
    df2.plot.barh()
    plt.xlabel('Total Votes (million)')
    plt.ylabel('Race')
    plt.title('Georgia Votes By Race')
    df1 = df1.drop(['2016', '2020'], axis=1)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.annotate(f"{df1.to_string(index=False)}", xy=(0.6, 0.2), fontsize=11, xycoords='axes fraction', bbox=props)
    plt.show()


# In[ ]:


def georgia_4():
    df = pd.read_csv(csv_file)
    columns = ['year', 'Metro_DEM_Votes', 'Metro_REP_Votes']
    df1 = df[columns]
    df1 = df1.dropna()
    df1[['year']] = df1[['year']].fillna(0.0).astype(int)
    df1 = df1.sort_values('year')
    df1 = pd.DataFrame({'Metro_DEM_Votes': df1['Metro_DEM_Votes'].tolist(),'Metro_REP_Votes': df1['Metro_REP_Votes'].tolist()}, index=df1['year'].tolist())
    df1.plot.barh()
    plt.xlabel('Total Votes (million)')
    plt.ylabel('Year')
    plt.title('Atlanta Metro Votes')
    plt.show()


# In[ ]:


def georgia_3():
    df = pd.read_csv(csv_file)
    columns = ['year', 'DEM_Rate', 'REP_Rate']
    df1 = df[columns]
    df1 = df1.dropna()
    df1[['year']] = df1[['year']].fillna(0.0).astype(int)
    df1[['DEM_Rate', 'REP_Rate']] = df1[['DEM_Rate', 'REP_Rate']].replace('[\%,]','',regex=True).astype(float)
    df1 = df1.sort_values('year')
    df1 = pd.DataFrame({'DEM_Rate': df1['DEM_Rate'].tolist(), 'REP_Rate': df1['REP_Rate'].tolist()}, index=df1['year'].tolist()) 
    df1.plot.barh()
    plt.xlabel('Rate(%)')
    plt.ylabel('Year')
    plt.title('Democrats vs. Republican Rate in Georgia')
    plt.show()


# In[ ]:


def georgia_2():
    from urllib.request import urlopen
    import plotly as py
    import json
    import webbrowser
    df = pd.read_csv(csv_file)
    #df['fips'] = df['fips'].apply(lambda x: '0'+x if len(x) == 4 else x)
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
    #geojson = px.data.election_geojson()
    #df = counties_fips_color[(counties_fips_color["state_code"] == "GA") | (counties_fips_color["state_code"] == "FL")]
    states = ['GA', 'SC', 'FL']
    df = df[df["state_code"].isin(states)]
    #df = counties_fips_color
    fig = px.choropleth(df, geojson=counties, locations='fips', color='color',
                                scope="usa",
                            
                            hover_data=["state","county", "candidate", "total_votes"])
    fig.update_geos(
                #lonaxis_range=[20, 380],
                projection_scale=2.7,
                center=dict(lat=31, lon=-83),
                visible=True)                      
    fig.update_layout(title= {"text": "Georgia vs South Carolina & Florida'\n' 2020 swing states total_votes", "xanchor": "center", "x": 0.5, "y": 0.95}, 
        margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    fig.show()
    fig.write_html("myplot.html")
    url = 'file://file:///Users/hiep_pham/Desktop/Analysis_Projects/Final_Project/myplot.html'
    webbrowser.open(url, new=2)  # open in new tab


# In[ ]:


def swing_state_3():
    df = pd.read_csv(csv_file)
    states = ["Arizona", "Florida", "Georgia", "Michigan", "Minnesota", "Nevada", "New Hampshire", "North Carolina", "Pennsylvania", "Texas"]
    df = df.loc[df.state.isin(states)]
    width = 0.25
    labels = ['AZ', 'FL', 'GA', 'MI', 'MN', 'NV', 'NH', 'NC', 'PA', 'TX']
    x = df.state
    y1 = df['2020']
    y2 = df['2016']
    y3 = df['2012']
    c1 = df['2020_results']
    c2 = df['2016_results']
    c3 = df['2012_results']
    display = [1, 2, 3]
    plt.bar(np.arange(len(x))- width, y1, color = c1, width=width) 
    plt.bar(np.arange(len(x)), y2, color = c2, width=width)
    plt.bar(np.arange(len(x)) + width, y3, color = c3, width=width)
    plt.ylabel('Margin Rate (%)')
    plt.xlabel('Swing State')
    plt.title('Swing States Margin (2020, 2016 & 2012)')
    plt.xticks(np.arange(len(x)),labels)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.annotate("1st Bar: 2020", xy=(0.85, 0.97), fontsize=6, xycoords='axes fraction', bbox=props)
    plt.annotate("2nd Bar: 2016", xy=(0.85, 0.92), fontsize=6, xycoords='axes fraction', bbox=props)
    plt.annotate("3rd Bar: 2012", xy=(0.85, 0.87), fontsize=6, xycoords='axes fraction', bbox=props)
    plt.annotate("2020", xy=(-0.45, 0.45), fontsize=7)
    plt.annotate("2016", xy=(-0.3, 3.6), fontsize=7)
    plt.annotate("2012", xy=(0.1, 9.15), fontsize=7)
    plt.show()


# In[ ]:


def swing_state_2():
    df = pd.read_csv(csv_file)
    states = ["Arizona", "Florida", "Georgia", "Michigan", "Minnesota", "Nevada", "New Hampshire", "North Carolina", "Pennsylvania", "Texas"]
    df = df.loc[df.state.isin(states)]
    df[['2020_Total_Contributions', '2018_Total_Contributions','2016_Total_Contributions']] = df[['2020_Total_Contributions', '2018_Total_Contributions','2016_Total_Contributions']].replace('[\$,]','',regex=True).astype(float)
    df1 = pd.DataFrame({'2020': df['2020_Total_Contributions'].tolist(), '2018': df['2018_Total_Contributions'].tolist(), '2016': df['2016_Total_Contributions'].tolist()}, index=df['state'].tolist()) 
    df1.plot.barh()
    plt.xlabel('Total Contributions (million)')
    plt.ylabel('Swing State')
    plt.title('Swing States Contributions')
    plt.show()


# In[ ]:


def epi6():
    df = pd.read_csv(csv_file)
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
    print(prediction)
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
    plt.show()


# In[ ]:


def epi5():
    df = pd.read_csv(csv_file)
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
    plt.show()


# In[ ]:


def epi4():
    df = pd.read_csv(csv_file)
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
    plt.show()


# In[ ]:


def epi3():
    df = pd.read_csv(csv_file)
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
    plt.show()


# In[ ]:


def epi2():
    df = pd.read_csv(csv_file)
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
    plt.show()


# In[2]:


def nn2_2layers():
    df = pd.read_csv(csv_file)
    columns = ['report_year', 'expenditure_amount', 'category_code_full', 'support_oppose_indicator', 'candidate_name', 'cand_office_state', 'cand_office_district', 'election_type']
    df = df[columns]
    df = df.dropna()

    report_year_counts = df.report_year.value_counts()
    replace_report_year = report_year_counts[report_year_counts < 10000].index
    for app in replace_report_year:
        df.report_year =  df.report_year.replace(app,"Other")

    election_type_count = df.election_type.value_counts()
    replace_election_type = election_type_count[election_type_count < 5000].index
    for app in replace_election_type:
        df.election_type  =  df.election_type.replace(app,"Other")

    def change_string(value):
        if value == "S":
            return 1
        else:
            return 0
    df["support_oppose_indicator"] = df["support_oppose_indicator"].apply(change_string)

    expenditures_cat = df.dtypes[df.dtypes == "object"].index.tolist()
    le = LabelEncoder()
    for col in expenditures_cat:
        df = df.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
    
    # Create a OneHotEncoder instance
    enc = OneHotEncoder(sparse=False)
    encode_df = pd.DataFrame(enc.fit_transform(df[expenditures_cat]))
    encode_df.columns = enc.get_feature_names(expenditures_cat)

    df = df.merge(encode_df,left_index=True, right_index=True)
    df = df.drop(columns = expenditures_cat)

    y = df.support_oppose_indicator.values
    X = df.drop("support_oppose_indicator", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    #Create a StandardScaler instances
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    # Scale the data
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    number_input_features = len(X_train_scaled[0])
    nodes_hidden_layer1 = 80
    nodes_hidden_layer2 = 40

    #Initial Network
    nn = tf.keras.models.Sequential()
    nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer1, activation="relu", input_dim=number_input_features))
    nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer2, activation="relu"))
    nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    print(nn.summary())
    nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    fit_model = nn.fit(X_train_scaled, y_train, epochs=100)
    # Evaluate the model using the test data
    model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test,verbose=2)
    print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
    # Export our model to HDF5 file
    nn.save("independent_expenditures.h5")


# In[3]:


def r2_2layers():
    df = pd.read_csv(csv_file)
    columns = ['report_year', 'image_number', 'file_number', 'payee_name', 'expenditure_date', 'dissemination_date', 'expenditure_amount', 'category_code_full', 'support_oppose_indicator', 'candidate_id', 'candidate_name', 'cand_office_state', 'cand_office_district', 
    'election_type', 'sub_id']
    df = df[columns]
    df = df.dropna()
    df =  df.fillna(0)
    #Create category_columns and numeric_columns variables
    numeric_columns = []
    category_columns = []
    for col in df.columns:
        if is_string_dtype(df[col]) == True:
            category_columns.append(col)
        elif is_numeric_dtype(df[col]) == True:
            numeric_columns.append(col)
    #Create dummy variables for the category_columns and merge on the numeric_columns to create an X dataset
    category_columns = pd.get_dummies(df[category_columns])
    X = df[numeric_columns].merge(category_columns, left_index= True, right_index= True)
    #Create an y dataset
    y = df['expenditure_amount'].values
    # Split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # Scale X_train and X_test
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Create a neural network model with keras
    nn = tf.keras.models.Sequential()
    # Add a hidden layer with twice as many neurons as there are inputs. Use 'relu'
    n_input = len(X_train_scaled[0])

    n_hidden = n_input * 2
    n_hidden_layer2 = n_input * 2 #2nd hidden layer

    nn.add(tf.keras.layers.Dense(units=n_hidden, input_dim=n_input, activation='relu'))
    nn.add(tf.keras.layers.Dense(units=n_hidden_layer2, activation='relu')) #2nd hidden layer

    # add an output layer with a 'linear' activation function.
    nn.add(tf.keras.layers.Dense(units=1,  activation='linear'))
    # print a summary of the model
    print(nn.summary())
    # compile the model using the "adam" optimizer and "mean_squared_error" loss function
    nn.compile(loss='mean_squared_error' , optimizer='adam' , metrics=['mse'])
    # train the model for 100 epochs
    model = nn.fit(X_train_scaled, y_train, epochs=100)
    # predict values for the train and test sets
    y_train_pred = nn.predict(X_train_scaled)
    y_test_pred = nn.predict(X_test_scaled)
    # score the training predictions with r2_score()
    print(f"r2_score of y_train: {r2_score(y_train, y_train_pred)}")
    # score the test predictions with r2_score()
    print(f"r2_score of y_test: {r2_score(y_test, y_test_pred)}")


# In[4]:


def expenditures_6():
    df = pd.read_csv(csv_file)
    #Trump
    df1 = df[df['candidate_name'].notna()]
    support =['S','SUP']
    word1 = ["TRUMP", "Trump"]
    for i in range(len(word1)):
        trump_entry = df1[df1.candidate_name.str.contains(word1[i])]
    trump_entry = trump_entry[trump_entry.support_oppose_indicator.isin(support)]
    #Biden
    df2 = df[df['candidate_name'].notna()]
    support =['S','SUP']
    word2 = ["BIDEN", "Biden"]
    for i in range(len(word2)):
        biden_entry = df2[df2.candidate_name.str.contains(word2[i])]
    biden_entry = biden_entry[biden_entry.support_oppose_indicator.isin(support)]

    total_trump = len(trump_entry)
    total_biden = len(biden_entry)
    total = len(df.cand_office_state)
    rate_trump = total_trump/total * 100
    rate_biden = total_biden/total * 100
    support_rate_list = [rate_biden, rate_trump]
    candidate = ['BIDEN, JOSEPH R JR', 'TRUMP, DONALD J']
    support_prob = pd.DataFrame({'Candidate':candidate,'Support Rate':support_rate_list})
    sns.barplot(data=support_prob,x='Candidate',y='Support Rate')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.annotate(f"Total Biden Support: {total_biden}", xy=(0.5, 0.7), fontsize=12, xycoords='axes fraction', bbox=props)
    plt.annotate(f"Total Trump Support: {total_trump}", xy=(0.5, 0.6), fontsize=12, xycoords='axes fraction', bbox=props)
    plt.ylabel('Support Rate')
    plt.xlabel('Candidate')
    plt.show()


# In[5]:


def expenditures_5():
    df = pd.read_csv(csv_file)
    df = df[df['candidate_name'].notna()]
    df = df[df.cand_office_state != "US"]
    support =['S','SUP']
    words = ["BIDEN", "Biden"]
    #biden_entry = df[df.candidate_name == 'Biden, Joseph']
    for i in range(len(words)):
        biden_entry = df[df.candidate_name.str.contains(words[i])]
    biden_entry = biden_entry[biden_entry.support_oppose_indicator.isin(support)]
    #biden_state = biden_entry[(biden_entry[['cand_office_state']] != 0).all(axis=1)]
    #biden_state = biden_entry.loc[biden_entry['cand_office_state']!=0].dropna()
    biden_state = biden_entry.dropna(subset=['cand_office_state'])
    biden_state = biden_state[(biden_state[['cand_office_state']] != '0').all(axis=1)]
    biden_state = biden_state.groupby(biden_state.cand_office_state)                [['cand_office_state','support_oppose_indicator']].size()
    biden_index = biden_state.index
    biden_val = biden_state.values
    fig,ax = plt.subplots(figsize=(8,6))
    sns.barplot(x = biden_val,y=biden_index,orient='h',ax=ax)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.annotate("Number of contributor Supporting Biden by state", xy=(0.3, 0.7), fontsize=12, xycoords='axes fraction', bbox=props)
    plt.ylabel('State')
    plt.xlabel('Count')
    plt.show()


# In[6]:


def expenditures_4():
    df = pd.read_csv(csv_file)
    df = df[df['candidate_name'].notna()]
    support =['S','SUP']
    words = ["TRUMP", "Trump"]
    for i in range(len(words)):
        trump_entry = df[df.candidate_name.str.contains(words[i])]
    trump_entry = trump_entry[trump_entry.support_oppose_indicator.isin(support)]
    trump_state = trump_entry.groupby(trump_entry.cand_office_state)                [['cand_office_state','support_oppose_indicator']].size()
    trump_index = trump_state.index
    trump_val = trump_state.values
    fig,ax = plt.subplots(figsize=(8,6))
    sns.barplot(x = trump_val,y=trump_index,orient='h',ax=ax)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.annotate("Number of contributor Supporting Trump by state", xy=(0.2, 0.5), fontsize=12, xycoords='axes fraction', bbox=props)
    plt.ylabel('State')
    plt.xlabel('Count')
    plt.show()


# In[7]:


def expenditures_3():
    df = pd.read_csv(csv_file)
    table_count = df.groupby(df.category_code_full)['expenditure_amount'].sum()
    table_count = table_count.sort_values(ascending=False)[:10]
    category_code_idx = table_count.index
    category_code_val = table_count.values
    fig,ax = plt.subplots(figsize=(8,6))
    sns.barplot(x = category_code_val,y=category_code_idx,orient='h')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.annotate("Top Ten Category vs Expenditure Amount", xy=(0.3, 0.5), fontsize=15, xycoords='axes fraction', bbox=props)
    plt.ylabel('Category')
    plt.xlabel('Expenditure Amount')
    plt.show()


# In[8]:


def expenditures_2():
    df = pd.read_csv(csv_file)
    table_count = df.groupby(df['payee_name'])['expenditure_amount'].sum()
    table_count = table_count.sort_values(ascending=False)[:10]
    payee_index = table_count.index
    payee_val = table_count.values
    sns.barplot(x = payee_val,y=payee_index,orient='h')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.annotate("Top Ten Payee", xy=(0.5, 0.5), fontsize=15, xycoords='axes fraction', bbox=props)
    plt.ylabel('Payee Name')
    plt.xlabel('Expenditure Amount')
    plt.show()


# In[9]:


def senate_predict():
    while True:
        df = pd.read_csv(csv_file)
        df = df.dropna()
        df= df[df['Results'] != 0]
        df.rename(columns = {'party_simplified' : 'winning_party'},inplace = True)
        try:
            state = input("Please enter the STATE to predict: ").strip().upper()   
            df = df.loc[df.state_po == state]   
            #print(red("Typo! Please try again."))
            columns = ['state_po', '_year', 'candidatevotes', 'totalvotes','winning_party']
            df = df[columns]    
            #print(f"Here's {state} DataFrame {df}"'\n')
            print(df)

            #FIRST PREDICTION
            ##the last column is our label
            y_train = df.totalvotes.values
            #drop last column of data
            X_train = df.drop(['winning_party', 'state_po'], axis=1)
            #drop first colum of data
            X_test = df.drop(['winning_party', 'state_po', '_year'], axis=1)
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(max_depth=5, random_state=1, n_estimators=1000).fit(X_train, y_train)
            y_pred = model.predict(X_train)
            new_pred = y_pred[-1]
            prediction = pd.DataFrame({'State':df.state_po,'Last_Election':df.totalvotes, 'Next_Election_Pred': y_pred.astype(int)}, index=None)
            prediction = prediction.iloc[[-1]]
            print (prediction.to_string(index=False))

            y = df.totalvotes.values
            #drop last column of data
            X = df.drop(['winning_party', 'state_po'], axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1, test_size=0.01) #0.2 means only 20% sample
            X_train.shape, X_test.shape, y_train.shape, y_test.shape
            model = RandomForestRegressor(max_depth=5, random_state=1, n_estimators=1000).fit(X_train, y_train)
            y_pred = model.predict(X_train).astype(int)
            #print(blue(f"Predict_Score of {state}: {model.score(X_train, y_train)}"))
            total_votes_pred = y_pred[-1]
            total_votes_pred

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
            print(blue(f"Predict_Score of {state}: {model2.score(X_train, y_train)}"))
            #print(y_pred)
        except ValueError:
            print(red("Typo! Please try again."'\n'))
            break

        lst1 = [x for x in y_train[-20:] if x == 1] 
        dem = len(lst1)
        lst2 = [x for x in y_train[-20:] if x == 2] 
        rep = len(lst2)
        lst3 = [x for x in y_train[-20:] if (x ==3)] 
        lib = len(lst3)
        lst4 = [x for x in y_train[-20:] if (x != 1 and x !=2 and x !=3)] 
        other = len(lst4)
        if other > dem and other > rep:
            print_pred = f"Predict OTHER will win in {state}"
            print(green(f"Predict OTHER will win in {state} next Election."))
        elif dem > rep and dem >= other:      
            print_pred = f"Predict SENATE DEMOCRAT will win in {state}"
            print(blue(f"Predict SENATE DEMOCRAT will win in {state} next Election."))
        elif rep >= dem and rep >= other:
            print_pred = f"Predict SENATE REPUBLICAN will win in {state}"
            print(red(f"Predict SENATE REPUBLICAN will win in {state} next Election."))
        #print(dem, rep, lib, other)

        #First Plot
        word = [""]
        old_votes = df['totalvotes'].iloc[-1]
        df1 = pd.DataFrame({"Next_Election_Pred_Votes": new_pred, "Last_Elecction_Votes": old_votes}, index=word) 
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        df1.plot.barh()
        plt.annotate(f"{prediction.to_string(index=False)}", xy=(0.25, 0.05), fontsize=11, xycoords='axes fraction', bbox=props)
        plt.annotate(print_pred, xy=(0.25, 0.18), fontsize=11, xycoords='axes fraction', bbox=props)
        plt.xlabel('Total Votes')
        plt.title(f"Next Senate Elections Predictions in {state}")
        plt.show()

        #Second Plot
        try:
            a = [int(x) for x in input("Enter a list[year, candidatevotes, totalvotes] to test the prediction: ").split()]
        except:
            print(red("Typo! Please try again."'\n'))
            break
        if a == []:
            print(red("Typo! Please try again."'\n'))
            break
        old_year = df['_year'].iloc[-1]
        new_year = a[0]
        new_votes = a[2]
        test_prediction = model2.predict([a])
        word = [""]
        old_votes = df['totalvotes'].iloc[-1]
        df = pd.DataFrame({f"{new_year}_PredictTotalVotes": new_votes, f"{old_year}_TotalVotes": old_votes}, index=word) 
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        df.plot.barh()
        if test_prediction < [1.5]:
            plt.annotate(f"Predict SENATE DEMOCRAT will win in {state}", xy=(0.25, 0.1), fontsize=11, xycoords='axes fraction', bbox=props)
        elif test_prediction > [1.5]:
            plt.annotate(f"Predict SENATE PUBLICAN will win in {state}", xy=(0.2, 0.1), fontsize=11, xycoords='axes fraction', bbox=props)
        else:
            plt.annotate(f"Predict Other will win in {state}", xy=(0.2, 0.1), fontsize=11, xycoords='axes fraction', bbox=props)
        plt.xlabel('Total Votes')
        plt.title(f"{new_year} Senate Elections Predictions in {state}")
        plt.show()


# In[10]:


def nn_2layers():
    df = pd.read_csv(csv_file)
    #Create category_columns and numeric_columns variables
    numeric_columns = []
    category_columns = []
    for col in df.columns:
        if is_string_dtype(df[col]) == True:
            category_columns.append(col)
        elif is_numeric_dtype(df[col]) == True:
            numeric_columns.append(col)
    #Create dummy variables for the category_columns and merge on the numeric_columns to create an X dataset
    category_columns = pd.get_dummies(df[category_columns])
    X = df[numeric_columns].merge(category_columns, left_index= True, right_index= True)
    X = X.fillna(0)
    #Create an y dataset
    y = df['totalvotes'].values
    # Split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # Scale X_train and X_test
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Create a neural network model with keras
    nn = tf.keras.models.Sequential()
    # Add a hidden layer with twice as many neurons as there are inputs. Use 'relu'
    n_input = len(X_train_scaled[0])

    n_hidden = n_input * 2
    n_hidden_layer2 = n_input * 2 #2nd hidden layer

    nn.add(tf.keras.layers.Dense(units=n_hidden, input_dim=n_input, activation='relu'))
    nn.add(tf.keras.layers.Dense(units=n_hidden_layer2, activation='relu')) #2nd hidden layer

    # add an output layer with a 'linear' activation function.
    nn.add(tf.keras.layers.Dense(units=1,  activation='linear'))
    # print a summary of the model
    
    print(nn.summary())
    # compile the model using the "adam" optimizer and "mean_squared_error" loss function
    nn.compile(loss='mean_squared_error' , optimizer='adam' , metrics=['mse'])
    # train the model for 100 epochs
    model = nn.fit(X_train_scaled, y_train, epochs=100)
    # predict values for the train and test sets
    y_train_pred = nn.predict(X_train_scaled)
    y_test_pred = nn.predict(X_test_scaled)
    # score the training predictions with r2_score()
    print(f"r2_score of y_train: {r2_score(y_train, y_train_pred)}")
    # score the test predictions with r2_score()
    print(f"r2_score of y_test: {r2_score(y_test, y_test_pred)}")


# In[11]:


def nn_1layer():
    df = pd.read_csv(csv_file)
    #Create category_columns and numeric_columns variables
    numeric_columns = []
    category_columns = []
    for col in df.columns:
        if is_string_dtype(df[col]) == True:
            category_columns.append(col)
        elif is_numeric_dtype(df[col]) == True:
            numeric_columns.append(col)
    #Create dummy variables for the category_columns and merge on the numeric_columns to create an X dataset
    category_columns = pd.get_dummies(df[category_columns])
    X = df[numeric_columns].merge(category_columns, left_index= True, right_index= True)
    #Create an y dataset
    y = df['totalvotes'].values
    # Split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # Scale X_train and X_test
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Create a neural network model with keras
    nn = tf.keras.models.Sequential()
    # Add a hidden layer with twice as many neurons as there are inputs. Use 'relu'
    n_input = len(X_train_scaled[0])

    n_hidden = n_input * 2
    #n_hidden_layer2 = n_input * 2 #2nd hidden layer

    nn.add(tf.keras.layers.Dense(units=n_hidden, input_dim=n_input, activation='relu'))
    #nn.add(tf.keras.layers.Dense(units=n_hidden_layer2, activation='relu')) #2nd hidden layer

    # add an output layer with a 'linear' activation function.
    nn.add(tf.keras.layers.Dense(units=1,  activation='linear'))
    # print a summary of the model
    print(nn.summary())
    # compile the model using the "adam" optimizer and "mean_squared_error" loss function
    nn.compile(loss='mean_squared_error' , optimizer='adam' , metrics=['mse'])
    # train the model for 100 epochs
    model = nn.fit(X_train_scaled, y_train, epochs=100)
    # predict values for the train and test sets
    y_train_pred = nn.predict(X_train_scaled)
    y_test_pred = nn.predict(X_test_scaled)
    # score the training predictions with r2_score()
    print(f"r2_score of y_train: {r2_score(y_train, y_train_pred)}")
    # score the test predictions with r2_score()
    print(f"r2_score of y_test: {r2_score(y_test, y_test_pred)}")


# In[12]:


def predict_2():
    from sklearn.ensemble import RandomForestRegressor
    df = pd.read_csv(csv_file)
    df = df[df.loc[:] != 'LIB'].dropna()
    df = df.drop(df.columns[0], axis=1)
    df.party = df.party.replace({'DEM': 1, 'REP': 2})
    dataset = df.drop(['state', 'party'],axis=1)
    # the last column is our label
    y_train = dataset.iloc[:,-1:]
    #drop last column of data
    X_train = dataset.iloc[:, :-1]
    #drop first colum of data
    X_test = dataset.iloc[:,1:]
    model = RandomForestRegressor(max_depth=5, random_state=1, n_estimators=1000).fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print(y_pred)
    print(f"Predict_Score: {model.score(X_train, y_train)}")
    prediction = pd.DataFrame({'state':df.state,'party':df.party, 'prediction_2024': y_pred.astype(int)})
    my_colors = list(islice(cycle(['b', 'r']), None, len(prediction)))
    prediction.party = prediction.party.map({1: 'DEM', 2: 'REP'})
    prediction.groupby('party')['prediction_2024'].sum().plot.bar(ylabel= "candidatevotes", title="2024 Party Prediction", color=my_colors)
    predict_winner = prediction.groupby('party')['prediction_2024'].sum()
    print(predict_winner)
    plt.show()


# In[13]:


def minority_2(): 
    import plotly.figure_factory as ff
    import numpy as np 
    import pandas as pd
    import plotly as py

    NE_states = ['Georgia', 'South Carolina']
    df = pd.read_csv(csv_file)
    df = df[df['STNAME'].isin(NE_states)]

    values = df['TOT_POP'].tolist()
    fips = df['FIPS'].tolist()
    count=df['Black'].tolist()

    color = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
                "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
                "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f",
                "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
                "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
                "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f",
                "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
                "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
                "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f"]
    colorscale = color * 6

    fig = ff.create_choropleth(
        fips=fips, values=values,
        #colorscale=colorscale, round_legend_values=True,
        simplify_county=0, simplify_state=0,
        scope=NE_states, county_outline={'color': 'rgb(255,255,255)', 'width': 0.5},
        state_outline={'color': 'rgb(0,0,0)', 'width': 2},
        #show_hover=True, centroid_marker={'opacity': 1},
        legend_title='Population per county',
        title='Georgia vs South Carolina Population'

    )

    fig.layout.template = None
    fig.show()
    py.offline.plot(fig,
    filename='Georgia vs South Carolina.html',
    include_plotlyjs='https://cdn.plot.ly/plotly-1.42.3.min.js')


# In[14]:


def county_4():
    from urllib.request import urlopen
    import plotly as py
    import json
    import webbrowser
    df = pd.read_csv(csv_file)
    #df['fips'] = df['fips'].apply(lambda x: '0'+x if len(x) == 4 else x)
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
    #geojson = px.data.election_geojson()
    #df = counties_fips_color[(counties_fips_color["state_code"] == "GA") | (counties_fips_color["state_code"] == "FL")]
    states = ['GA', 'SC', 'FL']
    df = df[df["state_code"].isin(states)]
    #df = counties_fips_color
    fig = px.choropleth(df, geojson=counties, locations='fips', color='color',
                                scope="usa",
                            
                            hover_data=["state","county", "candidate", "total_votes"])
    fig.update_geos(
                #lonaxis_range=[20, 380],
                projection_scale=2.7,
                center=dict(lat=31, lon=-83),
                visible=True)                      
    fig.update_layout(title= {"text": "Georgia vs South Carolina & Florida'\n' 2020 swing states total_votes", "xanchor": "center", "x": 0.5, "y": 0.95}, 
        margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    fig.show()
    fig.write_html("myplot.html")
    url = 'file://file:///Users/hiep_pham/Desktop/Analysis_Projects/Final_Project/myplot.html'
    webbrowser.open(url, new=2)  # open in new tab


# In[15]:


def county_3():
    df = pd.read_csv(csv_file)
    plt.scatter(df.median_age,df.total_votes)
    plt.xlabel("median_age")
    plt.ylabel('Total_Votes')
    plt.title("median_age vs Total_Votes")


# In[16]:


def county_2():
    df = pd.read_csv(csv_file)
    y = df.total_votes
    X = df.population.values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    plt.scatter(X,y)
    plt.plot(X, y_pred, color='red')
    plt.xlabel("Population")
    plt.ylabel('Total_Votes')
    plt.title("Population vs Total_Votes'\n' Linear Regression")


# In[17]:


def cost_3():
    df = pd.read_csv(csv_file)
    df = df[['year','House winner spending', 'Senate winner spending']] 
    df = df.sort_values(by='year', ascending=True)
    df.plot(
    x = 'year', 
    kind = 'barh', 
    stacked = True, 
    title = 'House winner vs. Senate winner Spending',)
    plt.xlabel('Spending')
    plt.show()


# In[18]:


def cost_2():
    df = pd.read_csv(csv_file)
    df = df[['year','Democrats', 'Republicans']] 
    df = df.sort_values(by='year', ascending=True)
    df.plot(
    x = 'year', 
    kind = 'barh', 
    stacked = True, 
    title = 'Democrats vs. Republican Spending',)
    plt.xlabel('Spending')
    plt.show()


# In[19]:


def df():
    df = pd.read_csv(csv_file)
    print(df)


# In[20]:


while True:
       plt.close()
       csv_file = input("Import CSV_file to analyze: ").strip()
       file_name1 = os.path.split(os.path.abspath(csv_file))[-1] 
       if len(csv_file) < 2: #hit enter to quit csv files while loop(enter is len = 1) 
              break  
       elif not file_name1.endswith('csv'):
              continue  # ignore it if not csv files
       file_name, file_extension = os.path.splitext(file_name1)   

       list1 = ["Option 1: Print Data Frame",
              "Option 2: Democrats vs. Republican Spending",
              "Option 3: House winner vs. Senate winner Spending"
              ]
       list2 = ["Option 1: Print Data Frame",
              "Option 2: Population vs Total_Votes & Linear Regression",
              "Option 3: Median Age vs Total_Votes",
              "Option 4: 2020 swing states total_votes(Georgia vs South Carolina & Florida)",
              ]
       list3 = ["Option 1: Print Data Frame",
              "Option 2: Georgia vs South Carolina Population",
              ]
       list4 = ["Option 1: Print Data Frame",
              "Option 2: 2024 Party Prediction (Randomforestclassifier)",
              ]
       list5 = ["Option 1: Print Data Frame",
              "Option 2: neural network with 1 hidden layer",
              "Option 3: neural network with 2 hidden layers",
              "Option 4: Senate Elections Predictions By State",
              ]
       list6 = ["Option 1: Print Data Frame",
              "Option 2: Top Ten Payeer",
              "Option 3: Top Ten Category vs Expenditure Amount",
              "Option 4: Number of contributor Supporting Trump by state",
              "Option 5: Number of contributor Supporting Biden by state",
              "Option 6: Trump vs Biden Supporting Rate",
              "Option 7: neural network with 2 hidden layers",
              ]
       list7 = ["Option 1: Print Data Frame",
              "Option 2: Top Ten Absentee Ballot Rejection by State",
              "Option 3: Top Ten Provisional Ballot Rejection by State",
              "Option 4: Top Ten UOCAVA Ballot Rejection by State",
              "Option 5: Top Ten VEP Turnout By State ",
              "Option 6: Predict Top Ten VEP Turnout By State",
              ]
       list8 = ["Option 1: Print Data Frame",
              "Option 2: Swing States Contributions",
              "Option 3: Swing States Margin",      
              ]
       list9 = ["Option 1: Print Data Frame",
              "Option 2: Georgia Votes By County",
              "Option 3: Democrats vs. Republican Rate in Georgia",
              "Option 4: Atlanta Metro Votes",
              "Option 5: Georgia Votes By Race",
              "Option 6: Demographic Comparisons",
              ]
       list10 = ["Option 1: Print Data Frame",
              "Option 2: Swing States By Population",
              "Option 3: Racial Breakdown Of Swing States",
              "Option 4: Gender By Population",
              "Option 5: 2019 Swing States Population",
              ]
       while True and len(file_name) > 2:
              if file_name.find('CostOf') != -1:
                     print(blue(f"The following options are available for {file_name1}:"))
                     print(*list1,sep='\n')
                     func = input("Please input the option #: ")
                     print("")
              elif file_name.find('president_counties') != -1:
                     print(blue(f"The following options are available for {file_name1}:"))
                     print(*list2,sep='\n')
                     func = input("Please input the option #: ")
                     print("")
              elif file_name.find('minority') != -1:
                     print(blue(f"The following options are available for {file_name1}:"))
                     print(*list3,sep='\n')
                     func = input("Please input the option #: ")
                     print("")
              elif file_name.find('president_dataset') != -1:
                     print(blue(f"The following options are available for {file_name1}:"))
                     print(*list4,sep='\n')
                     func = input("Please input the option #: ")
                     print("")
              elif file_name.find('senate_dataset') != -1:
                     print(blue(f"The following options are available for {file_name1}:"))
                     print(*list5,sep='\n')
                     func = input("Please input the option #: ")
                     print("")
              elif file_name.find('independent_expenditures') != -1:
                     print(blue(f"The following options are available for {file_name1}:"))
                     print(*list6,sep='\n')
                     func = input("Please input the option #: ")
                     print("")
              elif file_name.find('epi') != -1:
                     print(blue(f"The following options are available for {file_name1}:"))
                     print(*list7,sep='\n')
                     func = input("Please input the option #: ")
                     print("")
              elif file_name.find('swing_state') != -1:
                     print(blue(f"The following options are available for {file_name1}:"))
                     print(*list8,sep='\n')
                     func = input("Please input the option #: ")
                     print("")
              elif file_name.find('Georgia') != -1:
                     print(blue(f"The following options are available for {file_name1}:"))
                     print(*list9,sep='\n')
                     func = input("Please input the option #: ")
                     print("")
              elif file_name.find('race') != -1:
                     print(blue(f"The following options are available for {file_name1}:"))
                     print(*list10,sep='\n')
                     func = input("Please input the option #: ")
                     print("")
              
              if func == "": #hit enter to quit function while loop
                     break
              elif func not in ("1", "2", "3", "4", "5", '6','7'):
                     print(red("Typo! Please try again."))
              if func == "1" and file_name.find('CostOf') != -1:
                     df()
              elif func == "2" and file_name.find('CostOf') != -1:
                     cost_2()
              elif func == "3" and file_name.find('CostOf') != -1:
                     cost_3()
              elif func == "1" and file_name.find('president_counties') != -1:
                     df()       
              elif func == "2" and file_name.find('president_counties') != -1:
                     county_2()
              elif func == "3" and file_name.find('president_counties') != -1:
                     county_3()
              elif func == "4" and file_name.find('president_counties') != -1:
                     county_4()
              elif func == "1" and file_name.find('minority') != -1:
                     df()
              elif func == "2" and file_name.find('minority') != -1:
                     minority_2()
              elif func == "1" and file_name.find('president_dataset') != -1:
                     df()
              elif func == "2" and file_name.find('president_dataset') != -1:
                     predict_2()
              elif func == "1" and file_name.find('senate_dataset') != -1:
                     df()
              elif func == "2" and file_name.find('senate_dataset') != -1:
                     nn_1layer()
              elif func == "3" and file_name.find('senate_dataset') != -1:
                     nn_2layers()
              elif func == "4" and file_name.find('senate_dataset') != -1:
                     senate_predict()
              elif func == "1" and file_name.find('independent_expenditures') != -1:
                     df()
              elif func == "2" and file_name.find('independent_expenditures') != -1:
                     expenditures_2()
              elif func == "3" and file_name.find('independent_expenditures') != -1:
                     expenditures_3()
              elif func == "4" and file_name.find('independent_expenditures') != -1:
                     expenditures_4()
              elif func == "5" and file_name.find('independent_expenditures') != -1:
                     expenditures_5()
              elif func == "6" and file_name.find('independent_expenditures') != -1:
                     expenditures_6()
              elif func == "7" and file_name.find('independent_expenditures') != -1:
                     r2_2layers()
              elif func == "1" and file_name.find('epi') != -1:
                     df()
              elif func == "2" and file_name.find('epi') != -1:
                     epi2()
              elif func == "3" and file_name.find('epi') != -1:
                     epi3()
              elif func == "4" and file_name.find('epi') != -1:
                     epi4()
              elif func == "5" and file_name.find('epi') != -1:
                     epi5()
              elif func == "6" and file_name.find('epi') != -1:
                     epi6()
              if func == "1" and file_name.find('swing_state') != -1:
                     df()
              elif func == "2" and file_name.find('swing_state') != -1:
                     swing_state_2()
              elif func == "3" and file_name.find('swing_state') != -1:
                     swing_state_3()
              elif func == "1" and file_name.find('Georgia') != -1:
                     df()
              elif func == "2" and file_name.find('Georgia') != -1:
                     georgia_2()
              elif func == "3" and file_name.find('Georgia') != -1:
                     georgia_3()
              elif func == "4" and file_name.find('Georgia') != -1:
                     georgia_4()
              elif func == "5" and file_name.find('Georgia') != -1:
                     georgia_5()
              elif func == "6" and file_name.find('Georgia') != -1:
                     georgia_6()
              elif func == "1" and file_name.find('race') != -1:
                     df()
              elif func == "2" and file_name.find('race') != -1:
                     race_2()
              elif func == "3" and file_name.find('race') != -1:
                     race_3()
              elif func == "4" and file_name.find('race') != -1:
                     race_4()
              elif func == "5" and file_name.find('race') != -1:
                     race_5()
              
              plt.show()
              plt.close()
 

