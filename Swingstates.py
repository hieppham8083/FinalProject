import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import ticker
import seaborn as sns
import numpy as np 
st.set_option('deprecation.showPyplotGlobalUse', False)


def slide(): 
    a = st.slider("Year #1", 2010, 2019, 2011)
    b = st.slider("Year #2", 2010, 2019, 2019)
    #c = st.slider("Year #3", 2010, 2019, 2019)
    return a, b
list_years = slide()
list_years = list(list_years)


#st.write(list_years)

@st.cache #avoid to load csv again
def load_data2():
    global list_years
    df = pd.read_csv("./Resources/state_race.csv")
    df[['Population']] = df[['Population']] .replace('[\,,]','',regex=True).astype(int)
    #years = [int(x) for x in input("Enter a list of year[2010 2015 2019] to test: ").split()]
    #years = list_years
    years = list_years
    df = df.loc[df.Year.isin(years)]
    return df

@st.cache #avoid to load csv again
def load_data():
    df = pd.read_csv("./Resources/swing_state_dataset.csv")
    return df

df5 = load_data2()
df = load_data()

def swing_states():
    global df5, list_years
    #years = list_years
    years = list_years
    df1 = df5.loc[df5.Year.isin([years[0]])]
    table_0 = df1.groupby(df5['State'])['Population'].sum()
    table_0 = table_0.sort_values(ascending=True)
    try:
        df2 = df5.loc[df5.Year.isin([years[1]])]
        table_1 = df2.groupby(df5['State'])['Population'].sum()
        table_1 = table_1.sort_values(ascending=True)
    except:
        pass
    try:
        df3 = df5.loc[df5.Year.isin([years[2]])]
        table_2 = df3.groupby(df5['State'])['Population'].sum()
        table_2 = table_2.sort_values(ascending=True)
    except:
        pass
    if len(years) == 1:
        df4 = pd.DataFrame({years[0]: table_0.values.tolist()}, index=table_0.index.tolist())
    elif len(years) == 2:  
        df4 = pd.DataFrame({years[0]: table_0.values.tolist(), years[1]: table_1.values.tolist()}, index=table_1.index.tolist()) 
    elif len(years) == 3:
        df4 = pd.DataFrame({years[0]: table_0.values.tolist(), years[1]: table_1.values.tolist(), years[2]: table_2.values.tolist()}, index=table_2.index.tolist()) 
    df4.plot.barh()
    plt.xlabel('Population (million)')
    plt.ylabel('Swing State')
    plt.title('Swing States By Population')
    st.pyplot()

def Race_population():
    global df5, list_years
    years = list_years
    df1 = df5.loc[df5.Year.isin([years[0]])]
    table_0 = df1.groupby(df5['Race'])['Population'].sum()
    table_0 = table_0.sort_values(ascending=True)
    try:
        df2 = df5.loc[df5.Year.isin([years[1]])]
        table_1 = df2.groupby(df5['Race'])['Population'].sum()
        table_1 = table_1.sort_values(ascending=True)
    except:
        pass
    try:
        df3 = df5.loc[df5.Year.isin([years[2]])]
        table_2 = df3.groupby(df5['Race'])['Population'].sum()
        table_2 = table_2.sort_values(ascending=True)
    except:
        pass
    if len(years) == 1:
        df4 = pd.DataFrame({years[0]: table_0.values.tolist()}, index=table_0.index.tolist())
    elif len(years) == 2:  
        df4 = pd.DataFrame({years[0]: table_0.values.tolist(), years[1]: table_1.values.tolist()}, index=table_1.index.tolist()) 
    elif len(years) == 3:
        df4 = pd.DataFrame({years[0]: table_0.values.tolist(), years[1]: table_1.values.tolist(), years[2]: table_2.values.tolist()}, index=table_2.index.tolist()) 
    df4.plot.barh()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.annotate(f"Total population of all 7 Swing States", xy=(0.3, 0.5), fontsize=11, xycoords='axes fraction', bbox=props)
    plt.xlabel('Population (million)')
    plt.ylabel('Race')
    plt.title('Race By Population')
    st.pyplot()

def Contributions():
    global df
    states = ["Arizona", "Florida", "Georgia", "Michigan", "Minnesota", "Nevada", "New Hampshire", "North Carolina", "Pennsylvania", "Texas"]
    df = df.loc[df.state.isin(states)]
    df[['2020_Total_Contributions', '2018_Total_Contributions','2016_Total_Contributions']] = df[['2020_Total_Contributions', '2018_Total_Contributions','2016_Total_Contributions']].replace('[\$,]','',regex=True).astype(float)
    df1 = pd.DataFrame({'2020': df['2020_Total_Contributions'].tolist(), '2018': df['2018_Total_Contributions'].tolist(), '2016': df['2016_Total_Contributions'].tolist()}, index=df['state'].tolist()) 
    df1.plot.barh()
    plt.xlabel('Total Contributions (million)')
    plt.ylabel('Swing State')
    plt.title('Swing States Contributions')
    st.pyplot()

def Margin():
    
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
    plt.annotate("2020", xy=(-0.55, 0.45), fontsize=7)
    plt.annotate("2016", xy=(-0.3, 3.6), fontsize=7)
    plt.annotate("2012", xy=(0.1, 9.15), fontsize=7)
    st.pyplot()

swing_states() 
Race_population()
Contributions()
Margin()
