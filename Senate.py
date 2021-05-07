import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import ticker
import seaborn as sns



@st.cache #avoid to load csv again
def load_data():
    df = pd.read_csv("CostOfElection.csv")
    return df

@st.cache #avoid to load csv again
def load_data2():
    df = pd.read_csv("senate.csv")
    return df

df = load_data()
df2 = load_data2()


def House_vs__Senate():
    cost_election2 = df[['year','House winner spending', 'Senate winner spending']] 
    cost_election2 = cost_election2.sort_values(by='year', ascending=True)
    cost_election2.plot(
    x = 'year', 
    kind = 'barh', 
    stacked = True, 
    title = 'House winner vs. Senate winner Spending',)
    plt.xlabel('Spending')
    st.pyplot()

def dem_vs_rep(): 
    cost_election1 = df[['year','Democrats', 'Republicans']] 
    cost_election1 = cost_election1.sort_values(by='year', ascending=True)
    cost_election1.plot(
    x = 'year', 
    kind = 'barh', 
    stacked = True, 
    title = 'Democrats vs. Republican Spending',)
    plt.xlabel('Spending')
    st.pyplot()

def party_history(): 
    df= df2[df2['Results'] != 0]
    states = ['AZ', 'FL', 'GA', 'MI', 'MN', 'NV', 'NH', 'NC', 'PA', 'TX']
    df = df.loc[df.state_po.isin(states)]
    dem = df[df.party_simplified == 'DEMOCRAT']
    dem = dem.groupby(dem['state_po'])['party_detailed'].count()
    rep = df[df.party_simplified == 'REPUBLICAN']
    rep = rep.groupby(rep['state_po'])['party_detailed'].count()
    df = pd.DataFrame({'DEMOCRAT': dem.values, 'REPUBLICAN': rep.values}, index = dem.index) 
    df.plot.barh()
    plt.xlabel('Count')
    plt.ylabel('Swing State')
    plt.title('History Winning Party from 1976 - 2020')
    st.pyplot()


def image(): 
    from PIL import Image
    image = Image.open('Senate_Election_Map.png')
    st.image(image, caption='2022 Senate Election Interactive Map',use_column_width=True)








