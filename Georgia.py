import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import ticker
import seaborn as sns
import webbrowser




@st.cache #avoid to load csv again
def load_data():
    df = pd.read_csv("../main//Resources/Georgia_dataset.csv")
    return df

@st.cache #avoid to load csv again
def load_data2():
    df = pd.read_csv("../main//Resources/president_counties.csv")
    return df

df = load_data()
df2 = load_data2()

def map():
    col1, col2 = st.beta_columns(2)
    if col1.button('Georgia Counties Votes Map'):
        webbrowser.open('https://hieppham.s3.us-east-2.amazonaws.com/Final_project/georgia/Georgia+vs+South+Carolina+Total+Votes.html', new=2)
    if col2.button('Georgia Counties Population Map'):
        webbrowser.open('https://hieppham.s3.us-east-2.amazonaws.com/Final_project/georgia/Georgia+vs+South+Carolina+Population.html', new=2)     
def DEM_vs_REP():
    #fig, ax = plt.subplots()
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
    #st.pyplot(fig)
    st.pyplot()

def Metro():
    
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
    st.pyplot()

def Race():
  
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
    st.pyplot()

def Facts():

    columns = ['fact', 'Arizona', 'Georgia', 'South Carolina']
    df1 = df[columns]
    df1 = df1.dropna()
    df1[['Arizona', 'Georgia', 'South Carolina']] = df1[['Arizona', 'Georgia', 'South Carolina']].replace('[\%,]','',regex=True).astype(float)
    df1 = pd.DataFrame({'Georgia': df1['Georgia'].tolist(), 'Arizona': df1['Arizona'].tolist(), 'South Carolina': df1['South Carolina'].tolist()}, index=df1['fact'].tolist())
    df1.plot.barh()
    plt.xlabel('Percentage(%)')
    plt.ylabel('Facts')
    plt.title('Quick Facts Comparison')
    st.pyplot()

def map2():
    from urllib.request import urlopen
    import plotly as py
    import json
    import plotly.express as px
    fig, ax = plt.subplots()
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
    #geojson = px.data.election_geojson()
    #df = counties_fips_color[(counties_fips_color["state_code"] == "GA") | (counties_fips_color["state_code"] == "FL")]
    states = ['GA', 'SC', 'FL']
    df = df2[df2["state_code"].isin(states)]
    #df = counties_fips_color[counties_fips_color["state_code"].isin(states)]
    #df = counties_fips_color
    fig = px.choropleth(df, geojson=counties, locations='fips', color='color',
                                scope="usa",
                            
                            hover_data=["state","county", "candidate", "total_votes"])
    fig.update_geos(
                #lonaxis_range=[20, 380],
                projection_scale=2.7,
                center=dict(lat=31, lon=-83),
                visible=True)                      
    fig.update_layout(title= {"text": "Georgia vs South Carolina & Florida", "xanchor": "center", "x": 0.5, "y": 0.95}, 
        margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    st.pyplot(fig)







