import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import ticker
import seaborn as sns



@st.cache #avoid to load csv again
def load_data():
    df = pd.read_csv("fec_independent_expenditures_model.csv", low_memory=False)
    return df

df = load_data()


def Payee():
    fig, ax = plt.subplots()
    table_count = df.groupby(df['payee_name'])['expenditure_amount'].sum()
    table_count = table_count.sort_values(ascending=False)[:10]
    payee_index = table_count.index
    payee_val = table_count.values
    sns.barplot(x = payee_val,y=payee_index,orient='h')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.annotate("Top Ten Payee", xy=(0.5, 0.5), fontsize=15, xycoords='axes fraction', bbox=props)
    plt.ylabel('Payee Name')
    plt.xlabel('Expenditure Amount')
    st.pyplot(fig)

def Report():
    fig, ax = plt.subplots()
    sns.countplot(df.report_year,ax=ax)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.annotate("Number of Reported Year", xy=(0.1, 0.5), fontsize=12, xycoords='axes fraction', bbox=props)
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.xticks(fontsize=6)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def Category():
    fig, ax = plt.subplots()
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
    st.pyplot(fig)




