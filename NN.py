import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import ticker
import seaborn as sns



@st.cache #avoid to load csv again
def load_data():
    df = pd.read_csv("../main//Resources/senate_dataset.csv")
    return df

df = load_data()


def nn_2layers():
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from pandas.api.types import is_string_dtype
    from pandas.api.types import is_numeric_dtype
    global df
    try:
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
        #st.write(X)
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
        nn.summary()
        # compile the model using the "adam" optimizer and "mean_squared_error" loss function
        nn.compile(loss='mean_squared_error' , optimizer='adam' , metrics=['mse'])
        # train the model for 100 epoch
        model = nn.fit(X_train_scaled, y_train, epochs=50)
        # predict values for the train and test sets
        y_train_pred = nn.predict(X_train_scaled)
        y_test_pred = nn.predict(X_test_scaled)
        # score the training predictions with r2_score()
        result1 = r2_score(y_train, y_train_pred)
        result2 = r2_score(y_test, y_test_pred)
        st.write('\n')
        st.write(f"r2_score of y_train= {result1}")
        print(f"r2_score of y_train: {r2_score(y_train, y_train_pred)}")
        # score the test predictions with r2_score()
        st.write(f"r2_score of y_test= {result2}")
        print(f"r2_score of y_test: {r2_score(y_test, y_test_pred)}")
        return result1, result2
    except: 
        pass

def image2(): 
    from PIL import Image
    image = Image.open('test_in_progress.png')
    st.image(image, caption='Neural Network Model In Progress, the training model can be view in terminal',use_column_width=True)
    



