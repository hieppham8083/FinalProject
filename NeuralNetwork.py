import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import r2_score
# Creating a Neural Network Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
st.set_option('deprecation.showPyplotGlobalUse', False)

def epi_nn(): 
    try:
        df = pd.read_csv("./Resources/epi.csv")
        df = df.drop(['state_abbv', 'state_fips','year', 'website_pollingplace', 'website_reg_status', 'website_precinct_ballot', 'website_absentee_status', 'website_provisional_status', 'online_reg', 'residual'],axis=1)
        df = df.fillna(0)
        X = df.drop(['vep_turnout'],axis=1)
        y = df['vep_turnout'].values
        #splitting Train and Test 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
        #standardization scaler - fit&transform on train, fit only on test
        s_scaler = StandardScaler()
        X_train = s_scaler.fit_transform(X_train.astype(np.float))
        X_test = s_scaler.transform(X_test.astype(np.float))

        # Multiple Liner Regression
        regressor = LinearRegression()  
        regressor.fit(X_train, y_train)
        #evaluate the model (intercept and slope)
        #print(regressor.intercept_)
        #print(regressor.coef_)
        #predicting the test set result
        y_pred = regressor.predict(X_test)
        #put results as a DataFrame
        coeff_df = pd.DataFrame(regressor.coef_, df.drop('vep_turnout',axis =1).columns, columns=['Coefficient']) 
        st.write(coeff_df)

        # visualizing residuals
        fig, ax = plt.subplots()
        fig = plt.figure(figsize=(10,5))
        residuals = (y_test- y_pred)
        sns.distplot(residuals)
        st.pyplot(fig)


        y_pred = regressor.predict(X_test)
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        df1 = df.head(10)
        st.write(df1)

        # evaluate the performance of the algorithm (MAE - MSE - RMSE)
        from sklearn import metrics
        st.write('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
        st.write('MSE:', metrics.mean_squared_error(y_test, y_pred))  
        st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        st.write('VarScore:',metrics.explained_variance_score(y_test,y_pred))

        # having 19 neuron is based on the number of available features
        model = Sequential()
        model.add(Dense(15,activation='relu'))
        model.add(Dense(15,activation='relu'))
        model.add(Dense(15,activation='relu'))
        model.add(Dense(15,activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='Adam',loss='mse')
        model.fit(x=X_train,y=y_train,
              validation_data=(X_test,y_test),
              batch_size=128,epochs=400)
        model.summary() 
        # predict values for the train and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        # score the training predictions with r2_score()
        result = r2_score(y_train, y_train_pred)
        st.write(f"r2_score = {result}")

        loss_df = pd.DataFrame(model.history.history)
        loss_df.plot()
        st.pyplot()

        y_pred = model.predict(X_test)
        # Visualizing Our predictions
        fig = plt.figure(figsize=(9,6))
        plt.scatter(y_test,y_pred)
        # Perfect predictions
        plt.plot(y_test,y_test,'r')
        st.pyplot()

        # visualizing residuals
        fig, ax = plt.subplots()
        fig = plt.figure()
        residuals = (y_test- y_pred)
        sns.distplot(residuals)
        st.pyplot(fig)
        return result
    except:
        pass

#epi_nn()


