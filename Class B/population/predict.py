"""
Author: @ Abid Ebna Saif Utsha
Created Date: 28/06/2020
A simple linear Regression model to predict a country
population.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import json
import warnings
warnings.filterwarnings("ignore")


def country_list_gen(df):
 #   df.rename(columns={'Seafood Name':'seafoodname'},inplace=True)
   # df['seafood name'] = df['seafood Name'].apply(lambda row: row.lower())
   lists = df['Seafood Name'].unique().tolist()
   with open('country_list.json','w', encoding='utf-8') as f:
    return lists, df

def selecting_country(df,country):
    """
    this function will 
    """
   # df = df.loc[df['country_name']==country]
  #  df.drop(['country_name','Country Code','Indicator Name','Indicator Code'],axis=1,inplace=True)
    df = df.T
    df.dropna(inplace=True)
    df = df.reset_index()
    return df

def prediction_model(df):
    x = df.iloc[:, 0].values.reshape(-1,1)
    y = df.iloc[:, 1].values.reshape(-1,1)
    model = LinearRegression().fit(x,y)
    return model

def prediction(model, year):
    return int(model.coef_[0][0] * year + model.intercept_[0])


def main():
    country = input("Please input the country name: ").lower()
    year = int(input("Please input the year to predict: "))
    df = pd.read_csv('Seafood.csv', encoding='latin1')

    lists, df = country_list_gen(df)
    if country in lists:
        df = selecting_country(df, country)
        model = prediction_model(df)
        result = prediction(model,year)
        print(f"\n Result: {country.upper()} population in {year} will be {result:,d}")
    else:
        print('kindly check available country name and thier spelling from country_list.json')
    
if __name__ == "__main__":
    main()