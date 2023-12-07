import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import json
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import numpy as np


def country_list_gen(df):
    df.rename(columns={'Country Name':'country_name'},inplace=True)
    df['country_name'] = df['country_name'].apply(lambda row: row.lower())
    lists = df['country_name'].unique().tolist()
    with open('country_list.json','w', encoding='utf-8') as f:
        json.dump(lists, f, ensure_ascii=False,indent=4)
    return lists, df

def selecting_country(df,country):
    df = df.loc[df['country_name']==country]
    df.drop(['country_name','Country Code','Indicator Name','Indicator Code'],axis=1,inplace=True)
    df = df.T
    df.dropna(inplace=True)
    df = df.reset_index()
    return df

def prediction_model(df):
    x_ = df.iloc[:, 0]# year
    y_ = df.iloc[:, 1]# population
    x = df.iloc[:, 0].values.reshape(-1,1) # year
    y = df.iloc[:, 1].values.reshape(-1,1) # population

    model = LinearRegression()
    model.fit(x,y)
    return model, x_, y_

def prediction(model, year, x_, y_):
    result = int(model.coef_[0][0] * year + model.intercept_[0]) # prediction value
    next_year = []
    next_result = []

    # next 5 year
    for i in range(2019,year+5,1):
        next_year.append(i)

    # next 5 year prediction
    for j in range(len(next_year)):
        next_result.append(model.coef_[0][0] * next_year[j] + model.intercept_[0])

    year1 = x_.values.tolist()
    year1 = [int(x) for x in year1]
    pop1 = y_.values.tolist()

    #plot
    plt.figure(figsize=(10,7))
    plt.title("Population Prediction")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.plot(year1, pop1)
    plt.plot(next_year, next_result)
    plt.show()

    return result

def main():
    country = input("Please input the country name: ").lower()
    year = int(input("Please input the year to predict: "))
    df = pd.read_csv('pop.csv')
    lists, df = country_list_gen(df)
    if country in lists:
        df = selecting_country(df, country)
        model, x_, y_ = prediction_model(df)
        result = prediction(model,year, x_, y_)
        print(f"\n Result: {country.upper()} population in {year} will be {result:,d}")
    else:
        print('kindly check available country name and thier spelling from country_list.json')
    
if __name__ == "__main__":
    main()
    # Bangladesh
    # 2025