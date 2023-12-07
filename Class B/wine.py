import pandas as pd
red_df = pd.read_csv('./winequality-red.csv', sep = ';', header = 0, engine = 'python')
white_df = pd.read_csv('./winequality-white.csv', sep = ';', header = 0, engine= 'python')
red_df.to_csv('./winequality-red2.csv', index = False)
white_df.to_csv('./winequality-white2.csv', index = False)

print (red_df.head())
red_df.insert(0, column = 'type', value = 'red')
print (red_df.head())
