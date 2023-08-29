import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
 
# neuralprophet

#______________________________import data_____________________________________
start = dt.datetime(2012, 1, 1)
end = dt.datetime.now()

df = yf.download("GOOGL" , start, end, interval="1d")
df["ds"] = df.index
df.rename(columns={'Close': 'y'}, inplace=True)
df.drop(['Open','Volume','Adj Close','High','Low'], axis= 1, inplace = True)

#_____________________________split Train/Test_________________________________
train_size = int(len(df) *0.8)
test_size = int(len(df) *0.2)
train_X = df[:train_size].dropna()
test_X = df[train_size:].dropna()

# #______________________________difference______________________________________

df = train_X[['y']].copy()
df['log'] = np.log(df.y)
df['close_diff'] = df.log.diff()
differenced = (df['close_diff'])
differenced.dropna(inplace= True, axis= 0)

differenced = differenced.to_frame()
differenced.columns = ['y']
differenced["ds"] = differenced.index

#______________________________Model___________________________________________
periods = 100
m = NeuralProphet()
metrics = m.fit(train_X)
df_future = m.make_future_dataframe(train_X, periods=periods)
forecast = m.predict(df_future)

##________________________inverse - difference__________________________________
preds = forecast["yhat1"].to_frame()
preds.index = forecast['ds']
preds.columns = ['close_diff']
df2 = df['close_diff'].to_frame()

frames = [df2,preds]
what = pd.concat(
       frames,
       axis=0,
       join="outer",
       ignore_index=False)

what['reverse_diff'] = what['close_diff']
what['reverse_diff'].iloc[0] = df['log'].iloc[0]
what['rev_log'] = np.exp(what['reverse_diff'].cumsum())



#________________________________evaluate______________________________________
evaluation = pd.DataFrame()
evaluation['Close'] = test_X['y'].iloc[:periods]
evaluation['forecast'] = what['rev_log'].tail(periods).values

#_____________________________Bollinger bands__________________________________

def Bollinger(data):
    df_Bollinger = data.copy()
    for column in df_Bollinger.columns :

        # std - different periods
        df_Bollinger['MA20dSTD'] = df_Bollinger[column].rolling(window=20).std() 
        # Moving Averages 
        df_Bollinger['MA20'] = df_Bollinger[column].rolling(window=20).mean()
        
        # Bollinger Bands
        df_Bollinger['Bollinger_Upper'] = df_Bollinger['MA20'] + (df_Bollinger['MA20dSTD'] * 2)
        df_Bollinger['Bollinger_Lower'] = df_Bollinger['MA20'] - (df_Bollinger['MA20dSTD'] * 2)
          
    df_Bollinger.drop(['MA20','MA20dSTD'],inplace= True, axis=1)
    return df_Bollinger


stock_Bollinger = Bollinger(what['rev_log'].iloc[950:].to_frame())


#________________________________plot forecast_________________________________
plt.plot(evaluation['Close'])
plt.plot(evaluation['forecast'])
plt.plot(stock_Bollinger['Bollinger_Upper'].tail(periods).values)
plt.plot(stock_Bollinger['Bollinger_Lower'].tail(periods).values)
plt.legend()
plt.show()




