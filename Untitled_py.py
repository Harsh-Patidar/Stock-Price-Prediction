#!/usr/bin/env python
# coding: utf-8

# In[89]:


import pandas_datareader as pdr
import os


# In[90]:


import json
credentials = {}
try:
    with open('credentials.json') as file:
        credentials = json.load(file)
except FileNotFoundError:
    print("Error: credentials.json file not found.")


# In[91]:


os.environ['TIINGO_API_KEY']= credentials


# In[92]:


key=os.getenv('TIINGO_API_KEY')


# In[93]:


df = pdr.get_data_tiingo('AAPL', api_key=key)


# In[94]:


df.to_csv('APPL.csv')


# In[95]:


import pandas as pd


# In[96]:


df=pd.read_csv('APPL.csv')


# In[97]:


df.head()


# In[98]:


df.tail()


# In[99]:


df1=df.reset_index()['close']


# In[100]:


df1.shape


# In[101]:


df1


# In[102]:


import matplotlib.pyplot as plt
plt.plot(df1)


# ###### LSTM are sensitive to the scale of the data. so we apply MinMax scaler

# In[103]:


import numpy as np


# In[104]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[105]:


df1.shape


# In[106]:


df1


# In[107]:


## splitting dataset into train and test split
training_size = int(len(df1)*0.65)
test_size = len(df1) - training_size
train_data, test_data = df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[108]:


training_size, test_size


# In[109]:


#len of training data
len(train_data), len(test_data)


# In[110]:


train_data


# In[111]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, timestep=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]  ### i=0, 0,1,2,3
        dataX.append(a)
        dataY.append(dataset[i+ time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[112]:


# reshape into X=t, t+1, t+2, t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[113]:


print(X_train)


# In[114]:


print(X_test.shape), print(ytest.shape)


# In[115]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[116]:


### Create the Stacked LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[117]:


model= Sequential()
model.add(LSTM(50,return_sequences = True, input_shape=(100,1)))
model.add(LSTM(50,return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[118]:


model.summary()


# In[119]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[120]:


import tensorflow as tf


# In[121]:


tf.__version__


# In[122]:


### lets Do the prediction and check performance metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[123]:


## Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[124]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[125]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest, test_predict))


# In[126]:


### Plotting
# shift train predictions for plotting
look_back = 100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
#plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[127]:


len(test_data)


# In[128]:


x_input = test_data[340:].reshape(1,-1)
x_input.shape


# In[129]:


temp_input = list(x_input)
temp_input = temp_input[0].tolist()


# In[130]:


temp_input


# In[131]:


# demostrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps = 100
i = 0
while(i<30):
    if (len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1,n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
        
print(lst_output)


# In[132]:


day_new = np.arange(1,101)
day_pred = np.arange(101,131)


# In[133]:


import matplotlib.pyplot as plt


# In[134]:


len(df1)


# In[135]:


df3=df1.tolist()
df3.extend(lst_output)


# In[136]:


plt.plot(day_new,scaler.inverse_transform(df1[1156:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[137]:


# if you want to see complete output
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1000:])


# In[138]:


#see in proper way start with 1200
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[139]:


## yes, we can improve the accuracy by using bidirectional LSTM


# In[ ]:





# In[ ]:




