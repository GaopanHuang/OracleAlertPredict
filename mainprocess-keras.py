from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd

data_dim = 59
timesteps = 128
num_classes = 59
train_num = 30000
#train_num = 90

f = open('alert_hist.csv')
df = pd.read_csv(filepath_or_buffer = f, header=None)
abs_time = np.array(df.iloc[:,0].values)
diff_time = np.array(df.iloc[:,1].values)
log_hist = np.array(df.iloc[-35058:,2:].values)
#log_hist = np.array(df.iloc[-304:,2:].values)

def getdata():
  train_data=log_hist[:train_num+timesteps+1]
  train_x,train_y=[],[]
  for i in range(train_num):
    x=train_data[i:i+timesteps,:]
    y=train_data[i+timesteps,:]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

  val_data=log_hist[train_num+timesteps+1:]
  val_x,val_y=[],[]
  for i in range(len(val_data)-timesteps-1):
    x=val_data[i:i+timesteps,:]
    y=val_data[i+timesteps,:]
    val_x.append(x.tolist())
    val_y.append(y.tolist())

  return train_x,train_y,val_x,val_y

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(128, return_sequences=True,
               input_shape=(timesteps, data_dim),batch_size=30,stateful = True))
#model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128,stateful = True))
model.add(Dense(59, activation='relu'))

model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['accuracy'])


x_train,y_train,x_val,y_val=getdata()
#model.summary()
#print model.layers
model.fit(x_train, y_train,
          batch_size=30, epochs=50,
          validation_data=(x_val, y_val),shuffle = False)
