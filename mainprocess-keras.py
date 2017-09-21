import os
#gpu_id = '1,2'
gpu_id = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

import tensorflow as tf
#config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4

sess = tf.Session(config=config)

from keras import backend as K
K.set_session(sess)

import keras
from keras.models import Model
from keras.layers import LSTM, Dense,Input
import numpy as np
from keras import backend as K

data_dim = 59
timesteps = 250
predictsteps = 50
batch_size = 64
train_steps  = 2000
val_steps = 200
epochs = 80
train_num = train_steps*batch_size

def pred_acc(y_true, y_pred):
  class_center = np.loadtxt(open("cluster_center_500.csv","rb"),delimiter=",")
  y_t = K.reshape(x=y_true[:,-1*predictsteps:,:], shape=(-1,data_dim))
  y_p = K.reshape(x=y_pred[:,-1*predictsteps:,:], shape=(-1,data_dim))
  y_t1 = K.variable(value=np.array([]))
  y_p1 = K.variable(value=np.array([]))
  #accvar = K.variable(value=np.array([0.]))
  for i in range(batch_size*predictsteps):
    samp1 = K.reshape(K.tile(y_t[i],500), shape=(-1,data_dim))
    t_cls = K.reshape(K.cast(K.argmin(K.mean(K.square(samp1-class_center),axis=-1)), K.floatx()),shape=(1,))
    y_t1 = K.concatenate([y_t1, t_cls], axis=-1)

    samp2 = K.reshape(K.tile(y_p[i],500), shape=(-1,data_dim))
    p_cls =  K.reshape(K.cast(K.argmin(K.mean(K.square(samp2-class_center),axis=-1)), K.floatx()),shape=(1,))
    y_p1 = K.concatenate([y_p1, p_cls], axis=-1)

    #mn = K.reshape(K.mean(K.equal(t_cls, p_cls)),shape=(1,))
    #accvar = K.update_add(accvar, mn)
  #return accvar/(batch_size*predictsteps)
  return K.cast(K.equal(y_t1, y_p1),K.floatx())


input = Input(shape=(timesteps,data_dim), name='input')
x = LSTM(128, return_sequences=True)(input)
#x = LSTM(128, return_sequences=True)(x)
x = LSTM(64, return_sequences=True)(x)
#x = Dense(256, activation='relu')(x)
output = Dense(data_dim, activation='relu')(x)

model = Model(inputs=input, outputs=output)

model.compile(optimizer='rmsprop', loss= 'mean_squared_error', metrics=[pred_acc])

from keras.utils import plot_model                                                                                                                
plot_model(model, to_file='model2.png',show_shapes=True)

############################generator for getting data and model fit
def get_batch(log_hist, hist_class, i, batch_size):
  x = np.empty(shape=[0,250, 59])  
  y = np.empty(shape=[0,250, 59])
  index = i*batch_size
  padding =np.zeros(shape=(predictsteps,data_dim))
  for j in range(batch_size):
    ##generate each step input
    x1 = log_hist[index+j:index+j+(timesteps-predictsteps),:]
    x1 = np.append(x1, padding, axis=0)
    x1 = x1[np.newaxis,:,:]
    x = np.append(x, x1, axis=0)

    y1 = log_hist[index+j+1:index+j+1+timesteps,:]
    y1 = y1[np.newaxis,:,:]
    y = np.append(y, y1, axis=0)

  return x, y

def generate_train(batch_size):
  samples = np.loadtxt(open("cluster_hist_500_2.csv","rb"),delimiter=",")          
  abs_time = samples[:,0]
  diff_time = samples[:,1]                                                       
  hist_class = samples[:,2]
  log_hist = samples[:,3:]
  while 1:
    for i in range(train_steps):
      # create numpy arrays of input data
      # and labels, from each line in the file
      x,y = get_batch(log_hist, hist_class, i, batch_size)
      yield (x, y)

def generate_val(batch_size):
  samples = np.loadtxt(open("cluster_hist_500_2.csv","rb"),delimiter=",",skiprows=train_num)
  abs_time = samples[:,0]                                                                                                                         
  diff_time = samples[:,1]                                                                                                                        
  hist_class = samples[:,2]                                                                                                                       
  log_hist = samples[:,3:]
  while 1:
    for i in range(val_steps):                                                                                                                        
      # create numpy arrays of input data                                                                                                         
      # and labels, from each line in the file                                                                                                    
      x,y = get_batch(log_hist, hist_class, i, batch_size)                                                                                        
      yield (x, y)                                                                                                                                

#model.summary()
#print model.layers
from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='./logs64', write_images=True,histogram_freq=2, batch_size=batch_size, write_graph=True)

model.fit_generator(generator=generate_train(batch_size),steps_per_epoch=train_steps,
    epochs=epochs, callbacks=[tensorboard],
    validation_data=generate_val(batch_size), validation_steps=val_steps)
'''
###############################################
def get_data():                                               
  samples = np.loadtxt(open("cluster_hist_500.csv","rb"),delimiter=",")          
  abs_time = samples[:,0]                                                     
  diff_time = samples[:,1]                                                       
  hist_class = samples[:,2]                                                   
  log_hist = samples[:,3:]                             


  x = np.empty(shape=[0,250, 59])  
  y = np.empty(shape=[0,250, 59])
  padding =np.zeros(shape=(predictsteps,data_dim))
  for j in range(train_num):
    ##generate each step input
    x1 = log_hist[j:j+(timesteps-predictsteps),:]
    x1 = np.append(x1, padding, axis=0)
    x1 = x1[np.newaxis,:,:]
    x = np.append(x, x1, axis=0)

    y1 = log_hist[j+1:j+1+timesteps,:]
    y1 = y1[np.newaxis,:,:]
    y = np.append(y, y1, axis=0)
    print ('%d samples load!' % j)

  return (x, y) 

x,y = get_data()
print x.shape, y.shape

#model.summary()                                                              
#print model.layers
from keras.callbacks import TensorBoard                                       
tensorboard = TensorBoard(log_dir='./logs', write_images=True,histogram_freq=2, batch_size=batch_size, write_graph=True)                                                      
model.fit(x,y, batch_size=batch_size, epochs=20,callbacks=[tensorboard])
'''
######################save and load model
# serialize model to JSON
model_json = model.to_json()
with open("model-64.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model-64.h5")
print("Saved model to disk")
 
## load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
###############################################