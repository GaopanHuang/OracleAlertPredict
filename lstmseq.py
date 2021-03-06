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

import time
import keras
from keras.callbacks import LambdaCallback
from keras.models import Model
from keras.layers import Dense, Activation,Conv1D,Reshape,Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input, LSTM, Dropout,BatchNormalization
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

print('Load log string...')
data = pd.read_csv('./data/dljydb_proc.csv', header=None)
logstrs = data[2].values
abstime = data[1].values
print('log counts:', len(logstrs))

logdict = sorted(list(set(logstrs)))
log_dim = len(logdict)
print ('log dict:', log_dim)
log_indices = dict((c, i) for i, c in enumerate(logdict))
indices_log = dict((i, c) for i, c in enumerate(logdict))

time_len = 12*3600#half a day for training
time_future = 0.17*3600 #predict 6 hours in future
timestep = 0
#for i in range(len(abstime)):
#    count = 0
#    for j in range(i, len(abstime)):
#        if abstime[j]-abstime[i]<time_len:
#            count += 1
#        else:
#            break
#    if timestep < count:
#        timestep = count
timestep = 180 #1468 is 6 hours for 2 hours
print ('encoder input:', timestep)

print ('building model...')
batch_size = 128  # Batch size for training.
epochs = 100  # Number of epochs to train for.
train_num = int(0.9*len(logstrs))
val_num = int(0.1*len(logstrs)-3000)

encoder_inputs = Input(shape=(None, log_dim))
#encoder = LSTM(1024, return_state=True, return_sequences=True)(encoder_inputs)
#encoder = LSTM(1024, return_state=True, return_sequences=True)(encoder)
#encoder = LSTM(1024, return_state=True, return_sequences=True)(encoder)
#encoder = LSTM(1024, return_state=True, return_sequences=True)(encoder)
#encoder = LSTM(1024, return_state=True, return_sequences=True)(encoder)
#encoder = LSTM(1024, return_state=True, return_sequences=True)(encoder)
#encoder = LSTM(1024, return_state=True, return_sequences=True)(encoder)
encoder_outputs, state_h, state_c = LSTM(1024, return_state=True)(encoder_inputs)
encoder_states = keras.layers.concatenate([state_h, state_c])
#print(encoder_states.get_shape())
x = Reshape((1024, 2))(encoder_states)
#print (x.get_shape())

x = Conv1D(64, 5, strides=2, padding='valid')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)

x = Conv1D(64, 5, strides=4, padding='valid')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)

#x = Conv1D(32, 5, strides=2, padding='valid')(x)
#x = BatchNormalization()(x)
#x = Activation('relu')(x)
#x = Dropout(0.2)(x)

#x = Conv1D(32, 5, strides=2, padding='valid')(x)
#x = BatchNormalization()(x)
#x = Activation('relu')(x)
#x = Dropout(0.2)(x)

x = GlobalMaxPooling1D()(x)

# and a logistic layer -- let's say we have 2 classes
predictions = Dense(2, activation='softmax')(x)

model = Model(encoder_inputs, predictions)
model.compile(optimizer='Nadam', metrics=['accuracy'],loss='binary_crossentropy')

def generate_data(batch_size,num, index):
    while True:
        steps = int(num/batch_size)
        for i in range(steps):
            x = np.zeros(
                    (batch_size, timestep, log_dim),
                    dtype='float32')
            y = np.zeros(
                    (batch_size, 2),
                    dtype='float32')
            for m in range(batch_size):
                log_idx = np.random.randint(index,num+index)
                logstr = logstrs[log_idx]
                x[m, 0, log_indices[logstr]] = 1.
                for j in range(log_idx+1, len(logstrs)):
                    if abstime[j]-abstime[log_idx]<time_len:
                        logstr = logstrs[j]
                        if (j-log_idx)>(timestep-1):
                            break
                        x[m, j-log_idx, log_indices[logstr]] = 1.
                    else:
                        break
                tmp_str = ''
                for k in range(j+1, len(logstrs)):
                    if abstime[k]-abstime[j]<time_future:
                        tmp_str += logstrs[k]
                    else:
                        break
                if tmp_str.upper().find('ERROR')!=-1:
                    y[m, 1] = 1.
                y[m,0] = 1-y[m,1]
#            print ('\n              err: %.4f' % (1.*np.sum(y[:,1])/len(y)))
            yield (x, y)

print ('training...')
from keras.callbacks import TensorBoard, CSVLogger,EarlyStopping
class BatchPerf(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        print (time.strftime("\n%Y-%m-%d %H:%M:%S", time.localtime()))
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

batPerf = BatchPerf()

def prt(batch, logs):
    print ('\nbatch %d train loss: %.4f' % (batch, logs.get('loss')))
batch_print_callback = LambdaCallback(on_batch_end=prt)

tensorboard = TensorBoard(log_dir='./results/logs')
csv_logger = CSVLogger('./results/epoch_perf.log')
es = EarlyStopping(monitor='val_loss', mode='auto',patience=6)

his = model.fit_generator(generator=generate_data(batch_size, train_num, 0),
        steps_per_epoch=int(train_num/batch_size),
        epochs=epochs,
        callbacks=[tensorboard,csv_logger,batPerf,es],
        validation_data=generate_data(batch_size, val_num, train_num),
        validation_steps=int(val_num/batch_size))

loss_df = pd.DataFrame([batPerf.losses, batPerf.acc]).T
loss_df.to_csv('./results/batch_perf.log',header=['loss', 'acc'],index=None)

#print("Saved model architecture")
#model_json = model.to_json()
#with open("model_arch_lstm1.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5, saves the weights of the model as a HDF5 file.
#print("Saved model weights")
#model.save_weights("model_lstm1.h5")


test_num = 1000
test_x = np.zeros(
    (test_num, timestep, log_dim),
    dtype='float32')
test_y = np.zeros(
    (test_num, 2),
    dtype='float32')

for m in range(test_num):
    log_idx = np.random.randint(0,len(logstrs)-200)
    logstr = logstrs[log_idx]
    test_x[m, 0, log_indices[logstr]] = 1.
    for j in range(log_idx+1, len(logstrs)):
        if abstime[j]-abstime[log_idx]<time_len:
            logstr = logstrs[j]
            if (j-log_idx)>(timestep-1):
                break
            test_x[m, j-log_idx, log_indices[logstr]] = 1.
        else:
            break
    tmp_str = ''
    for k in range(j+1, len(logstrs)):
        if k>=len(logstrs):
            break
        if abstime[k]-abstime[j]<time_future:
            tmp_str += logstrs[k]
        else:
            break
    if tmp_str.upper().find('ERROR')!=-1:
        test_y[m, 1] = 1.
    test_y[m,0] = 1-test_y[m,1]
print ('\nerr: %.4f' % (1.*np.sum(test_y[:,1])/len(test_y)))

y_ = model.predict(test_x)
print ('true:')
print (test_y[:20])
print ('predict:')
print (y_[:20])
y_pred = np.zeros(y_.shape, dtype='float32')
for i in range(len(y_)):
    if y_[i,0]>y_[i,1]:
        y_pred[i,0] = 1.
    else:
        y_pred[i,1] = 1.
    
acc = np.mean(np.equal(y_pred, test_y))
logloss = log_loss(test_y,y_)
print ('acc:%f; logloss:%f' % (acc, logloss))
