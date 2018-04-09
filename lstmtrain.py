#conding:utf-8

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import random
import sys
import io

print('Load log string...')
data = pd.read_csv('./data/dljydb.log', header=None)
logstrs = data[2].values
print('log counts:', len(logstrs))

logdict = sorted(list(set(logstrs)))
print ('log dict:', len(logdict))
log_indices = dict((c, i) for i, c in enumerate(logdict))
indices_log = dict((i, c) for i, c in enumerate(logdict))

val_num = 3000#last val_num logs as valuation set
timestep = 30
step = 3
log_seqs = []
next_logstr = []
for i in range(0, len(logstrs) - timestep - val_num, step):
    log_seqs.append(logstrs[i: i + timestep])
    next_logstr.append(logstrs[i + timestep])
print('training log_seqs counts:', len(log_seqs))

print('Vectorization...')
x = np.zeros((len(log_seqs), timestep, len(logdict)), dtype=np.bool)
y = np.zeros((len(log_seqs), len(logdict)), dtype=np.bool)
for i, log_seq in enumerate(log_seqs):
    for t, logstr in enumerate(log_seq):
        x[i, t, log_indices[logstr]] = 1
    y[i, log_indices[next_logstr[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(timestep, len(logdict))))
#model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(logdict), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#from keras.utils import plot_model
#plot_model(model, to_file='./results/model.png')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#on each epoch end, test model precison
def on_epoch_end(epoch, logs):
    print('\n----- Performance after Epoch: %d' % epoch)

    predict_len = 20
    precision =0.                                                                 
    recall = 0.
    ave_time = 20                                                                   
    for at in range(ave_time):
        start_index = random.randint(len(logstrs) - val_num, 
                len(logstrs) - timestep - predict_len - 1)

        generated = []
        reallogidx = []
        log_seq = logstrs[start_index: start_index + timestep]

        precision_s =0.
        recall_s = 0.
        for i in range(predict_len): #generated predict_len logs
            x_pred = np.zeros((1, timestep, len(logdict)))
            for t, logstr in enumerate(log_seq):
                x_pred[0, t, log_indices[logstr]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            #next_index = sample(preds, diversity)
            next_index = np.argmax(preds)
            next_logstr = indices_log[next_index]
            generated.append(next_index)
            reallogidx.append(log_indices[logstrs[start_index + timestep+i]])

            log_seq = np.append(log_seq[1:],next_logstr)

            #print('%d: %d: %s' % (i,next_index,next_logstr))
            #print('%d: %d: %s' %(i, log_indices[logstrs[start_index + timestep+i]],
            #        logstrs[start_index + timestep+i]))
            #sys.stdout.flush()
        for i in range(predict_len):
            if (generated[i] in reallogidx):
                precision_s += 1.
            if (reallogidx[i] in generated):
                recall_s += 1.
        precision_s = precision_s/predict_len
        precision += precision_s
        recall_s = recall_s/predict_len
        recall += recall_s
        print('%d val precision: %.4f' % (at, precision_s))
        print('%d val recall: %.4f' %(at, recall_s))
    precision = precision/ave_time
    recall = recall /ave_time
    print('ave val precision: %.4f' % precision)
    print('ave val recall: %.4f' % recall)
    print ()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=2,
          callbacks=[print_callback])

#print("Saved model architecture to disk")
model_json = model.to_json()
with open("./results/model_arch.json", "w") as json_file:
    json_file.write(model_json)
#print("Saved model weights to disk")
model.save_weights("./results/model_w.h5")
