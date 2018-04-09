#coding:utf-8
from keras.models import model_from_json
import numpy as np
import pandas as pd
import random

print("load model")
# load json and create model
json_file = open('./results/model_arch.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./results/model_w.h5")

print('Load log string...')
data = pd.read_csv('./data/dljydb.log', header=None)
logstrs = data[2].values
print('log counts:', len(logstrs))

logdict = sorted(list(set(logstrs)))
print ('log dict:', len(logdict))
log_indices = dict((c, i) for i, c in enumerate(logdict))
indices_log = dict((i, c) for i, c in enumerate(logdict))

val_num = 30000#last val_num logs as valuation set
timestep = 30

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # param temperature controls the probability difference level, the bigger, the less of difference
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def predictlogs():
    predict_len = 20
    precision =0.
    recall = 0.
    ave_time = 300
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

            preds = loaded_model.predict(x_pred, verbose=0)[0]

            next_index = sample(preds,0.6)
            #next_index = np.argmax(preds)

            #print np.argmax(preds),preds[np.argmax(preds)]                        
            #print next_index, preds[next_index]
            #print log_indices[logstrs[start_index + timestep+i]], preds[log_indices[logstrs[start_index + timestep+i]]] 

            next_logstr = indices_log[next_index]
            generated.append(next_index)
            reallogidx.append(log_indices[logstrs[start_index + timestep+i]])

            log_seq = np.append(log_seq[1:],next_logstr)

            print('%d: %d: %d: ' % (i,next_index,log_indices[logstrs[start_index + timestep+i]]))
            #print('predict: %d: %d: %s' % (i,next_index,next_logstr))
            #print('real: %d: %d: %s' %(i, log_indices[logstrs[start_index + timestep+i]], logstrs[start_index + timestep+i]))
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
        print('%d %d val precision: %.4f' % (at,start_index, precision_s))
        print('%d %d val recall: %.4f' %(at,start_index, recall_s))
    precision = precision/ave_time
    recall = recall /ave_time
    print('ave val precision: %.4f' % precision)
    print('ave val recall: %.4f' % recall)
    print ()


predictlogs()
