import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('Load log string...')
data = pd.read_csv('./data/dljydb.log', header=None)
logstrs = data[2].values
print('log counts:', len(logstrs))

data_proc = data.loc[data.iloc[:,2]!='Archived Log entry  added for thread  sequence  ID xfffffffffebec dest :\r\n',:]
data_proc = data_proc.loc[data_proc.iloc[:,2]!='Thread  advanced to log sequence  (LGWR switch)\r\n  Current log#  seq#  mem# : /hafs/oradata/dljydb/redo.log\r\n',:]
data_proc.to_csv('./data/dljydb_proc.csv',header=None,index=None)
print data_proc.shape
