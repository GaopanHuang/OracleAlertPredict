import numpy as np
import pandas as pd
import csv
from sklearn.cluster import KMeans

f = open('alert_hist.csv')
df = pd.read_csv(filepath_or_buffer = f, header=None)
abs_time = np.array(df.iloc[:,0].values)
diff_time = np.array(df.iloc[:,1].values)
log_hist = np.array(df.iloc[:,2:].values)

def getmaxlengthinoneday():
  intervaltime = 10000
  count = 0
  index = -1
  start_index = 0
  starttime = abs_time[0]
  maxlength = 0
  for curtime in abs_time:
    index += 1
    while (curtime-starttime > intervaltime):
      start_index += 1
      starttime = abs_time[start_index]
      count -= 1
    if (curtime-starttime <= intervaltime):
      count += 1
      if count > maxlength:
        maxlength = count
        #print "%d, %d" %(abs_time[index],maxlength)

  return maxlength

def kcluster(k=500):
  kmeans = KMeans(n_clusters = k).fit(log_hist)
  cluster_rst = np.c_[abs_time,diff_time,kmeans.labels_,log_hist]
  cluster_rst = cluster_rst.astype(int)
  fw = open('cluster_hist_500.csv','w')
  csvw=csv.writer(fw)
  csvw.writerows(cluster_rst)
  fw.close()

  fw = open('cluster_center_500.csv','w')
  csvw=csv.writer(fw)
  csvw.writerows(kmeans.cluster_centers_)
  fw.close()


#print getmaxlengthinoneday()
kcluster()
