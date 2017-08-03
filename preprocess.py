"""
transforming alter log to the histogram of characters with 10+26+24 bins,
where 26 for all lower characters, 10 for numbers, 24 for specific syymbols

the characters are as follows:
abcdefghijklmnopqrstuvwxyz
0123456789
~!@#$%&*()_-+=}]{[:;?>< 

by hgp 2017.7.31
"""

from os import walk
import re
import csv
import numpy as np

dirpath = '../oracle_alert/a'
log_time = re.compile('^(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2}\s\d{2}:\d{2}:\d{2}\s\d{4}')
def str2hist(seq):
  characters = '0123456789abcdefghijklmnopqrstuvwxyz~!@#$%&*()_-+=}]{[:;?>< '
  hist = np.zeros(60)
  seq.lower()
  for i in range(60):
    hist[i] = seq.count(characters[i])

  return hist

def time2num(seq):
  #example: Fri Oct 02 06:47:35 2015
  #basetime is Jan 01 00:00:00 2001
  year_int = 31536000
  month = {'Jan':0, 'Feb':2678400, 'Mar':5097600,'Apr':7776000,'May':10368000,'Jun':13046400,'Jul':15638400,
      'Aug':18316800,'Sep':20995200,'Oct':23587200,'Nov':26265600,'Dec':28857600}
  diff_day = 86400
  day = int(seq[8:10])
  hour = int(seq[11:13])
  min = int(seq[14:16])
  sec = int(seq[17:19])
  year = int(seq[20:24])
  diff_time = (year-2001)*year_int+(year-2001)/4*diff_day + month[seq[4:7]] + (day-1)*diff_day + hour*3600 + min*60 +sec

  return diff_time

def process():
  global dirpath
  global log_time
  i = 0
  for (dirpath, dirnames, filenames) in walk(dirpath):
    if not filenames:
      continue

    fw = open('alert_hist.info','w')
    csvw=csv.writer(fw)
    filenames.sort()
    #print filenames
    hist1 = np.zeros(60)
    hist = hist1.astype(int)
    for fn in filenames:
      fr = open(dirpath+'/'+fn)
      print fn
      
      startflag = 0
      timenum = 0
      flag = 0
      while 1:
        line = fr.readline()
        if not line:
          break

        if log_time.search(line):#search log time in one line by re module
          if startflag == 1:
            logrst = np.r_[timenum,hist]
            logrst = logrst.astype(int)
            csvw.writerow(logrst)
          timenum = time2num(line)
          startflag = 1
          flag = 1
        elif startflag == 1:
          if flag == 1:
            flag = 0
            hist = np.zeros(60)
          hist += str2hist(line)
      hist = hist.astype(int)
      logrst = np.r_[timenum,hist]
      logrest = logrst.astype(int)
      csvw.writerow(logrst)
    fw.close()

process()
