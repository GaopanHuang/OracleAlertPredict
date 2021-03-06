#coding:utf-8

from os import walk
import re
import csv
import numpy as np
import pandas as pd

#read logfile
dirpath = './data/dljydb/'
logpath = './results/dljydb.log'
#dirpath = './data/test/'
log_time = re.compile('^(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2}\s\d{2}:\d{2}:\d{2}\s\d{4}')
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

  month_index = {'Jan':'01', 'Feb':'02', 'Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
  month = month_index[seq[4:7]]
  abs_time = seq[20:24]+month+seq[8:10]+seq[11:13]+seq[14:16]+seq[17:19]

  return int(abs_time),diff_time

#generate log string file, three columns: 'abstime','difftime','logstr'
def process():
  global dirpath
  global log_time
  for (dirpath, dirnames, filenames) in walk(dirpath):
    if not filenames:
      continue

    filenames.sort()
    #print filenames
    print 'generating log string'
    for fn in filenames:
      fr = open(dirpath+fn)
      print fn
      
      startflag = 0
      flag = 0
      abstime = 0
      difftime = 0
      logstr = ''
      while 1:
        line = fr.readline()
        if not line:
          break
          
        if log_time.search(line):#search log time in one line by re module
          if startflag == 1:            
            #record forehead log
            df = pd.DataFrame([[abstime, difftime, logstr]])
            df.to_csv(logpath,mode='a+',header=None,index=None)
          abstime, difftime = time2num(line)
          startflag = 1
          flag = 1
        elif startflag == 1:
          if flag == 1:
            flag = 0
            logstr = ''
          logstr += re.sub('\d','',line)#remove log number variable
          
      #record last log
      df = pd.DataFrame([[abstime, difftime, logstr]])
      df.to_csv(logpath,mode='a+',header=None,index=None)

process()
