from os import walk

basedir='/home/huanggp/ora_alert_process/'
i = 0
pre = []
for (dirpath, dirnames, filenames) in walk(basedir+'oracle_alert/'):
  pre = dirnames
  break
for (dirpath, dirnames, filenames) in walk(basedir+'oracle_alert/'):
  #print dirpath, dirnames, filenames
  if not filenames:
    continue
  fw = open(basedir+'OracleAlertPredict/'+pre[i]+'_alert_ora.info','w')
  filenames.sort()
  for fn in filenames:
    file = open(dirpath+'/'+fn)
    #print dirpath+'/'+fn
    while 1:
      line = file.readline()
      if not line:
        break
      if (line.find('ORA-')>-1):
        fw.write(line)
  i += 1
  fw.close()
