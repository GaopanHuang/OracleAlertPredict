# OracleAlertPredict
This project proposes a new deeplearning model, lstm+cnn, for oracle alert log predicting. The model consists of two parts: lstm and cnn. LSTM is the former encoder based on sequence-to-sequence model in translation, then the state h and c is formed into two channel signals, followed by cnn model. Here is the steps:


1.preprocessing: original logs to sequences file

    python genlogstr.py

    modify logfir to your own


2.preprocessing: remove switch logs

    python feature.py


3.train model

    python lstmseq.py


Wtih my own oracle alert logs, the accuracy of predicting error log is 91.3%.


Meanwhile, the model can be used in other domains of time sequence predicting, such as traffic road flow predicting, stock price predicting, and so on.
