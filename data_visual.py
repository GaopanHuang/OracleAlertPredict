import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

start = 15000
end = 20000

f = open('alert_hist.csv')
df = pd.read_csv(filepath_or_buffer = f, header=None)
log_hist = np.array(df.iloc[start:end,2:].values)

model = TSNE(n_components = 2, random_state = 0)
x = model.fit_transform(log_hist)
plt.scatter(x[:,1],x[:,0])
plt.show()
