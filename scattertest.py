import matplotlib.pyplot as plt
import numpy as np

a=np.array([[1,2],[3,4],[5,6]])
plt.figure(1)
x1=plt.subplot(121)
x2=plt.subplot(122)
x1.scatter(a[:,0],a[:,1])
x2.scatter(a[:,0],a[:,1])
plt.show()
