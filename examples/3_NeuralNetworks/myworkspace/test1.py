import numpy as np

a = np.zeros((128,16))
b = np.zeros((128,64))
c = np.concatenate((a,b), axis=1)
print("c shape:" ,(c.shape))