import numpy as np

data = np.load('data/PEMS04/pems04.npz')
print("data.files:", data.files)
print("data['data'].shape:", data['data'].shape)

data2 = np.load('data/PEMS08/pems08.npz')
print("data2.files:", data2.files)
print("data2['data'].shape:", data2['data'].shape)