import numpy as np

data = np.load('data/PEMS04/pems04.npz')
print("data.files:", data.files)
print("data['data'].shape:", data['data'].shape)