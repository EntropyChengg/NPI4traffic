import h5py


# 示例用法
metr_la_path = "data/METR-LA/METR-LA.h5"  # 替换为实际路径
pems_bay_path = "data/PEMS-bay/pems-bay.h5" # 替换为实际路径

hdf = h5py.File(pems_bay_path, 'r')
print("hdf.keys():", hdf.keys())
print("hdf['speed].keys():", hdf['speed'].keys())
print("hdf['speed']['axis0'].keys():", hdf['speed']['axis0'].shape)
print("hdf['speed']['axis1'].keys():", hdf['speed']['axis1'].shape)
print("hdf['speed']['block0_items'].keys():", hdf['speed']['block0_items'].shape)
print("hdf['speed']['block0_values'].keys():", hdf['speed']['block0_values'].shape)
# 读取数据
# data = hdf['df']['block0_values'][:]
# print("data:", data)
# 读取数据
# data = hdf['df']['block0_items'][:]
# print("data:", data)