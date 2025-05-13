import h5py
import pandas as pd
import numpy as np

def convert_h5_to_csv_METR(h5_path, csv_path):
    with h5py.File(h5_path, 'r') as hdf:
        # 获取原始时间戳数值（不转换为字符串）
        raw_timestamps = hdf['df']['axis1'][:]  # 保持为numpy数组
        
        # 明确指定为纳秒级Unix时间戳
        time_index = pd.to_datetime(raw_timestamps, unit='ns')
        
        # 其余处理保持不变
        data = hdf['df']['block0_values'][:]
        columns = hdf['df']['axis0'][:].astype(str)
        df = pd.DataFrame(data, index=time_index, columns=columns)
        df.to_csv(csv_path)
        
        # 保存为CSV
        df.to_csv(csv_path)
        print(f"成功转换 {h5_path} -> {csv_path}")
        print(f"输出形状: {df.shape} (时间步×传感器)")

def convert_h5_to_csv_PEMS(h5_path, csv_path):
    with h5py.File(h5_path, 'r') as hdf:
        # 获取原始时间戳数值（不转换为字符串）
        raw_timestamps = hdf['speed']['axis1'][:]  # 保持为numpy数组
        
        # 明确指定为纳秒级Unix时间戳
        time_index = pd.to_datetime(raw_timestamps, unit='ns')
        
        # 其余处理保持不变
        data = hdf['speed']['block0_values'][:]
        columns = hdf['speed']['axis0'][:].astype(str)
        df = pd.DataFrame(data, index=time_index, columns=columns)
        df.to_csv(csv_path)
        
        # 保存为CSV
        df.to_csv(csv_path)
        print(f"成功转换 {h5_path} -> {csv_path}")
        print(f"输出形状: {df.shape} (时间步×传感器)")

# 执行转换
# convert_h5_to_csv_METR('data/METR-LA/METR-LA.h5', 'data/METR-LA/METR-LA.csv')
convert_h5_to_csv_PEMS('data/PEMS-bay/pems-bay.h5', 'data/PEMS-bay/pems-bay.csv')
