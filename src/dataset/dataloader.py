import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class TrafficDataset(Dataset):
    def __init__(self, csv_path, seq_len=12, pred_len=3, target_sensor=None, mode='train', split_ratio=(0.7, 0.1, 0.2)):
        """
        交通流时间序列数据集类
        
        参数:
            csv_path: CSV文件路径
            seq_len: 输入序列长度(历史时间步数)
            pred_len: 预测序列长度(未来时间步数)
            target_sensor: 指定预测的目标传感器列名(None表示预测所有传感器)
            mode: 数据集模式('train', 'val', 'test')
            split_ratio: 训练/验证/测试集划分比例
        """
        # 1. 读取原始数据
        df = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
        
        # 2. 数据标准化
        self.scaler = StandardScaler()
        data = self.scaler.fit_transform(df.values)  # (total_timesteps, num_sensors)
        
        # 3. 数据集划分
        total_samples = len(data) - seq_len - pred_len + 1
        train_size = int(total_samples * split_ratio[0])
        val_size = int(total_samples * split_ratio[1])
        
        if mode == 'train':
            self.data = data[:train_size]
        elif mode == 'val':
            self.data = data[train_size:train_size+val_size]
        else:  # test
            self.data = data[train_size+val_size:]
        
        # 4. 参数存储
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_sensor = target_sensor
        self.num_sensors = data.shape[1]
        
        # 5. 目标列处理
        if target_sensor:
            self.target_col = df.columns.get_loc(target_sensor)
        else:
            self.target_col = None

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # 获取输入序列 (seq_len, num_sensors)
        x = self.data[idx:idx+self.seq_len]
        
        # 获取目标序列 (pred_len, num_sensors或pred_len, 1)
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        
        if self.target_col is not None:
            y = y[:, self.target_col:self.target_col+1]  # 单目标预测
            
        return torch.FloatTensor(x), torch.FloatTensor(y)

def create_data_loaders(csv_path, batch_size=64, **kwargs):
    """
    创建训练、验证、测试集的DataLoader
    
    返回:
        train_loader, val_loader, test_loader, scaler
    """
    # 创建不同模式的数据集
    train_set = TrafficDataset(csv_path, mode='train',  **kwargs)
    val_set = TrafficDataset(csv_path, mode='val', **kwargs)
    test_set = TrafficDataset(csv_path, mode='test', **kwargs)
    
    # 创建DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_set.scaler

# if __name__ == "__main__":
#     # 示例用法
#     train_loader, val_loader, test_loader, scaler = create_data_loaders(
#         csv_path='data/METR-LA/METR-LA.csv',
#         seq_len=24,
#         pred_len=6,
#         batch_size=32,
#         target_sensor=None  # 关键修改：设置为None表示多目标
#     )

#     def inverse_scale_3d(scaler, data_3d):
#         """处理三维数据的逆标准化"""
#         original_shape = data_3d.shape
#         # 展平为二维 (batch*seq_len, num_sensors)
#         flattened = data_3d.reshape(-1, original_shape[-1])
#         # 逆变换
#         inverted = scaler.inverse_transform(flattened)
#         # 恢复原始形状
#         return inverted.reshape(original_shape)

#     # 修改后的可视化函数
#     def visualize_data_loader(loader, scaler, num_samples=5):
#         for x, y in loader:
#             # 逆标准化
#             x_orig = inverse_scale_3d(scaler, x.numpy())  # (batch, seq_len, sensors)
#             y_orig = inverse_scale_3d(scaler, y.numpy())  # (batch, pred_len, sensors)
            
#             # 可视化第一个样本的第一个传感器
#             plt.figure(figsize=(12, 6))
#             plt.plot(x_orig[0, :, 0], label='Input')
#             plt.plot(range(len(x_orig[0]), len(x_orig[0])+len(y_orig[0])), 
#                     y_orig[0, :, 0], 'r--', label='Target')
#             plt.legend()
#             plt.show()
            
#             if (num_samples := num_samples - 1) <= 0:
#                 break
#     # 可视化训练集
#     visualize_data_loader(train_loader, scaler, num_samples=5)

