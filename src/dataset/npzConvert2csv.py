import numpy as np
import pandas as pd

def extract_flow_from_npz_04(npz_path, output_csv=None):
    """
    从PEMS04的npz文件中提取流量数据
    :param npz_path: .npz文件路径
    :param output_csv: 输出CSV路径（可选）
    :return: 流量DataFrame
    """
    # 加载npz文件
    data = np.load(npz_path)
    
    # 确认数据结构
    print("文件中包含的数组:", list(data.keys()))
    print("主数据形状:", data['data'].shape if 'data' in data else "未找到data键")
    
    # 提取流量数据（假设是第0个特征）
    flow_data = data['data'][:, :, 0]  # 取所有时间步、所有传感器的第0个特征
    
    # 创建时间索引（假设5分钟间隔，从2018-01-01开始）
    time_index = pd.date_range(
        start="2018-01-01", 
        periods=flow_data.shape[0], 
        freq="5min"
    )
    
    # 创建列名（传感器ID）
    sensor_ids = [f"sensor_{i}" for i in range(flow_data.shape[1])]
    
    # 构建DataFrame
    df = pd.DataFrame(flow_data, index=time_index, columns=sensor_ids)
    
    # 保存为CSV
    if output_csv:
        df.to_csv(output_csv)
        print(f"流量数据已保存至 {output_csv}")
    
    return df


def extract_flow_from_npz_08(npz_path, output_csv=None):
    """
    从PEMS08的npz文件中提取流量数据
    :param npz_path: .npz文件路径
    :param output_csv: 输出CSV路径（可选）
    :return: 流量DataFrame
    """
    # 加载npz文件
    data = np.load(npz_path)
    
    # 确认数据结构
    print("文件中包含的数组:", list(data.keys()))
    print("主数据形状:", data['data'].shape if 'data' in data else "未找到data键")
    
    # 提取流量数据（假设是第0个特征）
    flow_data = data['data'][:, :, 0]  # 取所有时间步、所有传感器的第0个特征
    
    # 创建时间索引（假设5分钟间隔，从2016-07-01开始）
    time_index = pd.date_range(
        start="2016-07-01", 
        periods=flow_data.shape[0], 
        freq="5min"
    )
    
    # 创建列名（传感器ID）
    sensor_ids = [f"sensor_{i}" for i in range(flow_data.shape[1])]
    
    # 构建DataFrame
    df = pd.DataFrame(flow_data, index=time_index, columns=sensor_ids)
    
    # 保存为CSV
    if output_csv:
        df.to_csv(output_csv)
        print(f"流量数据已保存至 {output_csv}")
    
    return df
# 使用示例
flow_df = extract_flow_from_npz_04("data/PEMS04/pems04.npz", "data/PEMS04/pems04_flow.csv")
# 打印前几行数据
print(flow_df.head())
flow_df = extract_flow_from_npz_08("data/PEMS08/pems08.npz", "data/PEMS08/pems08_flow.csv")
# 打印前几行数据
print(flow_df.head())