import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use('Agg')  # 设置为非交互式后端，适用于无图形界面环境
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle

# 加载数据
data = pd.read_csv('./output_folder/汇总/三合一(2).csv', encoding='gbk',
                   names=['每半小时时间', '城市', '区县', '订单量', '是否节假日', '天气', 'GDP', '常驻人口'],
                   skiprows=1,  # 跳过第一行（通常是列名）
                   parse_dates=['每半小时时间'])  # 将 '每半小时时间' 列解析为日期时间格式

# 提取时间特征
data['日期'] = data['每半小时时间'].dt.date  # 提取日期部分
data['小时'] = data['每半小时时间'].dt.hour  # 提取小时部分
data['分钟'] = data['每半小时时间'].dt.minute  # 提取分钟部分

# 循环特征编码
data['sin_hour'] = np.sin(2 * np.pi * data['小时'] / 23)  # 对小时进行正弦转换
data['cos_hour'] = np.cos(2 * np.pi * data['小时'] / 23)  # 对小时进行余弦转换
data['sin_minute'] = np.sin(2 * np.pi * data['分钟'] / 59)  # 对分钟进行正弦转换
data['cos_minute'] = np.cos(2 * np.pi * data['分钟'] / 59)  # 对分钟进行余弦转换

# 从日期中提取更多特征
data['星期几'] = data['每半小时时间'].dt.dayofweek  # 提取星期几（0-6，周一到周日）
data['月份'] = data['每半小时时间'].dt.month  # 提取月份（1-12）
data['季度'] = data['每半小时时间'].dt.quarter  # 提取季度（1-4）

# 删除原始时间戳列
data = data.drop(['每半小时时间', '小时', '分钟'], axis=1)  # 移除不再需要的时间戳列

# 处理类别特征（只保留天气）
categorical_features = ['天气']  # 只保留天气特征
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()  # 初始化LabelEncoder
    data[feature] = le.fit_transform(data[feature])  # 对分类变量进行标签编码
    label_encoders[feature] = le  # 保存LabelEncoder对象

# 标准化数值特征
numerical_features = ['GDP', '常驻人口']  # 定义数值变量列表
scaler = StandardScaler()  # 初始化StandardScaler
data[numerical_features] = scaler.fit_transform(data[numerical_features])  # 对数值变量进行标准化

# 标准化目标变量
target_scaler = StandardScaler()  # 初始化StandardScaler
data['订单量'] = target_scaler.fit_transform(data[['订单量']])  # 对目标变量进行标准化

# 删除日期列
data = data.drop('日期', axis=1)  # 移除日期列，因为已经从中提取了所有必要的信息

# 确保所有列都是数值类型
numeric_columns = [col for col in data.columns if col not in ['订单量', '城市', '区县']]  # 排除订单量、城市和区县列
data[numeric_columns] = data[numeric_columns].astype('float32')  # 将所有非目标变量列的数据类型转换为浮点型

# 分割数据集为训练集和验证集
X = data.drop(['订单量', '城市', '区县'], axis=1)  # 排除订单量、城市和区县列
y = data['订单量']  # 目标数据
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  # 按80:20比例分割数据集

# 转换为张量（Tensor）
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)  # 将训练特征转换为PyTorch张量
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # 将训练目标转换为PyTorch张量，并调整形状
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)  # 将验证特征转换为PyTorch张量
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)  # 将验证目标转换为PyTorch张量，并调整形状

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # 创建训练数据集
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)  # 创建验证数据集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)  # 创建训练数据加载器
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)  # 创建验证数据加载器

# Transformer模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # 输入嵌入层
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                       batch_first=True),  # Transformer编码器层
            num_layers=num_layers  # 编码器层数
        )
        self.fc = nn.Linear(d_model, 1)  # 全连接输出层
        self.initialize_weights()  # 初始化权重

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)  # 使用Xavier初始化方法对权重进行初始化
            elif 'bias' in name:
                nn.init.zeros_(param)  # 对偏置项初始化为零

    def forward(self, x):
        x = self.embedding(x)  # 通过输入嵌入层
        x = x.unsqueeze(1)  # 增加一个维度以匹配Transformer的输入要求
        x = self.transformer(x)  # 通过Transformer编码器
        x = x.squeeze(1)  # 移除多余的维度
        x = self.fc(x)  # 通过全连接输出层
        return x.squeeze()  # 返回预测结果，移除多余的维度

if __name__ == '__main__':
    # 使用 GPU 加速
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查是否有可用的GPU并设置设备

    # 初始化模型
    input_dim = X_train.shape[1]  # 更新输入维度
    d_model = 64  # Transformer模型的隐藏维度
    nhead = 4     # Transformer模型的多头注意力机制中的头数
    num_layers = 2  # Transformer编码器层数
    dim_feedforward = 128  # Transformer模型的前馈神经网络维度
    model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward).to(device)  # 实例化模型并将模型移动到指定设备

    # 损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # 使用Adam优化器

    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # 当验证损失不再减少时降低学习率

    # 训练模型并记录损失
    train_losses = []  # 保存训练损失
    val_losses = []  # 保存验证损失
    best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
    early_stopping_counter = 0  # 早停计数器
    patience = 50  # 早停耐心值
    num_epochs = 200  # 训练轮次数

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        epoch_train_loss = 0.0  # 初始化每轮次的训练损失
        for i, (inputs, labels) in enumerate(train_loader):  # 遍历训练数据加载器
            optimizer.zero_grad()  # 清空梯度
            inputs, labels = inputs.to(device), labels.to(device).squeeze()  # 将数据移动到指定设备并调整形状
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            epoch_train_loss += loss.item()  # 累加训练损失

        epoch_train_loss /= (i + 1)  # 计算平均训练损失
        train_losses.append(epoch_train_loss)  # 保存训练损失

        model.eval()  # 设置模型为评估模式
        val_loss = 0.0  # 初始化每轮次的验证损失
        with torch.no_grad():  # 关闭梯度计算
            for inputs, labels in val_loader:  # 遍历验证数据加载器
                inputs, labels = inputs.to(device), labels.to(device).squeeze()  # 将数据移动到指定设备并调整形状
                outputs = model(inputs)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失
                val_loss += loss.item() * inputs.size(0)  # 累加验证损失

        val_loss /= len(val_dataset)  # 计算平均验证损失
        val_losses.append(val_loss)  # 保存验证损失

        scheduler.step(val_loss)  # 根据验证损失更新学习率

        if val_loss < best_val_loss:  # 如果当前验证损失小于历史最佳验证损失
            best_val_loss = val_loss  # 更新最佳验证损失
            early_stopping_counter = 0  # 重置早停计数器
            torch.save(model.state_dict(), '无城市区县模型.pth')  # 保存当前模型状态
        else:
            early_stopping_counter += 1  # 增加早停计数器

        if early_stopping_counter >= patience:  # 如果早停计数器达到耐心值
            print(f"Early stopping after {epoch} epochs.")  # 打印早停信息
            break  # 提前终止训练

        if (epoch + 1) % 10 == 0:  # 每10个轮次打印一次训练和验证损失
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # 在训练完成后保存预处理对象
    with open('label_encoders.pkl', 'wb') as f:  # 只保存天气的LabelEncoder
        pickle.dump(label_encoders, f)

    with open('scaler.pkl', 'wb') as f:  # 保存StandardScaler对象
        pickle.dump(scaler, f)

    with open('target_scaler.pkl', 'wb') as f:  # 保存目标变量的StandardScaler对象
        pickle.dump(target_scaler, f)

        # 加载最佳模型
        model.load_state_dict(torch.load('无城市区县模型.pth', weights_only=True))

        # 获取预测值
        model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy().flatten())

        # 反标准化预测值
        predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()