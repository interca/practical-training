import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 设定设备（CPU或GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预处理对象
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)

# 重新定义TransformerModel类
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x.squeeze()

# 定义模型结构，与训练时保持一致
input_dim = 11  # 由于去掉了两个类别特征，输入维度减少到11
d_model = 64
nhead = 4
num_layers = 2
dim_feedforward = 128
model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward).to(device)

# 加载保存的模型权重
model.load_state_dict(torch.load('无城市区县模型.pth', map_location=device))

# 设置模型为评估模式
model.eval()

# 假设我们有一个新的数据点来进行预测
new_data_example = {
    '订单量': [0],  # 目标变量，这里不需要实际值
    '是否节假日': [0],
    '天气': ['Clear'],
    'GDP': [2087],
    '常驻人口': [119],
    '每半小时时间': ['2024-12-23 10:30:00']
}

# 将新数据点转换成DataFrame
new_data_df = pd.DataFrame(new_data_example)

# 对新数据进行与训练数据相同的预处理步骤
# 提取时间特征
new_data_df['每半小时时间'] = pd.to_datetime(new_data_df['每半小时时间'])
new_data_df['小时'] = new_data_df['每半小时时间'].dt.hour
new_data_df['分钟'] = new_data_df['每半小时时间'].dt.minute

new_data_df['sin_hour'] = np.sin(2 * np.pi * new_data_df['小时'] / 23)
new_data_df['cos_hour'] = np.cos(2 * np.pi * new_data_df['小时'] / 23)
new_data_df['sin_minute'] = np.sin(2 * np.pi * new_data_df['分钟'] / 59)
new_data_df['cos_minute'] = np.cos(2 * np.pi * new_data_df['分钟'] / 59)

new_data_df['星期几'] = new_data_df['每半小时时间'].dt.dayofweek
new_data_df['月份'] = new_data_df['每半小时时间'].dt.month
new_data_df['季度'] = new_data_df['每半小时时间'].dt.quarter

# 删除原始时间戳列和城市区县两列
new_data_df = new_data_df.drop(['每半小时时间', '小时', '分钟'], axis=1)

# 处理类别特征（仅保留天气）
categorical_features = ['天气']

# 添加未知标签处理逻辑
def handle_unknown_labels(encoder, df, feature):
    known_labels = encoder.classes_
    unknown_labels = df[feature].unique()

    for label in unknown_labels:
        if label not in known_labels:
            df.loc[df[feature] == label, feature] = 'unknown'

    df[feature] = encoder.transform(df[feature]).astype(int)  # 明确转换为整数类型

for feature in categorical_features:
    handle_unknown_labels(label_encoders[feature], new_data_df, feature)

# 标准化数值特征
numerical_features = ['GDP', '常驻人口']
new_data_df[numerical_features] = scaler.transform(new_data_df[numerical_features])

# 确保所有类别特征也是整数类型
for col in categorical_features:
    new_data_df[col] = new_data_df[col].astype('int64')  # 再次确认转换为整数类型

# 将所有列转换为浮点类型
numeric_columns = [col for col in new_data_df.columns if col not in ['订单量']]

# 确保所有数值列都是浮点类型
new_data_df[numeric_columns] = new_data_df[numeric_columns].astype('float32')

# 打印DataFrame的dtype以检查
print("DataFrame dtypes:")
print(new_data_df.dtypes)

# 检查所有列是否都是数值类型
assert all(new_data_df.dtypes.apply(lambda x: x in [np.float32, np.int64])), "DataFrame contains non-numeric columns"

# 准备输入数据
X_new_tensor = torch.tensor(new_data_df.drop('订单量', axis=1).values, dtype=torch.float32).to(device)

# 进行预测
with torch.no_grad():
    prediction = model(X_new_tensor)
    predicted_order_volume = prediction.cpu().numpy().flatten()

# 反标准化预测值
predicted_order_volume = target_scaler.inverse_transform(np.array(predicted_order_volume).reshape(-1, 1)).flatten()

print(predicted_order_volume)