import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence

# 加载数据
data = pd.read_csv('test.csv', header=None)

# 数据预处理
def preprocess_data(data):
    # 解析元组数据
    data[2] = data[2].apply(lambda x: eval(x))
    # 构建特征和标签
    features = []
    labels = []
    for _, row in data.iterrows():
        feature_row = []
        # 处理元组中的数据
        for key in row[2]:
            feature_row.extend(row[2][key])
        # 填充到49个元素
        padding_length = 49 - len(feature_row)
        padding_value = [400, 180, 0]
        feature_row.extend(padding_value * padding_length)
        features.append(feature_row)
        labels.append([row[0], row[1]])  # 构建标签为 [运动状态, 旋转状态]
    return features, labels

features, labels = preprocess_data(data)

# 标准化特征
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 转换数据为 PyTorch 张量
X = [torch.tensor(feature, dtype=torch.float32) for feature in features]
X_padded = pad_sequence(X, batch_first=True)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_padded, labels, test_size=0.2, random_state=42)

# 转换标签为 PyTorch 张量
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 定义 LSTM 模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 实例化模型
input_size = X_train.shape[2]  # 输入大小为填充后的特征的维度
hidden_size = 64
output_size = 2  # 运动状态和旋转状态的类别数
model = LSTM(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# 测试模型
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Test Accuracy: {accuracy}')
