import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np

# # 读取并预处理数据
# train_df = pd.read_csv("Train_Reordered_Mentalillness.csv")
# val_df = pd.read_csv("Validation_Reordered_Mentalillness.csv")
# test_df = pd.read_csv("Test_Reordered_Mentalillness.csv")

# 读取并预处理数据
train_df = pd.read_csv("Train_Mentalillness.csv")
val_df = pd.read_csv("Validation_Mentalillness.csv")
test_df = pd.read_csv("Test_Mentalillness.csv")

# 排除 ID 列
train_df = train_df.drop(columns=["ID"])
val_df = val_df.drop(columns=["ID"])
test_df = test_df.drop(columns=["ID"])

# 定义心理疾病列和症状列
mental_diseases = [
    "PTSD",
    "Bipolar disorder",
    "Depression",
    "Anxiety disorder",
    "Schizophrenia",
]
symptom_columns = [col for col in train_df.columns if col not in mental_diseases]

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[symptom_columns])
X_val = scaler.transform(val_df[symptom_columns])
X_test = scaler.transform(test_df[symptom_columns])

# 将心理疾病列作为标签（多标签分类）
y_train = train_df[mental_diseases].values
y_val = val_df[mental_diseases].values
y_test = test_df[mental_diseases].values

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(2)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义多标签分类 CNN 模型
class MultiLabelCNN(nn.Module):
    def __init__(self, input_size, output_size, num_features, level=1):
        super(MultiLabelCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_features, level * 32, kernel_size=2, padding=1)
        self.conv1_bn = nn.BatchNorm1d(level * 32)
        self.conv2 = nn.Conv1d(level * 32, level * 32 * 2, kernel_size=2, padding=1)
        self.conv2_bn = nn.BatchNorm1d(level * 32 * 2)
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.relu = nn.Softplus()
        self.dropout = nn.Dropout(0.6)

        # 计算卷积输出大小
        conv_output_size = self.calculate_conv_output_size(input_size)

        # 全连接层
        self.fc1 = nn.Linear(level * 32 * 2 * conv_output_size, 512)
        self.fc2 = nn.Linear(512, output_size)

    def calculate_conv_output_size(self, input_size):
        size = input_size
        for layer in [self.conv1, self.conv2]:
            size = (
                size + 1 * 2 - layer.kernel_size[0]
            ) // 1 + 1  # padding=1, kernel_size=5, stride=1
            size = size // 2  # Pooling with kernel_size=2, stride=2
        return size

    def forward(self, x):
        x = self.pool(self.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(self.relu(self.conv2_bn(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # 用于多标签分类
        return x


# 初始化模型
input_size = X_train_tensor.shape[2]
output_size = len(mental_diseases)
model = MultiLabelCNN(
    input_size=input_size,
    output_size=output_size,
    num_features=len(symptom_columns),
    level=2,
)


# 训练模型
# 损失函数和优化器
criterion = nn.BCELoss()  # 二进制交叉熵损失，用于多标签分类
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=2, factor=0.5
)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=30,
    patience=6,
):
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}"
        )

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break


# 训练和验证模型
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)


# 测试模型，输出每个类别的 F1 Score 和准确率
def test_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions = (outputs > 0.5).float()  # 使用阈值0.5将概率转换为0或1
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算每个疾病的 F1 Score 和准确率
    f1_scores = f1_score(y_true, y_pred, average=None)
    accuracies = [
        accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])
    ]

    # 输出每个疾病的 F1 Score 和准确率
    for i, disease in enumerate(mental_diseases):
        print(
            f"{disease} - F1 Score: {f1_scores[i]:.4f}, Accuracy: {accuracies[i]:.4f}"
        )

    # 输出平均 F1 Score 和平均准确率
    print(f"Average F1 Score: {f1_scores.mean():.4f}")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")


# 在测试集上评估模型
test_model(model, test_loader)

# def preprocess_input(input_data, symptom_columns):
#     """
#     将输入的症状数组转换为固定长度的症状向量。
#     - input_data: 用户输入的症状数组 [{"name": str, "value": bool}]
#     - symptom_columns: 预定义的45个症状名称列表
#     """
#     # 初始化症状向量，默认值为0
#     symptom_vector = np.zeros(len(symptom_columns), dtype=np.float32)

#     # 将输入数据映射到症状向量
#     for symptom in input_data:
#         if symptom["name"] in symptom_columns:
#             idx = symptom_columns.index(symptom["name"])
#             symptom_vector[idx] = 1.0 if symptom["value"] else 0.0

#     # 转换为 PyTorch Tensor，并调整维度以匹配模型输入
#     return torch.tensor(symptom_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(2)


# def predict(input_data, model, symptom_columns):
#     # 预处理输入数据
#     input_tensor = preprocess_input(input_data, symptom_columns)

#     # 预测
#     with torch.no_grad():
#         outputs = model(input_tensor)
#         predictions = (outputs > 0.5).float().numpy()  # 0.5 阈值

#     # 返回预测结果
#     return {
#         disease: bool(predictions[0][i]) for i, disease in enumerate(mental_diseases)
#     }


# if True:
#     input_data = [
#         {"name": "Symptom1", "value": True},
#         {"name": "Symptom5", "value": False},
#         {"name": "Symptom10", "value": True},
#     ]

#     symptom_columns = [
#         "Symptom1",
#         "Symptom2",
#         "Symptom3",
#         "Symptom4",
#         "Symptom5",
#         "Symptom6",
#         "Symptom7",
#         "Symptom8",
#         "Symptom9",
#         "Symptom10",
#         "Symptom11",
#         "Symptom12",
#         "Symptom13",
#         "Symptom14",
#         "Symptom15",
#         "Symptom16",
#         "Symptom17",
#         "Symptom18",
#         "Symptom19",
#         "Symptom20",
#         "Symptom21",
#         "Symptom22",
#         "Symptom23",
#         "Symptom24",
#         "Symptom25",
#         "Symptom26",
#         "Symptom27",
#         "Symptom28",
#         "Symptom29",
#         "Symptom30",
#         "Symptom31",
#         "Symptom32",
#         "Symptom33",
#         "Symptom34",
#         "Symptom35",
#         "Symptom36",
#         "Symptom37",
#         "Symptom38",
#         "Symptom39",
#         "Symptom40",
#         "Symptom41",
#         "Symptom42",
#         "Symptom43",
#         "Symptom44",
#         "Symptom45",
#     ]

#     # 加载模型
#     model = MultiLabelCNN(
#         input_size=1,
#         output_size=5,
#         num_features=45,
#         level=2,
#     )
#     model.load_state_dict(torch.load("cnn_multi_target_prediction_model.bin"))
#     model.eval()

#     # 获取预测结果
#     predicted_results = predict(input_data, model, symptom_columns)

#     # 输出预测结果
#     print("Prediction Results:")
#     for disease, predicted in predicted_results.items():
#         print(f"{disease}: {'Yes' if predicted else 'No'}")


# if False:
#     model = torch.load("cnn_multi_target_prediction_model.pth")
#     # 使用 TorchScript 脚本化模型
#     scripted_model = torch.jit.script(model)

#     # 保存脚本化模型
#     scripted_model.save("cnn_model_scripted.pt")


# torch.save(model.state_dict(), "cnn_multi_target_prediction_model.bin")