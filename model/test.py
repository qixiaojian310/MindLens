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


def preprocess_input(input_data, symptom_columns):
    """
    将输入的症状数组转换为固定长度的症状向量。
    - input_data: 用户输入的症状数组 [{"name": str, "value": bool}]
    - symptom_columns: 预定义的45个症状名称列表
    """
    # 初始化症状向量，默认值为0
    symptom_vector = np.zeros(len(symptom_columns), dtype=np.float32)

    # 将输入数据映射到症状向量
    for symptom in input_data:
        if symptom["name"] in symptom_columns:
            idx = symptom_columns.index(symptom["name"])
            symptom_vector[idx] = 1.0 if symptom["value"] else 0.0

    # 转换为 PyTorch Tensor，并调整维度以匹配模型输入
    return torch.tensor(symptom_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(2)


def predict(input_data, model, symptom_columns):
    # 预处理输入数据
    input_tensor = preprocess_input(input_data, symptom_columns)

    # 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = (outputs > 0.5).float().numpy()  # 0.5 阈值

    # 返回预测结果
    return {
        disease: bool(predictions[0][i]) for i, disease in enumerate(mental_diseases)
    }


if True:
    input_data = [
        {"name": "Symptom1", "value": True},
        {"name": "Symptom5", "value": False},
        {"name": "Symptom10", "value": True},
    ]

    symptom_columns = [
        "Symptom1",
        "Symptom2",
        "Symptom3",
        "Symptom4",
        "Symptom5",
        "Symptom6",
        "Symptom7",
        "Symptom8",
        "Symptom9",
        "Symptom10",
        "Symptom11",
        "Symptom12",
        "Symptom13",
        "Symptom14",
        "Symptom15",
        "Symptom16",
        "Symptom17",
        "Symptom18",
        "Symptom19",
        "Symptom20",
        "Symptom21",
        "Symptom22",
        "Symptom23",
        "Symptom24",
        "Symptom25",
        "Symptom26",
        "Symptom27",
        "Symptom28",
        "Symptom29",
        "Symptom30",
        "Symptom31",
        "Symptom32",
        "Symptom33",
        "Symptom34",
        "Symptom35",
        "Symptom36",
        "Symptom37",
        "Symptom38",
        "Symptom39",
        "Symptom40",
        "Symptom41",
        "Symptom42",
        "Symptom43",
        "Symptom44",
        "Symptom45",
    ]

    # 加载模型
    model = MultiLabelCNN(
        input_size=1,
        output_size=5,
        num_features=45,
        level=2,
    )
    model.load_state_dict(torch.load("cnn_multi_target_prediction_model.bin"))
    model.eval()

    # 获取预测结果
    predicted_results = predict(input_data, model, symptom_columns)

    # 输出预测结果
    print("Prediction Results:")
    for disease, predicted in predicted_results.items():
        print(f"{disease}: {'Yes' if predicted else 'No'}")
