import json
import joblib
import torch
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
import numpy as np
from torch import nn
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import uuid

app = FastAPI()

prefix = "api"
# 配置允许跨域的源
origins = [
    "http://localhost:3333",  # 允许前端 React 应用在这个地址进行跨域请求
]

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许的源
    allow_credentials=True,  # 是否允许携带 Cookie
    allow_methods=["GET", "POST"],  # 允许的 HTTP 方法
    allow_headers=["*"],  # 允许的 HTTP 头部
)

# 定义心理疾病列
mental_diseases = [
    "Bipolar disorder",
    "Schizophrenia",
    "Depression",
    "Anxiety disorder",
    "PTSD",
]

mental_symptom = [
    "The disturbance is not substance induced",
    "Increase in goal directed activity",
    "Restlessness",
    "X6 month duration",
    "Dissociative reaction",
    "Psychomotor agitation",
    "Hypervigilance",
    "The disturbance causes clinically significant distress",
    "Loss of interest or pleasure in activities",
    "Lack of sleep or oversleeping",
    "Intrusive memories or flashbacks",
    "Experiencing traumatic event",
    "Persistent sadness or low mood",
    "Witnessing traumatic event",
    "Hallucinations",
    "Exaggerated startle response",
    "Depressed mood",
    "Irritability",
    "More talkative than usual",
    "Angry outburst",
    "X1 month duration",
    "Feeling of detachment",
    "Diminished interest",
    "Fatigue or loss of energy",
    "More than one month of disturbance",
    "Racing thoughts",
    "Persistent negative emotional state",
    "Excessive involvement in activities with high potential for painful consequences",
    "Diminished emotional expression",
    "Catatonic behavior",
    "Recurrent distressing dreaming affiliated with the traumatic event",
    "Recklessness",
    "Intense distress or reaction when exposed to cues affiliated with the traumatic event",
    "Persistent inability to experience positive emotions",
    "Sleep disturbance",
    "Persistent and exaggerated negative belief about oneself or the world",
    "Delusions",
    "Inflated self esteem",
    "Disorganized thinking or speech",
    "Excessive worry or fear",
    "Persistent loss of memory about the cause or consequences of the traumatic event",
    "Difficulty concentrating or making decisions",
    "Weight loss or gain",
    "Thoughts of suicide",
    "Avoidance of traumatic event",
]


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


class MultiLabelFFNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiLabelFFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))  # 用于多标签分类
        return x


# 加载模型
cnn_model_hierarchy = MultiLabelCNN(
    input_size=1,
    output_size=len(mental_diseases),
    num_features=len(mental_symptom),  # 特征数量
    level=2,
)
cnn_model_random = MultiLabelCNN(
    input_size=1,
    output_size=len(mental_diseases),
    num_features=len(mental_symptom),  # 特征数量
    level=2,
)
ffnn_model = MultiLabelFFNN(
    input_size=len(mental_symptom), output_size=len(mental_diseases)
)
cnn_model_hierarchy.load_state_dict(
    torch.load("model/cnn_multi_target_prediction_model_hierarchy.bin")
)
cnn_model_random.load_state_dict(
    torch.load("model/cnn_multi_target_prediction_model_random.bin")
)
ffnn_model.load_state_dict(torch.load("model/ffnn_multi_target_prediction_model.bin"))
xgb_model_data = joblib.load("model/xgb_multi_target_prediction_model.bin")
xgb_model = xgb_model_data["model"]
mental_diseases1 = xgb_model_data["diseases"]
print(mental_diseases1)
cnn_model_hierarchy.eval()
cnn_model_random.eval()
ffnn_model.eval()


# 定义请求体
class Feature(BaseModel):
    name: str
    value: bool


class Request(BaseModel):
    features: List[Feature]
    worker_choice: str  # 'cnn' 或 'xgb'


# 修改结果模型为包含概率
class ResultRaw(BaseModel):
    BipolarDisorder: float
    Schizophrenia: float
    Depression: float
    AnxietyDisorder: float
    PTSD: float


# 修改结果模型为包含概率
class Result(BaseModel):
    result: ResultRaw
    key: str
    name: str
    model: str


def preprocess_input_cnn(input_data):
    # symptom_columns = [f"Symptom{i + 1}" for i in range(len(mental_symptom))]
    symptom_vector = np.zeros(len(mental_symptom), dtype=np.float32)

    for i in range(len(input_data)):
        symptom_vector[i] = 1.0 if input_data[i].value else 0.0

    return torch.tensor(symptom_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(2)


# 创建预测函数
def preprocess_input_ffnn(input_data):
    # symptom_columns = [f"Symptom{i + 1}" for i in range(len(mental_symptom))]
    symptom_vector = np.zeros(len(mental_symptom), dtype=np.float32)

    for i in range(len(input_data)):
        symptom_vector[i] = 1.0 if input_data[i].value else 0.0
    return torch.tensor(symptom_vector, dtype=torch.float32).unsqueeze(0)


# 假设我们将症状数据转换为模型可以理解的格式
def prepare_features(request: Request):
    feature_vector = [int(feature.value) for feature in request.features]
    return np.array(feature_vector).reshape(1, -1)


@app.post(f"/{prefix}/prediction", response_model=List[Result])
def get_prediction(req: Request):
    input_tensor_cnn = preprocess_input_cnn(req.features)
    input_tensor_ffnn = preprocess_input_ffnn(req.features)
    xgb_x_input = prepare_features(req)
    xgb_probabilities = [
        prob[0, 0] for prob in xgb_model.predict_proba(xgb_x_input)
    ]  # 假设模型返回概率列表
    outputs = {"cnn_hierarchy": [], "cnn_random": [], "ffnn": [], "xgb": []}
    with torch.no_grad():
        outputs["cnn_hierarchy"] = cnn_model_hierarchy(input_tensor_cnn)
        outputs["cnn_random"] = cnn_model_random(input_tensor_cnn)
        outputs["ffnn"] = ffnn_model(input_tensor_ffnn)

    # 获取每个疾病的概率
    probabilities_cnn_hierarchy = outputs["cnn_hierarchy"].numpy()[0]
    probabilities_cnn_random = outputs["cnn_random"].numpy()[0]
    probabilities_ffnn = outputs["ffnn"].numpy()[0]

    prediction = [
        {
            "key": str(uuid.uuid4()),
            "model": "cnn_hierarchy",
            "name": "Hierarchy Cluster Features CNN",
            "result": {
                "BipolarDisorder": round(
                    float(probabilities_cnn_hierarchy[0]) * 100, 2
                ),
                "Schizophrenia": round(float(probabilities_cnn_hierarchy[1]) * 100, 2),
                "Depression": round(float(probabilities_cnn_hierarchy[2]) * 100, 2),
                "AnxietyDisorder": round(
                    float(probabilities_cnn_hierarchy[3]) * 100, 2
                ),
                "PTSD": round(float(probabilities_cnn_hierarchy[4]) * 100, 2),
            },
        },
        {
            "key": str(uuid.uuid4()),
            "model": "cnn_random",
            "name": "Random Features CNN",
            "result": {
                "BipolarDisorder": round(float(probabilities_cnn_random[0]) * 100, 2),
                "Schizophrenia": round(float(probabilities_cnn_random[1]) * 100, 2),
                "Depression": round(float(probabilities_cnn_random[2]) * 100, 2),
                "AnxietyDisorder": round(float(probabilities_cnn_random[3]) * 100, 2),
                "PTSD": round(float(probabilities_cnn_random[4]) * 100, 2),
            },
        },
        {
            "key": str(uuid.uuid4()),
            "model": "ffnn",
            "name": "FFNN",
            "result": {
                "BipolarDisorder": round(float(probabilities_ffnn[0]) * 100, 2),
                "Schizophrenia": round(float(probabilities_ffnn[1]) * 100, 2),
                "Depression": round(float(probabilities_ffnn[2]) * 100, 2),
                "AnxietyDisorder": round(float(probabilities_ffnn[3]) * 100, 2),
                "PTSD": round(float(probabilities_ffnn[4]) * 100, 2),
            },
        },
        {
            "key": str(uuid.uuid4()),
            "model": "xgb",
            "name": "XGBoost",
            "result": {
                "BipolarDisorder": round(float(xgb_probabilities[0]) * 100, 2),
                "Schizophrenia": round(float(xgb_probabilities[4]) * 100, 2),
                "Depression": round(float(xgb_probabilities[2]) * 100, 2),
                "AnxietyDisorder": round(float(xgb_probabilities[3]) * 100, 2),
                "PTSD": round(float(xgb_probabilities[1]) * 100, 2),
            },
        },
    ]

    # 将结果保存到文件
    with open("predictions.json", "a") as json_file:
        json.dump(prediction, json_file)
        json_file.write("\n")  # 每个预测结果换行存储

    return prediction


@app.get(f"/{prefix}/symptom", response_model=List)
def get_symptom():
    return mental_symptom


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
