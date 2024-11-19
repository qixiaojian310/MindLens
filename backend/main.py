import json
import torch
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
import numpy as np
from torch import nn
from typing import List
from fastapi.middleware.cors import CORSMiddleware

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
    "PTSD",
    "Bipolar disorder",
    "Depression",
    "Anxiety disorder",
    "Schizophrenia",
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
            size = (size + 1 * 2 - layer.kernel_size[0]) // 1 + 1  # padding=1
            size = size // 2  # Pooling
        return size

    def forward(self, x):
        x = self.pool(self.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(self.relu(self.conv2_bn(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # 多标签分类
        return x


# 加载模型
model = MultiLabelCNN(
    input_size=1,
    output_size=len(mental_diseases),
    num_features=len(mental_symptom),  # 特征数量
    level=2,
)
model.load_state_dict(torch.load("model/cnn_multi_target_prediction_model.bin"))
model.eval()


# 定义请求体
class Feature(BaseModel):
    name: str
    value: bool


class Request(BaseModel):
    features: List[Feature]
    worker_choice: str  # 'cnn' 或 'xgb'


# 修改结果模型为包含概率
class Result(BaseModel):
    BipolarDisorder: float
    Depression: float
    AnxietyDisorder: float
    Schizophrenia: float
    PTSD: float


def preprocess_input(input_data):
    symptom_columns = [f"Symptom{i + 1}" for i in range(len(mental_symptom))]
    symptom_vector = np.zeros(len(symptom_columns), dtype=np.float32)

    for symptom in input_data:
        if symptom.name in symptom_columns:
            idx = symptom_columns.index(symptom.name)
            symptom_vector[idx] = 1.0 if symptom.value else 0.0

    return torch.tensor(symptom_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(2)


@app.post(f"/{prefix}/prediction", response_model=Result)
def get_prediction(req: Request):
    input_tensor = preprocess_input(req.features)

    with torch.no_grad():
        outputs = model(input_tensor)

    # 获取每个疾病的概率
    probabilities = outputs.numpy()[0]

    prediction = {
        "BipolarDisorder": float(probabilities[1]),
        "Depression": float(probabilities[2]),
        "AnxietyDisorder": float(probabilities[3]),
        "Schizophrenia": float(probabilities[4]),
        "PTSD": float(probabilities[0]),
    }

    # 将结果保存到文件
    with open("predictions.json", "a") as json_file:
        json.dump(prediction, json_file)
        json_file.write("\n")  # 每个预测结果换行存储

    return Result(**prediction)


@app.get(f"/{prefix}/symptom", response_model=List)
def get_symptom():
    return mental_symptom


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
