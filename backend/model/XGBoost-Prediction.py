import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

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
    "Bipolar disorder",
    "Schizophrenia",
    "Depression",
    "Anxiety disorder",
    "PTSD",
]
symptom_columns = [col for col in train_df.columns if col not in mental_diseases]

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[symptom_columns])
X_val = scaler.transform(val_df[symptom_columns])
X_test = scaler.transform(test_df[symptom_columns])

# 将心理疾病列作为标签（多标签分类）
y_train = train_df[mental_diseases].values
y_val = val_df[mental_diseases].values
y_test = test_df[mental_diseases].values

# 使用 XGBoost 进行多任务分类
model = XGBClassifier(
    max_depth=8,  # 控制树的最大深度
    learning_rate=0.5,  # 较低的学习率，防止过拟合
    n_estimators=4000,  # 较多的树数量
    reg_alpha=0.2,  # L1正则化
    reg_lambda=0.4,  # L2正则化
    eval_metric="logloss",  # 使用对数损失作为评估标准
)

multi_target_model = MultiOutputClassifier(model)
multi_target_model.fit(X_train, y_train)

y_val_pred = multi_target_model.predict(X_val)
y_test_pred = multi_target_model.predict(X_test)


# 计算 F1 Score 和准确率
def evaluate_model(y_true, y_pred, diseases):
    f1_scores = f1_score(y_true, y_pred, average=None)
    accuracies = [
        accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])
    ]

    # 输出每个疾病的 F1 Score 和准确率
    for i, disease in enumerate(diseases):
        print(
            f"{disease} - F1 Score: {f1_scores[i]:.6f}, Accuracy: {accuracies[i]:.6f}"
        )

    # 输出平均 F1 Score 和平均准确率
    print(f"Average F1 Score: {f1_scores.mean():.6f}")
    print(f"Average Accuracy: {np.mean(accuracies):.6f}")


# 验证集结果
print("Validation Results:")
evaluate_model(y_val, y_val_pred, mental_diseases)

# 测试集结果
print("\nTest Results:")
evaluate_model(y_test, y_test_pred, mental_diseases)

joblib.dump(
    {"model": multi_target_model, "diseases": mental_diseases},
    "xgb_multi_target_prediction_model.bin",
)
