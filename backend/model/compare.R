# 1. 读取数据，保留原始列名
setwd("E:/HKU/7103data-mining/data-mining/project/backend/model/")

data <- read.csv("Mentalillness.csv")
print("Structure of data")
str(data)
head(data)
colSums(is.na(data))
rowSums(is.na(data))
duplicated_rows <- data[duplicated(data), ]
sum(duplicated(data))
colnames(data)

# 2. 定义要从聚类中排除的列（五种症状列）
exclude_columns <- c("ID", "Bipolar.disorder", "Schizophrenia", "Depression", "Anxiety.disorder", "PTSD")

# 3. 将五种症状列放到数据框的前面
# 通过指定这些列的位置，将它们放到数据框的最前面
data_reordered <- data[, c(exclude_columns, setdiff(names(data), exclude_columns))]

# 4. 保存原始数据集的重新排序结果
write.csv(data_reordered, "Reordered_Mentalillness.csv", row.names = FALSE)

# 5. 划分数据集
set.seed(123)
# 定义划分比例
train_ratio <- 0.8
val_ratio <- 0.05
test_ratio <- 0.15

# 确保比例之和为1
if ((train_ratio + val_ratio + test_ratio) != 1) {
  stop("训练集、验证集和测试集的比例之和必须为1。")
}
n_samples <- nrow(data_reordered)
# 随机打乱数据的索引
indices <- sample(1:n_samples)

# 计算各个数据集的样本数量
train_end <- floor(train_ratio * n_samples)
val_end <- floor((train_ratio + val_ratio) * n_samples)
test_end <- n_samples # 最后一个索引

# 获取各个数据集的索引
train_indices <- indices[1:train_end]
val_indices <- indices[(train_end + 1):val_end]
test_indices <- indices[(val_end + 1):test_end]

# 划分数据集
train_data <- data_reordered[train_indices, ]
val_data <- data_reordered[val_indices, ]
test_data <- data_reordered[test_indices, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Validation set size:", nrow(val_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

# 替换列名中的点号和下划线为空格（如果尚未替换）
colnames(data_reordered) <- gsub("[._]", " ", colnames(data_reordered))
colnames(train_data) <- colnames(data_reordered)
colnames(val_data) <- colnames(data_reordered)
colnames(test_data) <- colnames(data_reordered)

# 保存重新排序后的数据集，包括训练集、验证集和测试集
write.csv(train_data, "Train_Reordered_Mentalillness.csv", row.names = FALSE)
write.csv(val_data, "Validation_Reordered_Mentalillness.csv", row.names = FALSE)
write.csv(test_data, "Test_Reordered_Mentalillness.csv", row.names = FALSE)


# 13. 对每个症状列进行随机排序
data_randomized <- data
symptom_columns <- setdiff(names(data), exclude_columns) # 只选择症状列

# 对每一列进行随机排序
set.seed(456)
data_randomized[symptom_columns] <- lapply(data_randomized[symptom_columns], function(x) sample(x))

# 保存随机排序后的数据
colnames(data_randomized) <- gsub("[._]", " ", colnames(data_reordered))

# 14. 对随机排序后的数据集进行相同的训练集、验证集、测试集划分
# 使用与原始数据集相同的行索引，确保划分一致
train_data_randomized <- data_randomized[train_indices, ]
val_data_randomized <- data_randomized[val_indices, ]
test_data_randomized <- data_randomized[test_indices, ]

# 保存随机排序后的训练集、验证集、测试集
write.csv(train_data_randomized, "Train_Randomized_Mentalillness.csv", row.names = FALSE)
write.csv(val_data_randomized, "Validation_Randomized_Mentalillness.csv", row.names = FALSE)
write.csv(test_data_randomized, "Test_Randomized_Mentalillness.csv", row.names = FALSE)
