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
columns_to_drop <- c("Distractibility", "Decreased.need.for.sleep", "Fatigue", "Difficulty.concentrating", "Concentration.issues", "Sleep.disturbance.1")
data <- data[, !names(data) %in% columns_to_drop]
colnames(data)
data <- data[, !names(data) %in% c("Intrusive.memories.or.flashback")]
colnames(data)
data$Avoidance_of_traumatic_event <- as.integer(data$Avoidance.of.reminders.of.traumatic.event | data$Avoidance.of.external.reminders.of.traumatic.event)
data <- data[, !names(data) %in% c("Avoidance.of.reminders.of.traumatic.event", "Avoidance.of.external.reminders.of.traumatic.event")]
colnames(data)
new_position <- 40
data <- data[, c(names(data)[1:new_position], "Avoidance_of_traumatic_event", names(data)[(new_position + 1):(ncol(data) - 1)])]

# 3. 定义要从聚类中排除的列
exclude_columns <- c("ID", "Bipolar.disorder", "Schizophrenia", "Depression", "Anxiety.disorder", "PTSD")

# 4. 提取要进行聚类的列
cluster_columns <- setdiff(names(data), exclude_columns)

# 5. 提取用于聚类的数据
data_to_cluster <- data[, cluster_columns]

# 6. 安装并加载'proxy'包，用于计算列之间的距离
if (!require(proxy)) {
  install.packages("proxy")
  library(proxy)
} else {
  library(proxy)
}

# 7. 使用vegdist计算列之间的Jaccard距离
# 转置数据，使得计算的是列之间的距离
distance_matrix <- dist(t(data_to_cluster), method = "phi")

# 8. 对列进行层次聚类
hc <- hclust(distance_matrix, method = "ward.D")
plot(hc, main = "Hierarchical Clustering of Variables", xlab = "Variables", ylab = "Distance")

# 10. 根据聚类结果获取列的顺序
clustered_columns_order <- hc$order
ordered_columns <- colnames(data_to_cluster)[clustered_columns_order]

# 11. 根据聚类结果重新排列数据框的列
# 将排除的列放在前面，聚类后的列按新顺序排列
data_reordered <- data[, c(exclude_columns, ordered_columns)]

# 12. 保存没有进行层次聚类的数据集（按原始列顺序）

# 13. 划分数据集
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

# 保存完整的数据集，包括聚类结果
write.csv(data_reordered, "Processed_Mentalillness.csv", row.names = FALSE)
# 保存训练集
write.csv(train_data, "Train_Mentalillness.csv", row.names = FALSE)
# 保存验证集
write.csv(val_data, "Validation_Mentalillness.csv", row.names = FALSE)
# 保存测试集
write.csv(test_data, "Test_Mentalillness.csv", row.names = FALSE)

# 13. 对每个症状列进行随机排序
data_randomized <- data
symptom_columns <- setdiff(names(data), exclude_columns) # 只选择症状列
# 对每一列进行随机排序
set.seed(456)
data_randomized[symptom_columns] <- lapply(data_randomized[symptom_columns], function(x) sample(x))
ordered_columns_randomized <- c(exclude_columns, colnames(data_randomized)[!(colnames(data_randomized) %in% exclude_columns)])
# 重新排序数据框
data_reordered_random <- data_randomized[, ordered_columns_randomized]
# 14. 对随机排序后的数据集进行相同的训练集、验证集、测试集划分
# 使用与原始数据集相同的行索引，确保划分一致
train_data_randomized <- data_reordered_random[train_indices, ]
val_data_randomized <- data_reordered_random[val_indices, ]
test_data_randomized <- data_reordered_random[test_indices, ]
colnames(data_reordered_random) <- gsub("[._]", " ", colnames(data_reordered_random))
colnames(train_data_randomized) <- colnames(data_reordered_random)
colnames(val_data_randomized) <- colnames(data_reordered_random)
colnames(test_data_randomized) <- colnames(data_reordered_random)
# 保存随机排序后的训练集、验证集、测试集
write.csv(train_data_randomized, "Train_Reordered_Mentalillness.csv", row.names = FALSE)
write.csv(val_data_randomized, "Validation_Reordered_Mentalillness.csv", row.names = FALSE)
write.csv(test_data_randomized, "Test_Reordered_Mentalillness.csv", row.names = FALSE)
