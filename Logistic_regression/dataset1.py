import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(50)

# 创建两个均值分别为 [2, 2] 和 [-2, -2] 的正态分布
class_1 = np.random.randn(500, 2) + [2, 2]
class_2 = np.random.randn(500, 2) + [-2, -2]

# 将类别标签分别设置为0和1
labels_class_1 = np.zeros(500)
labels_class_2 = np.ones(500)

# 合并两个类别的数据和标签
data = np.vstack([class_1, class_2])
labels = np.concatenate([labels_class_1, labels_class_2])

dataset = np.column_stack((data, labels))

np.random.shuffle(dataset)

X = dataset[:, :2]
y = dataset[:, 2]

# 绘制数据集
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', marker='o')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
