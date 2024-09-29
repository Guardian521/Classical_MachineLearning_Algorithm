import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 创建三个类别的数据点
num_points_per_class = 333
X_class0 = np.random.randn(num_points_per_class, 2) + np.array([0, 2])
X_class1 = np.random.randn(num_points_per_class, 2) + np.array([2, -2])
X_class2 = np.random.randn(num_points_per_class, 2) + np.array([-2, -2])

# 将类别标签分配给每个数据点
y_class0 = np.zeros(num_points_per_class)
y_class1 = np.ones(num_points_per_class)
y_class2 = 2 * np.ones(num_points_per_class)

# 合并数据点和标签
X = np.vstack([X_class0, X_class1, X_class2])
y = np.concatenate([y_class0, y_class1, y_class2])

# 可视化生成的数据集
plt.scatter(X_class0[:, 0], X_class0[:, 1], label='Class 0', marker='o')
plt.scatter(X_class1[:, 0], X_class1[:, 1], label='Class 1', marker='x')
plt.scatter(X_class2[:, 0], X_class2[:, 1], label='Class 2', marker='s')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Linearly Separable Multi-Class Dataset')
plt.show()
