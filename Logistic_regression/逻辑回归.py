import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dataset1 import X,y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def entropy(y_tr, y_pr):
    epsilon = 1e-15
    y_pr = np.clip(y_pr, epsilon, 1 - epsilon)
    return - (y_tr * np.log(y_pr) + (1 - y_tr) * np.log(1 - y_pr)).mean()

def gradient_descent(X, y, rate=0.01, epochs=1000):
    m, n = X.shape
    X = np.column_stack((np.ones((m, 1)), X))  # 偏置项

    theta = np.zeros(n + 1)  # 初始化

    for epoch in range(epochs):
        z = np.dot(X, theta)
        h = sigmoid(z)
        loss = entropy(y, h)

        gradient = np.dot(X.T, (h - y)) / m
        theta -= rate * gradient

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return theta

def predict(X, theta):
    X = np.column_stack((np.ones((X.shape[0], 1)), X))
    predictions = sigmoid(np.dot(X, theta))
    return (predictions >= 0.5).astype(int)
def boundary(X, y, theta):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                           np.arange(x2_min, x2_max, 0.1))

    Z = predict(np.c_[xx1.ravel(), xx2.ravel()], theta)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', marker='o')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', marker='x')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
# 使用梯度下降法
theta = gradient_descent(X_train, y_train, rate=0.01, epochs=1000)
# 在测试集上进行预测
predictions = predict(X_test, theta)
# 计算准确率
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy}')
# 绘制决策边界
boundary(X,y,theta)