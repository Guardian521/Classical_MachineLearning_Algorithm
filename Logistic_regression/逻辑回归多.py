from dataset2 import X,y
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

def gradient_descent(X, y, num_classes, rate=0.01, epochs=1000):
    m, n = X.shape
    X = np.column_stack((np.ones((m, 1)), X))  # 添加偏置项
    y_one_hot = label_binarize(y, classes=range(num_classes))

    theta = np.zeros((n + 1, num_classes))  # 初始化参数

    for epoch in range(epochs):
        z = np.dot(X, theta)
        h = softmax(z)
        loss = cross_entropy(y_one_hot, h)

        gradient = np.dot(X.T, (h - y_one_hot)) / m
        theta -= rate * gradient

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return theta

def predict(X, theta):
    X = np.column_stack((np.ones((X.shape[0], 1)), X))
    predictions = softmax(np.dot(X, theta))
    return np.argmax(predictions, axis=1)

def boundary(X, y, theta):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                           np.arange(x2_min, x2_max, 0.1))

    Z = predict(np.c_[xx1.ravel(), xx2.ravel()], theta)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
# 使用梯度下降法
num_classes = len(np.unique(y))
theta = gradient_descent(X_train, y_train, num_classes, rate=0.01, epochs=1000)
# 在测试集上进行预测
predictions = predict(X_test, theta)
# 计算准确率
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy}')
# 绘制决策边界
boundary(X, y, theta)
