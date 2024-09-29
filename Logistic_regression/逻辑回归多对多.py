import numpy as np
from dataset2 import X, y
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

def gradient_descent(X, y, classes_combinations, rate=0.01, epochs=1000):
    m, n = X.shape
    X = np.column_stack((np.ones((m, 1)), X))

    num_classes = len(np.unique(y))
    theta = np.zeros((n + 1, num_classes))

    for epoch in range(epochs):
        all_h = softmax(np.dot(X, theta))

        for i, (class1, class2) in enumerate(classes_combinations):
            X_binary = X[(y == class1) | (y == class2)]
            y_binary = (y[(y == class1) | (y == class2)] == class1).astype(int)

            z = np.dot(X_binary, theta[:, i:i+1])
            h = softmax(z)
            all_h[:len(X_binary), i] = h[:, 0]

            loss = cross_entropy(y_binary, h)
            gradient = np.dot(X_binary.T, (h - y_binary)) / m
            theta[:, i] -= rate * gradient.sum(axis=1) / len(X_binary)


        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return theta

def predict(X, theta, classes_combinations):
    X = np.column_stack((np.ones((X.shape[0], 1)), X))
    all_predictions = softmax(np.dot(X, theta))

    return np.argmax(all_predictions, axis=1)

def boundary(X, y, theta, classes_combinations):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                           np.arange(x2_min, x2_max, 0.1))

    Z = predict(np.c_[xx1.ravel(), xx2.ravel()], theta, classes_combinations)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

classes_combinations = [(0, 1), (0, 2), (1, 2)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

theta = gradient_descent(X_train, y_train, classes_combinations, rate=0.01, epochs=1000)

predictions = predict(X_test, theta, classes_combinations)

accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy}')

boundary(X, y, theta, classes_combinations)
