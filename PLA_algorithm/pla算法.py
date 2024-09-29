import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
def pla(X,y,max_iter=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    iterations = 0
    while iterations < max_iter:
        mistakes = 0
        for i in range(n_samples):
            if y[i] * (np.dot(w, X[i]) + b) <= 0:
                w += y[i] * X[i]
                b += y[i]
                mistakes += 1
        if mistakes == 0:
            break
        iterations += 1
    return w, b



data = pd.read_csv('乳腺癌.csv', skiprows=1)
data1=data.iloc[:,0:29]
x=data1.values
# 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data1)

# 填补缺失值
imputer = SimpleImputer(strategy='mean')
data_filled = imputer.fit_transform(data_scaled)

# 将处理后的数据转换回DataFrame
data_final = pd.DataFrame(data_filled, columns=data1.columns)

y=data.iloc[:,30]
X = data_final.values
y = y.values
w,b=pla(X,y)
print(w,b)

predictions = np.sign(np.dot(X, w) + b)
# 计算一致的元素数量
num_correct = np.sum(predictions == y)

# 计算一致的比例
accuracy = num_correct / len(y)

print("一致的比例:", accuracy)








