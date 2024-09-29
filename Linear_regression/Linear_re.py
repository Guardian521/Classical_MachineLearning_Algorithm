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


X=[(1,2,1),(2,3,-1),(1.2,2.3,1),(2.1,3.2,1),(1.3,1.7,1),(2.4,3.2,-1)]
X=pd.DataFrame(X)
y=X.iloc[:,2]
X=X.iloc[:,0:2]
X = X.to_numpy()
w,b=pla(X,y)
print(w,b)

predictions = np.sign(np.dot(X, w) + b)
# 计算一致的元素数量
num_correct = np.sum(predictions == y)

# 计算一致的比例
accuracy = num_correct / len(y)

print("一致的比例:", accuracy)








