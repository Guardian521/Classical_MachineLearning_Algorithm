import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DecisionTree():
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        num_samples_per_class = [np.sum(y == i) for i in range(2)]
        predicted_class = np.argmax(num_samples_per_class)

        node = {
            'predicted_class': predicted_class,
            'num_samples': len(y),
            'impurity': self._calculate_gini_impurity(y),
        }

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['index'] = idx
                node['threshold'] = thr
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(2)]
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * 2
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(2)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(2)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _calculate_gini_impurity(self, y):
        m = len(y)
        if m == 0:
            return 0
        p = np.sum(y) / m
        return 1.0 - (p ** 2 + (1 - p) ** 2)

    def _predict_sample(self, x, tree):
        if 'predicted_class' in tree:
            return tree['predicted_class']
        else:
            if x[tree['index']] < tree['threshold']:
                return self._predict_sample(x, tree['left'])
            else:
                return self._predict_sample(x, tree['right'])

    def predict(self, X):
        return [self._predict_sample(x, self.tree) for x in X]


# 使用示例
diabetes = load_diabetes()
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = (diabetes.target > 150).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

tree_model = DecisionTree(max_depth=3)

# 训练模型
tree_model.fit(X_train.values, y_train.values)

# 在测试集上进行预测
y_pred = tree_model.predict(X_test.values)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f'准确率：{accuracy:.2f}')

feature_names = ['age', 'sex', 'bmi_index', 'blood_pressure', 'cholesterol', 'ldl','hdl','tch','ith','glu']


import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

def draw_tree(tree, feature_names, depth=0, G=None):
    if G is None:
        G = nx.DiGraph()
        G.add_node(0, description=f"Feature: {feature_names[tree['index']]}\n"
                                  f"Threshold: {tree['threshold']:.2f}\n"
                                  f"Impurity: {tree['impurity']:.2f}")
    if 'left' in tree:
        left_description = f"Feature: {feature_names[tree['left']['index']]}\n" \
                           f"Threshold: {tree['left']['threshold']:.2f}\n" \
                           f"Impurity: {tree['left']['impurity']:.2f}" if 'threshold' in tree['left'] else \
            f"Class: {tree['left']['predicted_class']}\n" \
            f"Samples: {tree['left']['num_samples']}"
        G.add_node(2 * depth + 1, description=left_description)
        G.add_edge(depth, 2 * depth + 1)
        draw_tree(tree['left'], feature_names, 2 * depth + 1, G)
    if 'right' in tree:
        right_description = f"Feature: {feature_names[tree['right']['index']]}\n" \
                            f"Threshold: {tree['right']['threshold']:.2f}\n" \
                            f"Impurity: {tree['right']['impurity']:.2f}" if 'threshold' in tree['right'] else \
            f"Class: {tree['right']['predicted_class']}\n" \
            f"Samples: {tree['right']['num_samples']}"
        G.add_node(2 * depth + 2, description=right_description)
        G.add_edge(depth, 2 * depth + 2)
        draw_tree(tree['right'], feature_names, 2 * depth + 2, G)
    return G



tree_model = DecisionTree(max_depth=3)
tree_model.fit(X_train.values, y_train.values)
G = draw_tree(tree_model.tree, feature_names)
pos = graphviz_layout(G, prog='dot')
plt.figure(figsize=(10, 10))
nx.draw(G, pos, with_labels=False, arrows=False)
labels = nx.get_node_attributes(G, 'description')
nx.draw_networkx_labels(G, pos, labels)
plt.show()