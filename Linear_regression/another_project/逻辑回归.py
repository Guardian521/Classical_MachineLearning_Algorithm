import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
data = pd.read_csv('/Users/gushuai/Desktop/adult.csv')  # 替换为实际的数据文件路径


data = data.drop(['occupation'], axis=1)

# 转换数值
education_mapping = {'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5,
                     '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10,
                     'Bachelors': 11, 'Masters': 12, 'Doctorate': 13,'Assoc-voc':10,'Assoc-acdm':10
                     ,'Prof-school': 11,
}
data['education'] = data['education'].str.strip()  # 去掉每个值的前后空格
data['education'] = data['education'].map(education_mapping)


data['workclass']= pd.Series(data['workclass'].factorize()[0], index=data.index)
data['martial_status']=pd.Series(data['martial_status'].factorize()[0],index=data.index)
data['relationship']= pd.Series(data['relationship'].factorize()[0], index=data.index)
data['race']= pd.Series(data['race'].factorize()[0], index=data.index)
data['sex']= pd.Series(data['sex'].factorize()[0], index=data.index)
data['native_country']= pd.Series(data['native_country'].factorize()[0], index=data.index)


X = data.iloc[:, :-1]
y = data.iloc[:, -1]
y = y.replace({' <=50K': 0, ' >50K': 1})


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 评估模型
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 计算混淆矩阵
m = confusion_matrix(y_test, y_pred)
a = pd.DataFrame(m, index=['0（实际不流失）', '1（实际流失）'], columns=['0（预测不流失）', '1（预测流失）'])
print(a)

# 计算AUC
score = roc_auc_score(y_test, y_pred_proba[:, 1])
print('AUC score is %f' % score)

# 绘制ROC曲线
fpr, tpr, thres = roc_curve(y_test, y_pred_proba[:, 1])
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(fpr, tpr)
plt.title('ROC曲线')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()


ks = tpr - fpr
plt.plot(thres[1:], tpr[1:])
plt.plot(thres[1:], fpr[1:])
plt.plot(thres[1:], ks[1:])
plt.xlabel('threshold')
plt.legend(['tpr', 'fpr', 'ks'])
plt.gca().invert_xaxis()
plt.show()

a = pd.DataFrame(index=range(1986))  # 指定索引长度为1986
a['KS'] = ks