import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# 读取CSV文件
data = pd.read_csv('ICE_new.csv').iloc[:, 1:]

# 假设最后一列是目标变量
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 目标变量

# 划分训练集和测试集
y = data['ICE']
X = data.drop('ICE', axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 归一化处理
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.05,
    'max_depth': 3,
    'subsample': 0.05,
    'colsample_bytree': 0.2
}
model = xgb.train(params, dtrain, num_boost_round=100)

# 预测结果
y_pred = model.predict(dtest)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
xgb.plot_importance(model)
plt.show()
# # 创建XGBoost分类器
# model = XGBClassifier()

# # 训练模型
# model.fit(X_train, y_train)

# # 预测
# y_pred = model.predict(X_test)

# 评估模型
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')

explainer = shap.TreeExplainer(model)
shap_interaction = explainer.shap_interaction_values(X)