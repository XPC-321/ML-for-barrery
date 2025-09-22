import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import Series, DataFrame
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
import shap
shap.initjs()


# 读取数据
cor = pd.read_excel('~/data.xlsx', sheet_name='Sheet1')
# 计算相关系数矩阵，包含了任意两个特征间的相关系数
print('各个特征的相关系数矩阵为：\n', cor.corr())


# 绘制相关性热力图
plt.subplots(figsize=(15, 15))  # 设置画面大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置横纵刻度大小
y_tick = np.linspace(0,20,5)
plt.yticks(y_tick,fontsize=20,color='#000000')
x_tick = np.linspace(0,20,5)
plt.xticks(x_tick,fontsize=20,color='#000000')

sns.heatmap(cor.corr(), annot=True, vmax=1, square=True, cmap="coolwarm_r")
plt.title('相关性热力图', font={'size':40})
plt.show()

data = pd.read_excel('~/data.xlsx', sheet_name='Sheet1')
y = data["Y"]
X = data[["d1","d2","d3","d4","d5"]]
scaler = StandardScaler()
ss = scaler.fit_transform(X)
DF = DataFrame(data=ss, columns=["d1","d2","d3","d4","d5"])

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(DF, y, test_size=0.3, random_state=40)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
rfc.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
cv_scores_1 = cross_val_score(rfc, DF, y, cv=5, scoring='r2' )
cv_scores_2 = cross_val_score(rfc, DF, y, cv=5, scoring='neg_mean_squared_error' )
r2 = r2_score(y, rfc.predict(DF))
rmse_1 = mean_squared_error(y_test, rfc.predict(X_test))
rmse_2 = mean_squared_error(y_train, rfc.predict(X_train))
print(r2, rmse_1, rmse_2)
print("Cross-validation r2 scores:", cv_scores_1)
print("Cross-validation MSE scores:", cv_scores_2)
print("Average cross-validation r2:", cv_scores_1.mean())
print("Average cross-validation MSE:", cv_scores_2.mean())
explainer_2 = shap.TreeExplainer(model=rfc)
shap_values_2 = explainer_2(DF)
import matplotlib.ticker as ticker

# 生成SHAP条形图
shap.plots.bar(shap_values_2, max_display=13, show=False)

# 设置全局字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['Arial']  # Arial支持中文需确保系统有中文字体
plt.rcParams['axes.unicode_minus'] = False

# 获取当前轴对象
ax = plt.gca()

# --- 横坐标格式修改（新增代码）---
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))  # 保留两位小数

# --- 原有样式设置（保留）---
# 刻度字体设置
ax.tick_params(axis='both', labelsize=16)  # 坐标轴刻度字体大小

# 坐标轴标签字体设置
ax.set_xlabel(ax.get_xlabel(), fontsize=16)  # X轴标签
ax.set_ylabel(ax.get_ylabel(), fontsize=16)  # Y轴标签

# 标题字体设置（如果存在标题）
if ax.get_title():
    ax.set_title(ax.get_title(), fontsize=18)

# 坐标轴线宽设置
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_visible(False)  # 隐藏右侧边框（可选）
ax.spines['top'].set_visible(False)  # 隐藏顶部边框（可选）

# 刻度线样式
ax.tick_params(axis='y', direction='out', length=5, width=2)
ax.tick_params(axis='x', length=5, width=2)

# 显示图表
plt.tight_layout()  # 优化布局（防止标签截断）
plt.show()
shap.summary_plot(shap_values_2, DF,show=None)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 获取当前轴对象
ax = plt.gca()

# 设置刻度大小
ax.tick_params(axis='both', labelsize=16)  # 设置横纵坐标刻度的字体大小，单位为 pt
plt.xticks(rotation=45, ha='right', fontsize=16, fontfamily='Arial')

# 设置标签字体大小
ax.set_xlabel(ax.get_xlabel(), fontsize=16)  # 设置 X 轴标签字体大小
ax.set_ylabel(ax.get_ylabel(), fontsize=16)  # 设置 Y 轴标签字体大小
ax.set_title(ax.get_title(), fontsize=18)  # 设置标题字体大小
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params(axis='y', direction='out', length=5, width=2)  # 刻度线朝外
ax.tick_params(axis='x', length=5, width=2)  # 可选：修改 x 轴刻度线样式

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# 显示图表
plt.show()

from matplotlib.ticker import MultipleLocator
fig, ax = plt.subplots(figsize=(10.72, 8.205)
                       #,dpi=1000
                      )
# 绘制散点图
ax.scatter(y_train, rfc.predict(X_train), color = "white", edgecolors="black",linewidth=2,s=100)
ax.scatter(y_test, rfc.predict(X_test), color = "white", edgecolors=(0, 0.3, 0.5),linewidth=2,s=100)

# y=x的黑色直线，'k--'为颜色，lw=4为线条大小

ax.plot([0, 100], [0, 100], 'k--', lw=1.5)
plt.ylim(1, 5)
plt.xlim(1, 5)
ax.set_xlabel('Measured (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=24)
ax.set_ylabel('Predicted (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=24)
plt.xticks(fontsize=20,fontfamily='Arial',
           # fontweight='bold'
          )
plt.yticks(fontsize=20,fontfamily='Arial',
           # fontweight='bold'
          )
# 设置横坐标刻度间隔为0.5
ax.xaxis.set_major_locator(MultipleLocator(1))
# 设置纵坐标刻度间隔为0.5
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.spines['top'].set_linewidth(2)    # 设置顶部边框宽度
ax.spines['right'].set_linewidth(2)  # 设置右侧边框宽度
ax.spines['bottom'].set_linewidth(2) # 设置底部边框宽度
ax.spines['left'].set_linewidth(2)   # 设置左侧边框宽度
# 修改刻度线方向和样式
ax.tick_params(axis='y', direction='out', width=2)  # 刻度线朝外
ax.tick_params(axis='x', length=5, width=2)  # 可选：修改 x 轴刻度线样式
plt.show()

# 其他模型
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression()
lr.fit(X_train,y_train)

fig, ax = plt.subplots(figsize=(10.72, 8.205)
                       ,dpi=500
                      )
# 绘制散点图，
ax.scatter(y_train, lr.predict(X_train), color = "white", edgecolors="black",linewidth=2,s=300)
ax.scatter(y_test, lr.predict(X_test), color = "white", edgecolors=(0, 0.3, 0.5),linewidth=2,s=300)

# y=x的黑色直线，'k--'为颜色，lw=4为线条大小

ax.plot([0, 100], [0, 100], 'k--', lw=1.5)
plt.ylim(1.5, 4.5)
plt.xlim(1.5, 4.5)
ax.set_xlabel('Measured (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
ax.set_ylabel('Predicted (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
plt.xticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
plt.yticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
# 设置横坐标刻度间隔为0.5
ax.xaxis.set_major_locator(MultipleLocator(1))
# 设置纵坐标刻度间隔为0.5
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.spines['top'].set_linewidth(2)    # 设置顶部边框宽度
ax.spines['right'].set_linewidth(2)  # 设置右侧边框宽度
ax.spines['bottom'].set_linewidth(2) # 设置底部边框宽度
ax.spines['left'].set_linewidth(2)   # 设置左侧边框宽度
# 修改刻度线方向和样式
ax.tick_params(axis='y', direction='out', width=2, length=10)  # 刻度线朝外
ax.tick_params(axis='x', length=10, width=2)  # 可选：修改 x 轴刻度线样式
plt.show()
from sklearn.model_selection import cross_val_score
cv_scores_lr1 = cross_val_score(lr, DF, y, cv=5, scoring='r2' )
cv_scores_lr2 = cross_val_score(lr, DF, y, cv=5, scoring='neg_mean_squared_error' )
r2_lr = r2_score(y, lr.predict(DF))
rmse_lr1 = mean_squared_error(y_test, lr.predict(X_test))
rmse_lr2 = mean_squared_error(y_train, lr.predict(X_train))
print(r2_lr, rmse_lr1, rmse_lr2)
print("Cross-validation r2 scores:", cv_scores_lr1)
print("Cross-validation MSE scores:", cv_scores_lr2)
print("Average cross-validation r2:", cv_scores_lr1.mean())
print("Average cross-validation MSE:", cv_scores_lr2.mean())

svr = SVR()
svr.fit(X_train,y_train)

fig, ax = plt.subplots(figsize=(10.72, 8.205)
                       ,dpi=500
                      )
# 绘制散点图，
ax.scatter(y_train, svr.predict(X_train), color = "white", edgecolors="black",linewidth=2,s=300)
ax.scatter(y_test, svr.predict(X_test), color = "white", edgecolors=(0, 0.3, 0.5),linewidth=2,s=300)

# y=x的黑色直线，'k--'为颜色，lw=4为线条大小

ax.plot([0, 100], [0, 100], 'k--', lw=1.5)
plt.ylim(1, 5)
plt.xlim(1, 5)
ax.set_xlabel('Measured (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
ax.set_ylabel('Predicted (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
plt.xticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
plt.yticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
# 设置横坐标刻度间隔为0.5
ax.xaxis.set_major_locator(MultipleLocator(1))
# 设置纵坐标刻度间隔为0.5
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.spines['top'].set_linewidth(2)    # 设置顶部边框宽度
ax.spines['right'].set_linewidth(2)  # 设置右侧边框宽度
ax.spines['bottom'].set_linewidth(2) # 设置底部边框宽度
ax.spines['left'].set_linewidth(2)   # 设置左侧边框宽度
# 修改刻度线方向和样式
ax.tick_params(axis='y', direction='out', width=2, length=10)  # 刻度线朝外
ax.tick_params(axis='x', length=10, width=2)  # 可选：修改 x 轴刻度线样式
plt.show()
from sklearn.model_selection import cross_val_score
cv_scores_svr1 = cross_val_score(svr, DF, y, cv=5, scoring='r2' )
cv_scores_svr2 = cross_val_score(svr, DF, y, cv=5, scoring='neg_mean_squared_error' )
r2_svr = r2_score(y, svr.predict(DF))
rmse_svr1 = mean_squared_error(y_test, svr.predict(X_test))
rmse_svr2 = mean_squared_error(y_train, svr.predict(X_train))
print(r2_svr, rmse_svr1, rmse_svr2)
print("Cross-validation r2 scores:", cv_scores_svr1)
print("Cross-validation MSE scores:", cv_scores_svr2)
print("Average cross-validation r2:", cv_scores_svr1.mean())
print("Average cross-validation MSE:", cv_scores_svr2.mean())

from sklearn.linear_model import Ridge
rd = Ridge(alpha=1)
rd.fit(X_train,y_train)

fig, ax = plt.subplots(figsize=(10.72, 8.205)
                       ,dpi=500
                      )
# 绘制散点图，
ax.scatter(y_train, rd.predict(X_train), color = "white", edgecolors="black",linewidth=2,s=300)
ax.scatter(y_test, rd.predict(X_test), color = "white", edgecolors=(0, 0.3, 0.5),linewidth=2,s=300)

# y=x的黑色直线，'k--'为颜色，lw=4为线条大小

ax.plot([0, 100], [0, 100], 'k--', lw=1.5)
plt.ylim(1, 5)
plt.xlim(1, 5)
ax.set_xlabel('Measured (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
ax.set_ylabel('Predicted (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
plt.xticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
plt.yticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
# 设置横坐标刻度间隔为0.5
ax.xaxis.set_major_locator(MultipleLocator(1))
# 设置纵坐标刻度间隔为0.5
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.spines['top'].set_linewidth(2)    # 设置顶部边框宽度
ax.spines['right'].set_linewidth(2)  # 设置右侧边框宽度
ax.spines['bottom'].set_linewidth(2) # 设置底部边框宽度
ax.spines['left'].set_linewidth(2)   # 设置左侧边框宽度
# 修改刻度线方向和样式
ax.tick_params(axis='y', direction='out', width=2, length=10)  # 刻度线朝外
ax.tick_params(axis='x', length=10, width=2)  # 可选：修改 x 轴刻度线样式
plt.show()
from sklearn.model_selection import cross_val_score
cv_scores_rd1 = cross_val_score(rd, DF, y, cv=5, scoring='r2' )
cv_scores_rd2 = cross_val_score(rd, DF, y, cv=5, scoring='neg_mean_squared_error' )
r2_rd = r2_score(y, rd.predict(DF))
rmse_rd1 = mean_squared_error(y_test, rd.predict(X_test))
rmse_rd2 = mean_squared_error(y_train, rd.predict(X_train))
print(r2_rd, rmse_rd1, rmse_rd2)
print("Cross-validation r2 scores:", cv_scores_rd1)
print("Cross-validation MSE scores:", cv_scores_rd2)
print("Average cross-validation r2:", cv_scores_rd1.mean())
print("Average cross-validation MSE:", cv_scores_rd2.mean())

from sklearn.linear_model import Lasso
la = Lasso(alpha=0.005)
la.fit(X_train,y_train)


fig, ax = plt.subplots(figsize=(10.72, 8.205)
                       ,dpi=500
                      )
# 绘制散点图，
ax.scatter(y_train, la.predict(X_train), color = "white", edgecolors="black",linewidth=2,s=300)
ax.scatter(y_test, la.predict(X_test), color = "white", edgecolors=(0, 0.3, 0.5),linewidth=2,s=300)

# y=x的黑色直线，'k--'为颜色，lw=4为线条大小

ax.plot([0, 100], [0, 100], 'k--', lw=1.5)
plt.ylim(1, 5)
plt.xlim(1, 5)
ax.set_xlabel('Measured (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
ax.set_ylabel('Predicted (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
plt.xticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
plt.yticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
# 设置横坐标刻度间隔为0.5
ax.xaxis.set_major_locator(MultipleLocator(1))
# 设置纵坐标刻度间隔为0.5
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.spines['top'].set_linewidth(2)    # 设置顶部边框宽度
ax.spines['right'].set_linewidth(2)  # 设置右侧边框宽度
ax.spines['bottom'].set_linewidth(2) # 设置底部边框宽度
ax.spines['left'].set_linewidth(2)   # 设置左侧边框宽度
# 修改刻度线方向和样式
ax.tick_params(axis='y', direction='out', width=2, length=10)  # 刻度线朝外
ax.tick_params(axis='x', length=10, width=2)  # 可选：修改 x 轴刻度线样式
plt.show()
from sklearn.model_selection import cross_val_score
cv_scores_la1 = cross_val_score(la, DF, y, cv=5, scoring='r2' )
cv_scores_la2 = cross_val_score(la, DF, y, cv=5, scoring='neg_mean_squared_error' )
r2_la = r2_score(y, la.predict(DF))
rmse_la1 = mean_squared_error(y_test, la.predict(X_test))
rmse_la2 = mean_squared_error(y_train, la.predict(X_train))
print(r2_la, rmse_la1, rmse_la2)
print("Cross-validation r2 scores:", cv_scores_la1)
print("Cross-validation MSE scores:", cv_scores_la2)
print("Average cross-validation r2:", cv_scores_la1.mean())
print("Average cross-validation MSE:", cv_scores_la2.mean())

from sklearn.neighbors import KNeighborsRegressor
kn = KNeighborsRegressor()
kn.fit(X_train,y_train)

fig, ax = plt.subplots(figsize=(10.72, 8.205)
                       ,dpi=500
                      )
# 绘制散点图，
ax.scatter(y_train, kn.predict(X_train), color = "white", edgecolors="black",linewidth=2,s=300)
ax.scatter(y_test, kn.predict(X_test), color = "white", edgecolors=(0, 0.3, 0.5),linewidth=2,s=300)

# y=x的黑色直线，'k--'为颜色，lw=4为线条大小

ax.plot([0, 100], [0, 100], 'k--', lw=1.5)
plt.ylim(1, 5)
plt.xlim(1, 5)
ax.set_xlabel('Measured (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
ax.set_ylabel('Predicted (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
plt.xticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
plt.yticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
# 设置横坐标刻度间隔为0.5
ax.xaxis.set_major_locator(MultipleLocator(1))
# 设置纵坐标刻度间隔为0.5
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.spines['top'].set_linewidth(2)    # 设置顶部边框宽度
ax.spines['right'].set_linewidth(2)  # 设置右侧边框宽度
ax.spines['bottom'].set_linewidth(2) # 设置底部边框宽度
ax.spines['left'].set_linewidth(2)   # 设置左侧边框宽度
# 修改刻度线方向和样式
ax.tick_params(axis='y', direction='out', width=2, length=10)  # 刻度线朝外
ax.tick_params(axis='x', length=10, width=2)  # 可选：修改 x 轴刻度线样式
plt.show()
from sklearn.model_selection import cross_val_score
cv_scores_kn1 = cross_val_score(kn, DF, y, cv=5, scoring='r2' )
cv_scores_kn2 = cross_val_score(kn, DF, y, cv=5, scoring='neg_mean_squared_error' )
r2_kn = r2_score(y, kn.predict(DF))
rmse_kn1 = mean_squared_error(y_test, kn.predict(X_test))
rmse_kn2 = mean_squared_error(y_train, kn.predict(X_train))
print(r2_kn, rmse_kn1, rmse_kn2)
print("Cross-validation r2 scores:", cv_scores_kn1)
print("Cross-validation MSE scores:", cv_scores_kn2)
print("Average cross-validation r2:", cv_scores_kn1.mean())
print("Average cross-validation MSE:", cv_scores_kn2.mean())

from sklearn.ensemble import AdaBoostRegressor
abr = AdaBoostRegressor(random_state=4).fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(10.72, 8.205)
                       ,dpi=500
                      )

# 绘制散点图，
ax.scatter(y_train, abr.predict(X_train), color = "white", edgecolors="black",linewidth=2,s=300)
ax.scatter(y_test, abr.predict(X_test), color = "white", edgecolors=(0, 0.3, 0.5),linewidth=2,s=300)

# y=x的黑色直线，'k--'为颜色，lw=4为线条大小
ax.plot([-10,10], [-10, 10], 'k--', lw=1.5)
plt.ylim(1,5)
plt.xlim(1, 5)
ax.set_xlabel('Measured (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
ax.set_ylabel('Predicted (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
plt.xticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
plt.yticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
# 设置横坐标刻度间隔为0.5
ax.xaxis.set_major_locator(MultipleLocator(1))
# 设置纵坐标刻度间隔为0.5
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.spines['top'].set_linewidth(2)    # 设置顶部边框宽度
ax.spines['right'].set_linewidth(2)  # 设置右侧边框宽度
ax.spines['bottom'].set_linewidth(2) # 设置底部边框宽度
ax.spines['left'].set_linewidth(2)   # 设置左侧边框宽度
# 修改刻度线方向和样式
ax.tick_params(axis='y', direction='out', width=2, length=10)  # 刻度线朝外
ax.tick_params(axis='x', length=10, width=2)  # 可选：修改 x 轴刻度线样式
plt.show()
from sklearn.model_selection import cross_val_score
cv_scores_abr1 = cross_val_score(abr, DF, y, cv=5, scoring='r2' )
cv_scores_abr2 = cross_val_score(abr, DF, y, cv=5, scoring='neg_mean_squared_error' )
r2_abr = r2_score(y, abr.predict(DF))
rmse_abr1 = mean_squared_error(y_test, abr.predict(X_test))
rmse_abr2 = mean_squared_error(y_train, abr.predict(X_train))
print(r2_abr, rmse_abr1, rmse_abr2)
print("Cross-validation r2 scores:", cv_scores_abr1)
print("Cross-validation MSE scores:", cv_scores_abr2)
print("Average cross-validation r2:", cv_scores_abr1.mean())
print("Average cross-validation MSE:", cv_scores_abr2.mean())

from sklearn.ensemble import HistGradientBoostingRegressor
hgb = HistGradientBoostingRegressor(random_state=16).fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(10.72, 8.205)
                       ,dpi=500
                      )

# 绘制散点图，
ax.scatter(y_train, hgb.predict(X_train), color = "white", edgecolors="black",linewidth=2,s=300)
ax.scatter(y_test, hgb.predict(X_test), color = "white", edgecolors=(0, 0.3, 0.5),linewidth=2,s=300)

# y=x的黑色直线，'k--'为颜色，lw=4为线条大小
ax.plot([-10,10], [-10, 10], 'k--', lw=1.5)
plt.ylim(1,5)
plt.xlim(1,5)
ax.set_xlabel('Measured (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
ax.set_ylabel('Predicted (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
plt.xticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
plt.yticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
# 设置横坐标刻度间隔为0.5
ax.xaxis.set_major_locator(MultipleLocator(1))
# 设置纵坐标刻度间隔为0.5
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.spines['top'].set_linewidth(2)    # 设置顶部边框宽度
ax.spines['right'].set_linewidth(2)  # 设置右侧边框宽度
ax.spines['bottom'].set_linewidth(2) # 设置底部边框宽度
ax.spines['left'].set_linewidth(2)   # 设置左侧边框宽度
# 修改刻度线方向和样式
ax.tick_params(axis='y', direction='out', width=2, length=10)  # 刻度线朝外
ax.tick_params(axis='x', length=10, width=2)  # 可选：修改 x 轴刻度线样式
plt.show()
from sklearn.model_selection import cross_val_score
cv_scores_hgb1 = cross_val_score(hgb, DF, y, cv=5, scoring='r2' )
cv_scores_hgb2 = cross_val_score(hgb, DF, y, cv=5, scoring='neg_mean_squared_error' )
r2_hgb = r2_score(y, hgb.predict(DF))
rmse_hgb1 = mean_squared_error(y_test, hgb.predict(X_test))
rmse_hgb2 = mean_squared_error(y_train, hgb.predict(X_train))
print(r2_hgb, rmse_hgb1, rmse_hgb2)
print("Cross-validation r2 scores:", cv_scores_hgb1)
print("Cross-validation MSE scores:", cv_scores_hgb2)
print("Average cross-validation r2:", cv_scores_hgb1.mean())
print("Average cross-validation MSE:", cv_scores_hgb2.mean())

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor(random_state=41)
rfc.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
cv_scores_1 = cross_val_score(rfc, DF, y, cv=5, scoring='r2' )
cv_scores_2 = cross_val_score(rfc, DF, y, cv=5, scoring='neg_mean_squared_error' )
r2 = r2_score(y, rfc.predict(DF))
rmse_1 = mean_squared_error(y_test, rfc.predict(X_test))
rmse_2 = mean_squared_error(y_train, rfc.predict(X_train))
print(r2, rmse_1, rmse_2)
print("Cross-validation r2 scores:", cv_scores_1)
print("Cross-validation MSE scores:", cv_scores_2)
print("Average cross-validation r2:", cv_scores_1.mean())
print("Average cross-validation MSE:", cv_scores_2.mean())
from matplotlib.ticker import MultipleLocator
fig, ax = plt.subplots(figsize=(10.72, 8.205)
                       ,dpi=500
                      )
# 绘制散点图
ax.scatter(y_train, rfc.predict(X_train), color = "white", edgecolors="black",linewidth=2,s=300)
ax.scatter(y_test, rfc.predict(X_test), color = "white", edgecolors=(0, 0.3, 0.5),linewidth=2,s=300)

# y=x的黑色直线，'k--'为颜色，lw=4为线条大小

ax.plot([0, 100], [0, 100], 'k--', lw=1.5)
plt.ylim(1, 5)
plt.xlim(1, 5)
ax.set_xlabel('Measured (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
ax.set_ylabel('Predicted (eV)',fontfamily='Arial',
              # fontweight='bold',
              fontsize=36)
plt.xticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
plt.yticks(fontsize=28,fontfamily='Arial',
           # fontweight='bold'
          )
# 设置横坐标刻度间隔为0.5
ax.xaxis.set_major_locator(MultipleLocator(1))
# 设置纵坐标刻度间隔为0.5
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.spines['top'].set_linewidth(2)    # 设置顶部边框宽度
ax.spines['right'].set_linewidth(2)  # 设置右侧边框宽度
ax.spines['bottom'].set_linewidth(2) # 设置底部边框宽度
ax.spines['left'].set_linewidth(2)   # 设置左侧边框宽度
# 修改刻度线方向和样式
ax.tick_params(axis='y', direction='out', width=2, length=10)  # 刻度线朝外
ax.tick_params(axis='x', length=10, width=2)  # 可选：修改 x 轴刻度线样式
plt.show()
