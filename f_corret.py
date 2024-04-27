import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 加载数据
file_path = 'E:\pythonProject\随机森林.xlsx'  # 请根据实际情况调整文件路径
data = pd.read_excel(file_path)

# 定义特征和目标变量
X = data.drop('Final_Evaluation', axis=1)  # 特征
y = data['Final_Evaluation']  # 目标变量

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],  # 树的数量
    'max_depth': [None, 10, 20],  # 树的最大深度
    'min_samples_split': [2, 4, 6],  # 分割内部节点所需的最小样本数
    'min_samples_leaf': [1, 2, 4],  # 叶节点必须拥有的最小样本数量
}

# 初始化网格搜索对象，使用随机森林作为分类器，采用交叉验证
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, verbose=1, n_jobs=-1)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 找到最佳参数组合
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("最佳参数：", best_params)
print("最佳交叉验证分数：", best_score)

# 使用最佳参数在测试集上进行评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("测试集上的准确率：", accuracy)
