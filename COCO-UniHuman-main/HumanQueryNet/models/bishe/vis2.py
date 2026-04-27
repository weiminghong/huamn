import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import platform
# ================= 1. 终极解决中文乱码问题 =================
# 自动检测操作系统，动态设置支持中文的字体，避免乱码
current_os = platform.system()
if current_os == 'Windows':
    # Windows系统通常都有微软雅黑
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
elif current_os == 'Darwin': # 苹果Mac系统
    # Mac系统通用的中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Helvetica']
else: # Linux或其他
    # Linux通常比较麻烦，尝试一些常用字体，若不成功需自行安装中文字体并设置
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'DejaVu Sans']

# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

# 1. 数据录入 (严格按照从 2016 到 2023 的顺序)
data = {
    'Year': [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Tech': [15064.3, 16952.3, 19057.9, 21424.5, 23720.6, 27108.5, 29794.8, 32137.1],
    'FDI': [1260.0, 1310.4, 1349.7, 1412.3, 1443.7, 1734.8, 1891.3, 1118.0],
    'Human': [958.6989, 1057.5606, 1086.8625, 1086.4227, 1184.0225, 1292.07, 1418.878, 1536.77],
    'RCA': [1.1357, 1.1446, 1.2194, 1.2264, 1.2000, 1.2087, 1.2945, 1.0864]
}

df = pd.DataFrame(data)
df.set_index('Year', inplace=True)

# 2. 对数化处理
df['ln_Tech'] = np.log(df['Tech'])
df['ln_FDI'] = np.log(df['FDI'])
df['ln_Human'] = np.log(df['Human'])
df['ln_RCA'] = np.log(df['RCA'])

X_cols = ['ln_Tech', 'ln_FDI', 'ln_Human']
X = df[X_cols]
Y = df['ln_RCA']

# 3. 相关性热力图
plt.figure(figsize=(8, 6))
corr_matrix = df[['ln_RCA', 'ln_Tech', 'ln_FDI', 'ln_Human']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title("变量对数化后的 Pearson 相关系数矩阵")
plt.savefig("Correlation_Matrix.png", dpi=300, bbox_inches='tight')

# 4. 多重共线性检验 (VIF)
X_vif = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# 5. OLS 回归模型构建
X_model = sm.add_constant(X)
model = sm.OLS(Y, X_model)
results = model.fit()

# 6. 输出学术级回归报表
print("\n========== OLS 线性回归详细大表 ==========")
print(results.summary())

# ================= 7. 【新增】直观提取回归系数与构建方程 =================
print("\n========== 核心系数提取与回归方程 ==========")
coef = results.params
pvalues = results.pvalues

print(f"常数项 (Intercept)   : {coef['const']:>8.4f}  |  P值: {pvalues['const']:.4f}")
print(f"科技投入弹性 (ln_Tech): {coef['ln_Tech']:>8.4f}  |  P值: {pvalues['ln_Tech']:.4f}")
print(f"外商投资弹性 (ln_FDI) : {coef['ln_FDI']:>8.4f}  |  P值: {pvalues['ln_FDI']:.4f}")
print(f"人力资本弹性 (ln_Human): {coef['ln_Human']:>8.4f}  |  P值: {pvalues['ln_Human']:.4f}")

print("\n【最终拟合的回归方程 (供论文直接使用)】:")
print(f"ln(RCA) = {coef['const']:.4f} + ({coef['ln_Tech']:.4f}) * ln(Tech) + ({coef['ln_FDI']:.4f}) * ln(FDI) + ({coef['ln_Human']:.4f}) * ln(Human)")
print("-" * 60)

# 8. 输出拟合效果对比图
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['ln_RCA'], marker='o', label='真实值 (ln_RCA)', color='blue')
plt.plot(df.index, results.fittedvalues, marker='s', linestyle='--', label='模型拟合值', color='red')
plt.title("RCA 真实值与回归拟合值对比 (对数尺度)")
plt.xlabel("年份")
plt.ylabel("ln(RCA)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("Regression_Fit.png", dpi=300, bbox_inches='tight')
plt.show()
print("✅ 所有图表已生成，回归方程已输出！")