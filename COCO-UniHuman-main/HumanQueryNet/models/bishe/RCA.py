import pandas as pd

# 1. 录入整理好的原始四个列表数据
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],

    # 中国数据 (单位：千美元)
    'China_Total_Export': [2273468224, 2097637243, 2283601502, 2505992913, 2516745192,
                           2606806545, 3338885320, 5331208215, 4906348475, 3480108359],
    'China_HighTech_Export': [345535579.1, 319510645.7, 352928714.9, 403367765.4, 408720905.5,
                              422966455.6, 537728401.3, 900545423.9, 794083059.0, 397864468.6],

    # 世界数据 (单位：万亿美元)
    'World_Total_Export': [16.53, 16.03, 17.7, 19.47, 18.88, 17.75, 22.44, 25.29, 24.03, 24.58],
    'World_HighTech_Export': [2.11, 2.15, 2.39, 2.57, 2.5, 2.4, 2.99, 3.3, 3.58, 2.97]
}

# 2. 创建 DataFrame
df = pd.DataFrame(data)

# 3. 计算占比 (这一步消除了量级差异)
# 中国高技术出口占中国总出口的比重
df['China_Share'] = df['China_HighTech_Export'] / df['China_Total_Export']
# 世界高技术出口占世界总出口的比重
df['World_Share'] = df['World_HighTech_Export'] / df['World_Total_Export']

# 4. 计算 RCA (显示性比较优势指数)
df['RCA'] = df['China_Share'] / df['World_Share']

# 保留四位小数让结果更美观
df['RCA'] = df['RCA'].round(4)
df['China_Share'] = df['China_Share'].round(4)
df['World_Share'] = df['World_Share'].round(4)

# 5. 打印结果
print("=================== 2015-2024 中国高新技术产品 RCA 指数计算结果 ===================")
# 只展示核心列
result_df = df[['Year', 'China_Share', 'World_Share', 'RCA']]
print(result_df.to_string(index=False))

# (可选) 将最终结果导出为 CSV 供论文使用
# df.to_csv('China_HighTech_RCA_2015_2024.csv', index=False, encoding='utf-8-sig')