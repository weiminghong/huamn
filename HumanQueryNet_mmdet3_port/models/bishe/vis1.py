import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
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

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 读取与准备数据 =================
folder_path = '/home/priv/wmh_project/New Folder'
csv_file = os.path.join(folder_path, 'HighTech_Detailed_Summary_Perfect.csv')

print(f"正在读取数据并准备绘图: {csv_file}")
df = pd.read_csv(csv_file)

# 将年份对齐为 X轴，方便堆叠处理
df['Year'] = df['Year'].astype(int)
df.set_index('Year', inplace=True)

# 自动提取所有以 'MajorCat_' 开头的大类列
major_cols = [col for col in df.columns if col.startswith('MajorCat_')]
total_col = 'Total_All_HighTech'

# ================= 3. 开始绘图 (学术风优化版) =================
# 创建画布，尺寸更加学术化 (比 PPT 稍微宽一点)
fig, ax1 = plt.subplots(figsize=(14, 8))

# --- 优化配色 1: 学术风配色体系 ---
# 将 tab10 替换为更高级、更柔和的 'Set3'，区分度高且不晃眼
plot_df = df[major_cols].copy()
# 使用学术风配色 Set3 绘制堆叠柱状图，alpha调小让颜色更通透
plot_df.plot(kind='bar', stacked=True, ax=ax1, colormap='Set3', alpha=0.9, width=0.6, edgecolor='white', linewidth=0.5)

# --- 核心 2: 趋势折线图 (保持鲜明对比) ---
x_positions = range(len(df.index))
ax1.plot(x_positions, df[total_col],
         color='#D7191C', # 使用深红色 marker='o', markersize=7, linewidth=2.5, linestyle='-',
         label='高新技术总出口额 (折线趋势)')

# --- 数值标签优化 ---
# 定义一个辅助函数：将金额自动换算为 亿美元 格式
def fmt_y亿(x, pos):
    return f'{x/1000000:.1f}亿' # 1万亿 = 1,000,000,000 千美元，这里我们要显示为亿，所以除以100万

# 在折线上方添加数值标签
for i, val in enumerate(df[total_col]):
    # 将单位换算为 亿美元 显示，避免数字太长
    val_亿 = val / 1000000
    ax1.text(i, val * 1.02, f'{val_亿:.1f}亿', ha='center', va='bottom',
             color='#D7191C', fontweight='bold', fontsize=10)

# ================= 4. 图表细节深度美化 (论文级排版) =================
# 设置标题
ax1.set_title('2015-2024年中国高新技术产品出口结构与趋势分析', fontsize=20, pad=25, fontweight='bold')
# 设置坐标轴标签
ax1.set_xlabel('年份', fontsize=14, labelpad=12)
ax1.set_ylabel('出口金额 (千美元)', fontsize=14, labelpad=12)

# Y轴数字格式化：添加千位分隔符，避免科学计数法，并简化刻度
ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# --- 学术美化 1: 隐藏顶部和右侧边框 ---
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# 隐藏刻度线，仅保留标签
ax1.tick_params(axis='x', length=0)

# 调整刻度字体大小
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)

# --- 学术美化 2: 优化图例布局和中文名称 ---
handles, labels = ax1.get_legend_handles_labels()
# 【关键修复】彻底剔除图例中的丑陋前缀 `'MajorCat_1_'`，只保留商品名称
clean_labels = [label.replace('MajorCat_', '').strip() for label in labels]

# 将图例移至下方横向排列，不遮挡数据
ax1.legend(handles, clean_labels, fontsize=12, handlelength=1.5,
           loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)

# 学术背景网格线 (仅Y轴显示，浅灰虚线，轻量化)
ax1.grid(axis='y', linestyle='-', color='#DDDDDD', alpha=0.6, linewidth=0.5)

# 自动调整布局，防止底部图例被裁剪
plt.tight_layout()

# ================= 5. 保存结果 =================
output_img = os.path.join(folder_path, "HighTech_Trend_Perfect_Academic.png")
# 使用utf-8-sig编码保存文件名，解决Windows文件名乱码问题
# 使用 300 dpi 的高清像素保存， bbox_inches='tight' 裁剪掉多余白边
plt.savefig(output_img, dpi=1300, bbox_inches='tight')
print(f"\n✅ 学术风高清图表已成功生成！不仅解决了中文乱码，还优化了配色和布局。已保存至: {output_img}")

# 在屏幕上展示
plt.show()