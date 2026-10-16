import pandas as pd
import glob
import os

# 1. 定义文件夹路径和获取所有 csv 文件
folder_path = '/home/priv/wmh_project/New Folder'
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

# 过滤掉之前生成的汇总文件
all_files = [f for f in all_files if "Summary" not in os.path.basename(f)]

if not all_files:
    print(f"提示: 在路径 {folder_path} 下没有找到有效的源 .csv 文件。")
else:
    df_list = []
    print(f"\n======== 开始精准提取核心数据，共找到 {len(all_files)} 个源 CSV ========")

    # 核心：明确指定只读取这三列
    target_columns = ['ProductCode', 'Year', 'TradeValue in 1000 USD']

    # 2. 循环读取文件
    for file in all_files:
        filename = os.path.basename(file)
        try:
            df = pd.read_csv(file, usecols=target_columns, encoding='latin-1', dtype=str, on_bad_lines='skip')
        except ValueError:
            print(f"❌ 缺少列名 | {filename} 中没有找到我们需要的那三列，请检查文件内容。")
            continue
        except Exception as e:
            print(f"❌ 读取失败 | {filename}: {e}")
            continue

        # --- 数据清洗 ---
        df['Year'] = df['Year'].astype(str).str.extract(r'((?:19|20)\d{2})')[0]
        df.dropna(subset=['Year'], inplace=True)

        if df.empty:
            continue

        df['TradeValue in 1000 USD'] = pd.to_numeric(df['TradeValue in 1000 USD'], errors='coerce')
        df['ProductCode'] = df['ProductCode'].astype(str).str.replace(r'\D', '', regex=True).str.zfill(4)

        years_in_file = df['Year'].unique()
        print(f"✅ 读取成功 | {filename} | 提取到年份: {list(years_in_file)} | 有效行数: {len(df)}")

        df_list.append(df)

    # ================= 匹配与统计逻辑 =================

    if not df_list:
        print("\n未能提取到任何有效数据，请检查上方日志。")
    else:
        print("\n======== 文件读取完毕，开始进行全面数据统计 ========")
        full_data = pd.concat(df_list, ignore_index=True)
        # 踢掉金额为空的数据
        full_data.dropna(subset=['TradeValue in 1000 USD'], inplace=True)

        col_val = 'TradeValue in 1000 USD'

        # 【新增需求】：在过滤高新技术之前，先计算每一年的“全部商品出口总额”
        yearly_absolute_total = full_data.groupby('Year')[col_val].sum().reset_index()
        yearly_absolute_total.rename(columns={col_val: 'Total_All_Products_Export'}, inplace=True)
        print("✅ 已完成每年度全部商品出口总额的计算。")

        # ================= 高新技术产品处理 =================
        # SITC Rev.4 字典
        category_dict = {
            '1_航空航天': ['0714', '7921', '7922', '7923', '7924', '7925', '79291', '79293', '87411'],
            '2_计算机与办公设备': ['75194', '75195', '0752', '75997'],
            '3_电子与通信': ['7641', '7643', '77318', '77625', '77627', '7763', '7764'],
            '4_医药产品': ['5413', '5414', '5415', '5416'],
            '5_科学仪器': ['0774', '0871', '87211', '0874', '88111', '88121', '88411', '88419'],
            '6_电气机械及高端装备': ['77869', '7787', '7788'],
            '7_精细化工与新材料': ['5222', '5223', '5224', '0525', '5311', '5312', '5743', '5744']
        }

        # 构建展平字典并按长度降序排列
        code_to_cat = {code: cat for cat, codes in category_dict.items() for code in codes}
        sorted_target_codes = sorted(code_to_cat.keys(), key=len, reverse=True)


        def match_info(prod_code):
            if not prod_code:
                return None, None
            for target in sorted_target_codes:
                if prod_code.startswith(target):
                    return code_to_cat[target], target
            return None, None


        print("正在匹配 SITC Rev.4 高新技术编码...")
        # 执行匹配
        matched_results = full_data['ProductCode'].apply(match_info)
        full_data['HighTech_Category'] = [x[0] for x in matched_results]
        full_data['Matched_SubCode'] = [x[1] for x in matched_results]

        # 过滤出成功匹配的高新产品数据
        filtered_data = full_data[full_data['HighTech_Category'].notnull()].copy()

        # 统计 A: 每年高新产品总计
        yearly_total = filtered_data.groupby('Year')[col_val].sum().reset_index()
        yearly_total.rename(columns={col_val: 'Total_All_HighTech'}, inplace=True)

        # 统计 B: 每年各大类
        yearly_cat_total = filtered_data.groupby(['Year', 'HighTech_Category'])[col_val].sum().reset_index()
        pivot_cat = yearly_cat_total.pivot(index='Year', columns='HighTech_Category', values=col_val).fillna(0)
        pivot_cat.columns = [f'MajorCat_{col}' for col in pivot_cat.columns]
        pivot_cat.reset_index(inplace=True)

        # 统计 C: 每年细分小类
        yearly_sub_total = filtered_data.groupby(['Year', 'Matched_SubCode'])[col_val].sum().reset_index()
        pivot_sub = yearly_sub_total.pivot(index='Year', columns='Matched_SubCode', values=col_val).fillna(0)
        pivot_sub.columns = [f'SubCode_{col}' for col in pivot_sub.columns]
        pivot_sub.reset_index(inplace=True)

        # ================= 合并超级大表 =================
        # 首先合并年份基准表（加入新增的全体商品总额）
        final_report = pd.merge(yearly_absolute_total, yearly_total, on='Year', how='outer')
        # 合并各大类
        final_report = pd.merge(final_report, pivot_cat, on='Year', how='outer')
        # 合并细分小类
        final_report = pd.merge(final_report, pivot_sub, on='Year', how='outer')

        # 排序
        final_report.sort_values(by='Year', inplace=True)

        # 保存结果
        output_file = os.path.join(folder_path, "HighTech_Detailed_Summary_Perfect.csv")
        final_report.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 完美生成！不仅包含高新技术数据，还加入了【全部商品年度出口总额】。结果已汇总至: {output_file}")