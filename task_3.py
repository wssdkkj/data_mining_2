import pandas as pd
import os
import gc
import time
import json
from tqdm import tqdm
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

dataframe = []

# 定义产品类别映射
category_mapping = {
    '电子产品': ['智能手机', '笔记本电脑', '平板电脑', '智能手表', '耳机', '音响', '相机', '摄像机', '游戏机'],
    '服装': ['上衣', '裤子', '裙子', '内衣', '鞋子', '帽子', '手套', '围巾', '外套'],
    '食品': ['零食', '饮料', '调味品', '米面', '水产', '肉类', '蛋奶', '水果', '蔬菜'],
    '家居': ['家具', '床上用品', '厨具', '卫浴用品'],
    '办公': ['文具', '办公用品'],
    '运动户外': ['健身器材', '户外装备'],
    '玩具': ['玩具', '模型', '益智玩具'],
    '母婴': ['婴儿用品', '儿童课外读物'],
    '汽车用品': ['车载电子', '汽车装饰']
}

# 从子类到父类的映射
product_to_category = {item: category for category, items in category_mapping.items() for item in items}

def load_catalog(file):
    with open(file, 'r', encoding='utf-8') as f:
        product_data = json.load(f)

    # 创建产品DataFrame
    products_df = pd.DataFrame(product_data['products'])

    return products_df

products = load_catalog('./product_catalog.json')

# 从ID到商品的映射
id_to_product = {row['id']: row['category'] for _, row in products.iterrows()}

def generate_frequency(data):
    for _, row in data.iterrows():
        purchase_data = json.loads(row['purchase_history'])
        date = purchase_data['purchase_date']
        item_count = len(purchase_data['items'])

        for item in purchase_data['items']:
            product_id = item['id']
            if product_id in id_to_product:
                category = product_to_category[id_to_product[product_id]]
                dataframe.append(
                    {
                        'purchase_date': datetime.strptime(date, '%Y-%m-%d'),
                        'categories': category,
                        'item_count': item_count
                    }
                )

def load_history(folder):
    parquet_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.parquet')]
    for file in tqdm(parquet_files,total=len(parquet_files), desc="Generating records"):
        part = pd.read_parquet(file, engine='fastparquet', columns=['purchase_history'])
        generate_frequency(part)
        del part
        gc.collect()

# print(df.head(10))

def time_series_analysis(series):
    start_time = time.time()

    total_purchases = series.groupby(series['purchase_date'].dt.to_period('W'))['item_count'].sum().reset_index(name='total_purchases')
    total_purchases['purchase_date'] = total_purchases['purchase_date'].dt.to_timestamp()
    # print('每周被购买的商品总数变化:')
    # print(total_purchases)

    # 绘制每种类别的折线图
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=total_purchases, x='purchase_date', y='total_purchases', marker='o')
    plt.title('Changes in the total number of products purchased per week')
    plt.xlabel('week')
    plt.ylabel('purchase count')
    plt.xticks(rotation=45)
    plt.savefig('./result/week_frequency.png')

    total_purchases = series.groupby(series['purchase_date'].dt.to_period('M'))['item_count'].sum().reset_index(name='total_purchases')
    total_purchases['purchase_date'] = total_purchases['purchase_date'].dt.to_timestamp()
    # print('每月被购买的商品总数变化:')
    # print(total_purchases)

    # 绘制每种类别的折线图
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=total_purchases, x='purchase_date', y='total_purchases', marker='o')
    plt.title('Changes in the total number of products purchased per month')
    plt.xlabel('month')
    plt.ylabel('purchase count')
    plt.xticks(rotation=45)
    plt.savefig('./result/month_frequency.png')

    total_purchases = series.groupby(series['purchase_date'].dt.to_period('Q'))['item_count'].sum().reset_index(name='total_purchases')
    total_purchases['purchase_date'] = total_purchases['purchase_date'].dt.to_timestamp()
    # print('每季度被购买的商品总数变化:')
    # print(total_purchases)

    # 绘制每种类别的折线图
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=total_purchases, x='purchase_date', y='total_purchases', marker='o')
    plt.title('Changes in the total number of products purchased per quarter')
    plt.xlabel('quarter')
    plt.ylabel('purchase count')
    plt.xticks(rotation=45)
    plt.savefig('./result/quarter_frequency.png')

    # 2. 每种商品类别在不同时间段内被购买的数量变化
    category_counts = series.groupby(['categories', series['purchase_date'].dt.to_period('M')])['item_count'].sum().reset_index(name='purchase_count')
    # 转换数据类型
    category_counts['purchase_date'] = category_counts['purchase_date'].dt.to_timestamp()
    category_counts['purchase_count'] = pd.to_numeric(category_counts['purchase_count'], errors='coerce')
    # print('每种商品类别在不同时间段内被购买的数量变化:')
    # print(category_counts)
    # 绘制每种类别的折线图
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=category_counts, x='purchase_date', y='purchase_count', hue='categories',marker='o')
    plt.title('Changes in the quantity of each product category purchased during different time periods')
    plt.xlabel('month')
    plt.ylabel('purchase count')
    plt.xticks(rotation=45)
    plt.legend(title='categories')
    plt.savefig('./result/category_frequency.png')

    # 3. 探索先购买A类别，后购买B类别的时序模式
    # 按时间排序
    series = series.sort_values(by='purchase_date')

    transactions = []

    # 生成类别组合
    for _, group in series.groupby(pd.Grouper(key='purchase_date', freq='M')):
        transactions.append(list(group['categories']))

    # print(len(transactions))

    # 转换为适合 Apriori 的格式
    te = TransactionEncoder()
    X = te.fit_transform(transactions)
    df_transactions = pd.DataFrame(X, columns=te.columns_)

    # print(df_transactions.head(10))

    # 使用 Apriori 算法生成频繁项集
    frequent_itemsets_date = apriori(df_transactions, min_support=0.01, use_colnames=True, low_memory=True)

    # 生成关联规则
    sequential_rules = association_rules(frequent_itemsets_date, metric="confidence", min_threshold=0.1)

    # 计算提升度
    sequential_rules['lift'] = sequential_rules['confidence'] / sequential_rules['antecedent support']

    # 输出结果
    # print('购买类别先后顺序的时序模式:')
    # print(sequential_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    sequential_rules.to_csv('./result/rules_sequential.csv', index=False)

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 选择要绘制的列
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sequential_rules, x='support', y='confidence', size='lift', sizes=(20, 200), alpha=0.6, hue='lift', palette='coolwarm')
    plt.title('Association Rules: Support vs Confidence')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.legend(title='Lift')
    plt.savefig('./result/rules_sequential.png')

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f'Series Pattern Mining time: {elapsed_time:.2f} seconds')  # 打印运行时间

if __name__ == "__main__":
    load_history('./30G_data_new')

    df = pd.DataFrame(dataframe)

    # 去除重复值
    df = df.drop_duplicates()

    time_series_analysis(df)