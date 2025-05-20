import pandas as pd
import os
import gc
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义商品类别映射
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

# 商品类别列表，便于后续直接使用
all_category = ['电子产品','服装','食品','家居','办公','运动户外','玩具','母婴','汽车用品']

# 从商品到类别的映射
product_to_category = {item: category for category, items in category_mapping.items() for item in items}

def load_catalog(file):
    with open(file, 'r', encoding='utf-8') as f:
        product_data = json.load(f)

    # 创建产品DataFrame
    products_df = pd.DataFrame(product_data['products'])

    return products_df

products = load_catalog('./product_catalog.json')

id_to_product = {row['id']: row['category'] for _, row in products.iterrows()} # 从ID到商品的映射

catalog_list = []

def generate_categories(data):
    for _, row in data.iterrows():
        purchase_data = json.loads(row['purchase_history'])
        yield {product_to_category[id_to_product[item['id']]] for item in purchase_data['items'] if
               item['id'] in id_to_product}

def load_history(folder):
    parquet_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.parquet')]
    for file in parquet_files:
        print('Loading {}'.format(file))
        part = pd.read_parquet(file, engine='fastparquet', columns=['purchase_history'])
        for category_ids in tqdm(generate_categories(part), total=part.shape[0], desc="Generating records"):
            catalog_list.append(list(category_ids))  # 将生成器表达式转换为列表
        del part
        gc.collect()

# print(catalog_list[:10])

def association_rule_mining(input_list):
    start_time = time.time()

    te = TransactionEncoder()
    X = te.fit_transform(input_list)
    cols = te.columns_
    df = pd.DataFrame(X, columns=cols)

    # 使用 Apriori 算法生成频繁项集
    frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True,low_memory=True)

    # print('频繁项集:')
    # print(frequent_itemsets)

    frequent_itemsets.to_csv('./result/frequent_itemsets_catalog.csv', index=False)

    # 生成关联规则
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    # 计算提升度
    rules['lift'] = rules['confidence'] / rules['antecedent support']

    # print('关联规则:')
    # print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    rules.to_csv('./result/rules_catalog.csv', index=False)

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 选择要绘制的列
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rules, x='support', y='confidence', size='lift', sizes=(20, 200), alpha=0.6, hue='lift', palette='coolwarm')
    plt.title('Association Rules: Support vs Confidence')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.legend(title='Lift')
    plt.savefig('./result/rules.png')

    # 特别关注电子产品与其他类别的关联关系
    electronic_rules = rules[rules['antecedents'].apply(
        lambda x: any(item == '电子产品' for item in x)) |
        rules['consequents'].apply(
            lambda x: any(item == '电子产品' for item in x))
    ]

    # 输出结果
    # print('电子产品相关的关联规则:')
    # print(electronic_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    electronic_rules.to_csv('./result/electronic_rules_catalog.csv', index=False)

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 选择要绘制的列
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=electronic_rules, x='support', y='confidence', size='lift', sizes=(20, 200), alpha=0.6, hue='lift', palette='coolwarm')
    plt.title('Association Rules: Support vs Confidence')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.legend(title='Lift')
    plt.savefig('./result/electronic_rules.png')

    # 找到置信度 ≥ 0.5 的规则子集
    high_confidence_rules = rules[rules['confidence'] >= 0.5]
    # print('置信度 ≥ 0.5 的规则子集:')
    # print(high_confidence_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    high_confidence_rules.to_csv('./result/rules_catalog_0.5_confidence.csv', index=False)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f'Categories Rule Mining time: {elapsed_time:.2f} seconds')  # 打印运行时间

if __name__ == "__main__":
    load_history('./30G_data_new')

    catalog_list = [list(x) for x in set(tuple(record) for record in catalog_list)]

    association_rule_mining(catalog_list)