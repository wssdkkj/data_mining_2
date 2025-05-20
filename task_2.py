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

all_category = ['电子产品', '服装', '食品', '家居', '办公', '运动户外', '玩具', '母婴', '汽车用品']

# 从子类到父类的映射
product_to_category = {item: category for category, items in category_mapping.items() for item in items}

def load_catalog(file):
    with open(file, 'r', encoding='utf-8') as f:
        product_data = json.load(f)

    # 创建产品DataFrame
    products_df = pd.DataFrame(product_data['products'])

    return products_df

products = load_catalog('./product_catalog.json')

id_to_product = {row['id']: row['category'] for _, row in products.iterrows()}  # 从ID到子类的映射
id_to_price = {row['id']: row['price'] for _, row in products.iterrows()}  # 从ID到价格的映射

catalog_list = []
high_value_products = []
payment_methods = []

def generate_payment_data(data):
    for _, row in data.iterrows():
        category_set = set()
        purchase_data = json.loads(row['purchase_history'])
        payment = purchase_data['payment_method']
        for item in purchase_data['items']:
            product_id = item['id']
            if product_id in id_to_product:
                category = product_to_category[id_to_product[product_id]]
                category_set.add(category)

                # 记录高价值商品
                price = id_to_price[product_id]
                if price > 5000:
                    product_name = id_to_product[product_id]
                    if product_name not in high_value_products:
                        high_value_products.append(product_name)
        category_set.add(payment)
        category_list = sorted(category_set)
        if category_list not in catalog_list:
            catalog_list.append(category_list)
        if payment not in payment_methods:
            payment_methods.append(payment)

def load_history(folder):
    parquet_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.parquet')]
    for file in tqdm(parquet_files,total=len(parquet_files), desc="Generating records"):
        part = pd.read_parquet(file, engine='fastparquet', columns=['purchase_history'])
        generate_payment_data(part)
        del part
        gc.collect()

# print(catalog_list[:10])
# print(high_value_products[:5])
# print(payment_methods[:5])

def association_analysis(input_list,high_value_list,payment_list):
    start_time = time.time()

    # 使用 TransactionEncoder 进行热编码
    te = TransactionEncoder()
    X = te.fit_transform(input_list)
    df_payment = pd.DataFrame(X, columns=te.columns_)

    # 使用 Apriori 算法生成频繁项集
    frequent_itemsets_payment = apriori(df_payment, min_support=0.01, use_colnames=True, low_memory=True)

    # print('频繁项集:')
    # print(frequent_itemsets_payment)

    frequent_itemsets_payment.to_csv('./result/frequent_itemsets_payment.csv', index=False)

    # 生成关联规则
    rules_payment = association_rules(frequent_itemsets_payment, metric="confidence", min_threshold=0.1)

    # 计算提升度
    rules_payment['lift'] = rules_payment['confidence'] / rules_payment['antecedent support']

    # 只保留包含支付方式的规则
    rules_payment = rules_payment[
        rules_payment['antecedents'].apply(lambda x: all(item in all_category for item in x)) &
        rules_payment['consequents'].apply(lambda x: all(item in payment_list for item in x))
    ]

    # 输出结果
    # print('支付方式与商品类别的关联规则:')
    # print(rules_payment[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    rules_payment.to_csv('./result/rules_payment.csv', index=False)

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 选择要绘制的列
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rules_payment, x='support', y='confidence', size='lift', sizes=(20, 200), alpha=0.6, hue='lift', palette='coolwarm')
    plt.title('Association Rules: Support vs Confidence')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.legend(title='Lift')
    plt.savefig('./result/rules_payment.png')

    # 分析高价值商品的首选支付方式
    high_value_payment_methods = {}

    for product in high_value_list:
        category = product_to_category[product]
        category_rules = rules_payment[rules_payment['antecedents'].apply(lambda x: category in x)]

        if not category_rules.empty:
            highest_confidence_rule = category_rules.loc[category_rules['confidence'].idxmax()]
            method = highest_confidence_rule['consequents']
            high_value_payment_methods[product] = method

    # 输出高价值商品的首选支付方式
    # print('高价值商品的首选支付方式:')
    # for product, payment_method in high_value_payment_methods.items():
    #     print(f'商品: {product}, 首选支付方式: {payment_method}')

    # 输出高价值商品的首选支付方式并保存到CSV
    high_value_payment_df = pd.DataFrame(high_value_payment_methods.items(), columns=['category', 'preferred_payment_method'])
    high_value_payment_df.to_csv('./result/high_value_payment_methods.csv', index=False)

    # 找到置信度 ≥ 0.6 的规则子集
    high_confidence_rules = rules_payment[rules_payment['confidence'] >= 0.6]
    # print('置信度 ≥ 0.6 的规则子集:')
    # print(high_confidence_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    high_confidence_rules.to_csv('./result/rules_payment_0.6_confidence.csv', index=False)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f'Payment Rule Mining time: {elapsed_time:.2f} seconds')  # 打印运行时间

if __name__ == "__main__":
    load_history('./30G_data_new')

    catalog_list = [list(x) for x in set(tuple(record) for record in catalog_list)]

    association_analysis(catalog_list,high_value_products,payment_methods)