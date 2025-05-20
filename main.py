import pandas as pd
import os
import gc
import json
import time
from datetime import datetime
from tqdm import tqdm
from task_1 import association_rule_mining
from task_2 import association_analysis
from task_3 import time_series_analysis
from task_4 import refund_pattern
import matplotlib.pyplot as plt
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

def load_catalog(file): #读取商品与类别的对照json文件
    with open(file, 'r', encoding='utf-8') as f:
        product_data = json.load(f)

    # 创建产品DataFrame
    products_df = pd.DataFrame(product_data['products'])

    return products_df

products = load_catalog('./product_catalog.json')

id_to_product = {row['id']: row['category'] for _, row in products.iterrows()} # 从ID到商品的映射
id_to_price = {row['id']: row['price'] for _, row in products.iterrows()} # 从ID到价格的映射

category_list = [] #用于作为任务目标1探索商品类别关系模式的输入
payment_list = [] #用于作为任务目标2探索支付方式关系模式的输入
high_value_products = [] #用于记录高价值商品（任务2）
payment_methods = [] #用于记录支付方式（任务2）
dataframe = [] #用于作为任务目标3的时间序列输入
date_list = [] #用于作为任务目标3探索购买先后关系模式的输入
status_list = [] #用于作为任务目标4探索造成退款的商品组合模式的输入

def set_to_list(set, list): #将集合转换为列表并保证不包含重复值
    sorted_list = sorted(set)
    if sorted_list not in list:
        list.append(sorted_list)

def generate_data(data):
    for _, row in data.iterrows():
        category_set = set() #类别集合
        payment_set = set() #支付方式集合
        status_set = set() #退款集合
        purchase_data = json.loads(row['purchase_history']) #读取json格式的记录
        payment = purchase_data['payment_method'] #支付方式
        date = purchase_data['purchase_date'] #交易日期
        payment_status = purchase_data['payment_status'] #交易状态
        item_count = len(purchase_data['items']) #商品计数
        for item in purchase_data['items']: #遍历记录中的id集合
            product_id = item['id']
            if product_id in id_to_product:
                category = product_to_category[id_to_product[product_id]]
                category_set.add(category) #记录任务1的项集
                payment_set.add(category) #记录任务2的项集

                # 记录高价值商品
                price = id_to_price[product_id]
                if price > 5000:
                    product_name = id_to_product[product_id]
                    if product_name not in high_value_products:
                        high_value_products.append(product_name)

                #记录任务3的时间序列
                dataframe.append(
                    {
                        'purchase_date': datetime.strptime(date, '%Y-%m-%d'),
                        'categories': category,
                        'item_count': item_count
                    }
                )

                #记录任务4的项集
                if payment_status in ['已退款', '部分退款']:
                    status_set.add(category)

        #将集合转换为列表
        set_to_list(category_set,category_list)
        payment_set.add(payment)
        set_to_list(payment_set,payment_list)
        set_to_list(status_set,status_list)

        if payment not in payment_methods:
            payment_methods.append(payment)

def load_data(folder):
    parquet_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.parquet')]
    for file in tqdm(parquet_files, total=len(parquet_files), desc="Generating records"):
        part = pd.read_parquet(file, engine='fastparquet', columns=['purchase_history'])
        generate_data(part)
        del part
        gc.collect()

if __name__ == "__main__":
    start_time = time.time()

    folder = './30G_data_new'
    load_data(folder)

    df = pd.DataFrame(dataframe)

    df = df.drop_duplicates()

    #任务1
    association_rule_mining(category_list)
    del category_list
    gc.collect()

    #任务2
    association_analysis(payment_list,high_value_products,payment_methods)
    del payment_list, high_value_products, payment_methods
    gc.collect()

    #任务3
    time_series_analysis(df)
    del df
    gc.collect()

    #任务4
    refund_pattern(status_list)
    del status_list
    gc.collect()

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f'Total time: {elapsed_time:.2f} seconds')  # 打印运行时间