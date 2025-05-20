import pandas as pd
import os
import time
import json
import gc

def load_data(folder_path):
    start_time = time.time()

    dataframes = []
    parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]
    for file in parquet_files:
        print('Loading {}'.format(file))
        part = pd.read_parquet(file,engine='fastparquet',columns=['purchase_history'])
        dataframes.append(part)
        del part
        gc.collect()
    df = pd.concat(dataframes, ignore_index=True)

    # 加载商品信息
    print('Loading ./product_catalog.json')

    with open('./product_catalog.json', 'r', encoding='utf-8') as f:
        product_data = json.load(f)

    # 创建产品DataFrame
    products_df = pd.DataFrame(product_data['products'])

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f'Load data time: {elapsed_time:.2f} seconds')  # 打印运行时间

    return df,products_df

if __name__ == '__main__':
    df = load_data('./30G_data_new')
    output_file = './output_data_30.parquet'
    df.to_parquet(output_file, index=False)
    print(f'DataFrame 已保存为 {output_file}')