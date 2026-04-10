import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

def prepare_datasets(excel_file, data_dir='data', force_split=False):
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    info_path = os.path.join(data_dir, 'dataset_info.json')

    if not force_split and os.path.exists(train_path) and os.path.exists(val_path):
        print("加载已划分的数据集...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                dataset_info = json.load(f)
            print(f"数据集信息: {dataset_info}")
        return train_df, val_df

    df = pd.read_excel(excel_file)

    labels = df.iloc[:, 0].values
    train_indices, val_indices = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=labels
    )

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

    def analyze_class_distribution(subset_df, subset_name):
        class_counts = subset_df.iloc[:, 0].value_counts()
        total_samples = len(subset_df)
        print(f"\n{subset_name}类别分布:")
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {class_name}: {count}个样本 ({percentage:.2f}%)")

    analyze_class_distribution(train_df, "训练集")
    analyze_class_distribution(val_df, "验证集")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    dataset_info = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'train_classes': train_df.iloc[:, 0].unique().tolist(),
        'val_classes': val_df.iloc[:, 0].unique().tolist()
    }
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    return train_df, val_df