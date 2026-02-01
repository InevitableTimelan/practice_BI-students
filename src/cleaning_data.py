# -*- coding: utf-8 -*-
"""
BI学生数据分析项目 - 主脚本
本地运行版本
安徽大学 23级互联网金融学生 
github：InevitableTimelan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import re
import warnings
import traceback
warnings.filterwarnings("ignore")

# 设置中文字体和图形样式

plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

#数据读取

def load_data(file_path = '../data/raw/bi.csv'):
    """读取本地文件"""
    try:
        df = pd.read_csv(file_path,encoding='gbk')
        print(f"成功读取数据：{df.shape[0]}行，{df.shape[1]}列")
        return df
    except Exception as e:
        print(f"读取失败：{e}")
        print("请保证文件存在，或者文件路径正确！")
        return None

#数据清洗

#清洗混乱字符

def clean_name(name):
    if pd.isna(name):
        return name
    else:
        cleaned = re.sub(r'[^a-zA-Z\s\-\'\.]','',str(name))
        return cleaned.strip()

def clean_student_data(df):
    """清洗bi学生数据"""
    df_clean = df.copy()
    cleaning_log = []
    
    print("开始数据清洗：")
    
    print("查看原始数据信息：")
    print(f"原始数据形状：{df_clean.shape}")
    print(f"列名：{list(df_clean.columns)}")
    
    df_clean.columns = [col.strip().lower().replace(' ','_') for col in df_clean.columns]
    print("列名已标准化!")
    
    #1.处理国家名称不一致
    #因为数据量小，故使用字典映射的方法(以下都采用此映射)
    
    if 'country' in df_clean.columns:
        country_mapping = {
                   'norge': 'norway',
                   'rsa': 'south africa', 
                   'uk': 'united kingdom',
                   'usa': 'united states',
                   'us': 'united states',
                   'somali':'somali',
                   'uganda': 'uganda',
                   'south africa': 'south africa',
                   'netherlands': 'netherlands',
                   'denmark': 'denmark',
                   'italy': 'italy',
                   'spain': 'spain',
                   'nigeria': 'nigeria',
                   'germany': 'germany',
                   'france': 'france',
                   }
        df_clean['country'] = df_clean['country'].str.lower().map(lambda x: country_mapping.get(str(x).strip(), x) if pd.notnull(x) else x)
        cleaning_log.append("国家名称已标准化")
    
    #2.统一居住地类型
    
    if 'residence' in df_clean.columns:
        residence_mapping = {
            'bi_residence': 'bi-residence',
            'biresidence': 'bi-residence',
            'bi residence': 'bi-residence',
            'bi-residence': 'bi-residence',
            'sognsvann': 'sognsvann',
            'private': 'private',
            }
        df_clean['residence'] = df_clean['residence'].str.lower().map(lambda x: residence_mapping.get(str(x).strip(), x) if pd.notnull(x) else x)
        cleaning_log.append("居住地已统一")
    
    #3.处理教育背景
    
    if 'preveducation' in df_clean.columns:
        preveducation_mapping = {
            'masters': 'masters',
            'diploma': 'diploma',
            'highschool': 'highschool',
            'high school': 'highschool',
            'bachelors': 'bachelors',
            'barrrchelors': 'bachelors',
            'diplomaaa': 'diploma',
            'doctorate': 'doctorate',
            }
       
        df_clean['preveducation'] = df_clean['preveducation'].str.lower().map(lambda x: preveducation_mapping.get(str(x).strip(), x) if pd.notnull(x) else x)
        cleaning_log.append("教育背景已统一")
    
    #4.处理性别
    
    if 'gender' in df_clean.columns:
        gender_mapping = {
            'female': 'female',
            'm': 'male',
            'male': 'male',
            'f': 'female',
            }
        df_clean['gender'] = df_clean['gender'].str.lower().map(lambda x: gender_mapping.get(str(x).strip(),x) if pd.notnull(x) else x)
        cleaning_log.append("性别已统一")
    
    #5.处理入学成绩
    
    if 'entryexam' in df_clean.columns:
        df_clean['entryexam'] = pd.to_numeric(df_clean['entryexam'],errors='coerce')
        invalid_exam = (df_clean['entryexam']<0)|(df_clean['entryexam']>100) #检查索引
        if invalid_exam.any():
            cleaning_log.append(f"entryexam列：发现{invalid_exam.sum()}个异常值")
    
    #6.处理分数列——Python、DB
    
    score_columns = ['python','db']
    for col in score_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col],errors='coerce')
            invalid_score = (df_clean[col]<0)|(df_clean[col]>100)
            if invalid_score.any():
                cleaning_log.append(f"{col}列：发现{invalid_score.sum()}个异常值")
            
            #缺失值处理
            
            missing_score = df_clean[col].isnull().sum()
            if missing_score > 0:
                median_score = df_clean[col].median()
                df_clean[col].fillna(median_score, inplace=True)
                cleaning_log.append(f"{col}: 用中位数{median_score}填充{missing_score}个缺失值")
    
    #7.处理学习时长
    
    if 'studyhours' in df_clean.columns:
        df_clean['studyhours'] = pd.to_numeric(df_clean['studyhours'],errors='coerce')
        cleaning_log.append("studyhours已转化为数值类型")
    
    #8.处理名字列
    
    if 'fname' in df_clean.columns:
        df_clean['fname'] = df_clean['fname'].apply(clean_name)
        cleaning_log.append("清洗fname中的乱码字符")
    
    if 'lname' in df_clean.columns:
        df_clean['lname'] = df_clean['lname'].apply(clean_name)
        cleaning_log.append("清洗lname中的乱码字符")
    
    #9.创建综合指标
    
    if all(col in df_clean.columns for col in ['python', 'db']):
        df_clean['total_score'] = df_clean['python'] + df_clean['db']
        df_clean['average_score'] = df_clean['total_score'] / 2
        
        # 根据平均分划分风险等级
        
        df_clean['risk_level'] = pd.cut(df_clean['average_score'], 
                                  bins=[0, 60, 75, 100],
                                  labels=['high_risk', 'medium_risk', 'low_risk'])
        cleaning_log.append("创建综合成绩指标和风险等级")
    
    #输出清洗日志
    
    print("\n清洗完成记录:")
    for i, log in enumerate(cleaning_log, 1):
        print(f"  {i}. {log}")
    
    #数据质量检查
    
    print("\n数据质量检查:")
    print(f"数据集形状: {df_clean.shape}")
    print(f"列名: {list(df_clean.columns)}")
    
    #检查缺失值
    
    missing_values = df_clean.isnull().sum()
    if missing_values.sum() > 0:
       print("\n缺失值统计:")
       missing_df = missing_values[missing_values > 0]
       for col, count in missing_df.items():
           percentage = (count / len(df_clean)) * 100
           print(f"  {col}: {count} 个 ({percentage:.1f}%)")
    else:
       print("✅ 无缺失值")
    
    return df_clean

# ========主函数========

def main():
    """主函数"""
    print("="*60)
    print("BI学生数据清洗项目")
    print("="*60)
    
    df_raw = load_data('../data/raw/bi.csv')
    
    if df_raw is None:
        # 尝试其他可能的位置
        print("尝试从当前目录读取...")
        df_raw = load_data('bi.csv')
        if df_raw is None:
            print("无法读取数据，请检查文件路径")
            return None
    
    #数据清洗
    
    df_clean = clean_student_data(df_raw)
    
    #数据保存
    
    import os
    os.makedirs('../data/processed', exist_ok=True)
   
    output_path = '../data/processed/cleaned_student_data.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"\n清洗后的数据已保存为: {output_path}")
    
    #显示清洗后的数据样本
    
    print("\n清洗后数据示例 (前5行):")
    print(df_clean.head())
    
    #基本的统计信息
    
    print("\n数值字段基本统计:")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df_clean[numeric_cols].describe().round(2))
    
    print("数据清洗完成！")
    
if __name__ == "__main__":
    main()
    