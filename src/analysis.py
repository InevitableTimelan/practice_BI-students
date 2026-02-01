# -*- coding: utf-8 -*-
"""
BI学生数据分析脚本
回答四个核心业务问题:
1. Admissions Optimization
Should entry exams remain the primary admissions filter?

Your task is to evaluate the predictive power of entry exam scores compared to other features such as prior education, age, gender, and study hours.

✅ Deliverables:

Feature importance ranking for predicting Python and DB scores
Admission policy recommendation (e.g., retain exams, add screening tools, adjust thresholds)
Business rationale and risk analysis

# 需要分析的内容：
# 1. 入学考试与成绩的相关性（相关系数矩阵）
# 2. 入学考试的预测能力（线性回归/R²分数）
# 3. 特征重要性排名（随机森林/特征重要性）
# 4. 与其他特征（教育背景、年龄、学习时长）的比较

# 机器学习应用：线性回归、随机森林回归
# 评估指标：R²分数、MSE、特征重要性

2. Curriculum Support Strategy
Are there at-risk student groups who need extra support?

Your task is to uncover whether certain backgrounds (e.g., prior education level, country, residence type) correlate with poor performance and recommend targeted interventions.

✅ Deliverables:

At-risk segment identification
Support program design (e.g., prep course, mentoring)
Expected outcomes, costs, and KPIs

# 需要分析的内容：
# 1. 识别高风险学生（平均分<60分）
# 2. 高风险学生的特征分析（分组统计）
# 3. 哪些背景因素与低成绩相关（卡方检验/相关性分析）

# 机器学习应用：逻辑回归分类、决策树
# 评估指标：准确率、召回率、F1分数

3. Resource Allocation & Program ROI

How can we allocate resources for maximum student success?

Your task is to segment students by success profiles and suggest differentiated teaching/facility strategies.

✅ Deliverables:

Performance drivers
Student segmentation
Resource allocation plan and ROI projection      

# 需要分析的内容：
# 1. 学生细分（聚类分析）
# 2. 不同群体的特征分析
# 3. ROI计算模型（成本效益分析）

# 机器学习应用：K-means聚类
# 评估指标：轮廓系数、聚类质量
 
Bonus Challenge

“If you could implement only one intervention to improve student outcomes, what would it be — and why?”

# 综合以上分析，提出一个最有效的干预措施
# 用数据支持选择
# 提供ROI预测

2026年2月
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score,classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,LabelEncoder
import warnings
warnings.filterwarnings("ignore")

#设置中文字体和图形样式

plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

#颜色方案

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

#读取数据

def load_data(filepath='../data/processed/cleaned_student_data.csv'):
    """读取本地文件"""
    try:
        df = pd.read_csv(filepath,encoding='gbk')
        print(f"成功读取数据：{df.shape[0]}行，{df.shape[1]}列")
        return df
    except Exception as e:
        print(f"读取失败：{e}")
        return None

#数据分析

def exploration_data_analysis(df):
    """基础性数据分析"""
    print("\n" + "="*60)
    print("探索性数据分析 (EDA)")
    print("="*60)
    
    #数据概况
    
    print(f"数据形状：{df.shape}")
    print(f"\n列名: {list(df.columns)}")
    print(f"\n数据类型:\n{df.dtypes}")
    
    #描述性统计
    
    print("描述性统计：")
    print(df.describe().round(2))
    
    #缺失值检查
    
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
       print("\n缺失值统计:")
       missing_df = missing_values[missing_values > 0]
       for col, count in missing_df.items():
           percentage = (count / len(df)) * 100
           print(f"  {col}: {count} 个 ({percentage:.1f}%)")
    else:
       print("✅ 无缺失值")
    
    #关键指标分布
    
    print("关键指标分布：")
    key = ['python','db','total_score','average_score','studyhours','entryexam']
    for col in key:
        print(f"{col}: 均值={df[col].mean():.1f}, 标准差={df[col].std():.1f}, "f"最小值={df[col].min():.1f}, 最大值={df[col].max():.1f}")
    
    return df

# ====== 问题1：招生优化 ======

def missions_recruit(df):
    """
    问题1: Should entry exams remain the primary admissions filter?
    分析入学考试作为主要录取筛选工具的合理性
    """
    
    print("\n" + "="*60)
    print("问题1: 招生优化分析")
    print("="*60)
    
    results = {}
    
    #1.相关系数分析
    
    print("\n相关系数分析 (入学考试与成绩的关系):")
    
    corr_features = ['entryexam', 'python', 'db', 'average_score', 
                     'studyhours', 'age']
    available_features = [col for col in corr_features if col in df.columns]
    
    if len(available_features) >= 2:
        corr_matrix = df[available_features].corr()
        print("\n相关系数矩阵：")
        print(corr_matrix.round(3))
    
    #可视化相关系数热图
    
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',center=0,fmt='.2f',square=True,linewidths=0.5)
    plt.title("特征相关系数热力图")
    plt.tight_layout()
    plt.savefig('../visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("相关系数热图已保存")
    
    #2.入学考试与平均成绩的散点图
    
    if all(col in df.columns for col in ['entryexam', 'average_score']):
        plt.figure(figsize=(12,5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df['entryexam'], df['average_score'], alpha=0.6, color=COLORS[0])
        plt.xlabel('入学成绩')
        plt.ylabel('平均成绩')
        plt.title('入学考试 vs 平均成绩')
        
        #添加回归线
        
        z = np.polyfit(df['entryexam'], df['average_score'], 1)
        p = np.poly1d(z)
        plt.plot(df['entryexam'], p(df['entryexam']), "r--", alpha=0.8)
        
        #计算R²
        
        r2 = r2_score(df['average_score'], p(df['entryexam']))
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        
        plt.subplot(1, 2, 2)
        
        # 入学考试成绩分布
        
        plt.hist(df['entryexam'], bins=20, color=COLORS[1], edgecolor='black', alpha=0.7)
        plt.xlabel('入学考试成绩')
        plt.ylabel('频次')
        plt.title('入学考试成绩分布')
        
        plt.tight_layout()
        plt.savefig('../visualizations/entry_exam_analysis.png', dpi=300, bbox_inches='tight')
        print("\入学考试分析图表已保存")
        
        results['entryexam_r2'] = r2
        
    #3.特征重要性分析（使用随机森林）
    
    print("\n特征重要性分析 (预测平均成绩):")
    
    #准备特征
    
    feature_cols = []
   
    if 'entryexam' in df.columns:
       feature_cols.append('entryexam')
    if 'age' in df.columns:
       feature_cols.append('age')
    if 'studyhours' in df.columns:
       feature_cols.append('studyhours')
    
    #添加编码后的分类特征
    
    categorical_cols = ['gender', 'preveducation', 'country', 'residence']
    for col in categorical_cols:
        if col in df.columns:
            # 使用标签编码
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            feature_cols.append(f'{col}_encoded')
    
    if len(feature_cols) >= 2 and 'average_score' in df.columns:
        X = df[feature_cols]
        y = df['average_score']
        
        # 训练随机森林模型
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # 特征重要性
        
        importance_df = pd.DataFrame({'feature': feature_cols,'importance': rf.feature_importances_ }).sort_values('importance', ascending=False)
        
        print("\n特征重要性排名:")
        print(importance_df)
        
        # 可视化特征重要性
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('特征重要性')
        plt.title('影响学生成绩的特征重要性')
        plt.gca().invert_yaxis()
        
        # 添加数值标签
        
        for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
            plt.text(importance + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{importance:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('../visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        print("✅ 特征重要性图表已保存")
        
        results['feature_importance'] = importance_df
        
    #4.多变量回归分析
    
    print("\n多变量回归分析:")
    if len(feature_cols) >= 2 and 'average_score' in df.columns:
        
       # 训练线性回归模型
       
       lr = LinearRegression()
       lr.fit(X, y)
       y_pred = lr.predict(X)
       
       # 评估指标
       
       r2 = r2_score(y, y_pred)
       mse = mean_squared_error(y, y_pred)
       
       print("多变量回归模型性能:")
       print(f"R²分数: {r2:.3f}")
       print(f"均方误差: {mse:.3f}")
       
       # 系数分析
       
       coefficients = pd.DataFrame({'feature': ['intercept'] + feature_cols,'coefficient': [lr.intercept_] + list(lr.coef_) })
       
       print("\n回归系数:")
       print(coefficients.round(3))
       
       results['regression_r2'] = r2
       results['regression_coefficients'] = coefficients
       
    # 5.招生政策建议
    
    print("\n 招生政策建议:")
    
    if 'feature_importance' in results:
        top_feature = results['feature_importance'].iloc[0]['feature']
        top_importance = results['feature_importance'].iloc[0]['importance']
        
        print(f"1.主要发现:最重要的预测特征是 '{top_feature}' (重要性: {top_importance:.3f})")
        
    if 'entryexam_r2' in results:
        entry_r2 = results['entryexam_r2']
        if entry_r2 > 0.5:
            print("2.入学考试有效性:入学考试成绩是成绩的强预测指标 (R² > 0.5)")
            print("建议:继续将入学考试作为主要筛选工具")
        elif entry_r2 > 0.3:
            print("2.入学考试有效性:入学考试成绩是成绩的中等预测指标 (0.3 < R² < 0.5)")
            print("建议:将入学考试与其他因素结合使用")
        else:
            print("2.入学考试有效性:入学考试成绩的预测能力较弱 (R² < 0.3)")
            print("建议:考虑补充其他筛选工具")
    
    print("3.推荐措施:")
    print(" -将入学考试与教育背景、学习时长等因素结合评估")
    print(" -为不同背景的学生设定差异化的录取标准")
    print(" -建立预测模型，提前识别有潜力的学生")
    
    return results

# ====== 问题2: 课程支持策略 ======

def missions_support(df):
    """
    问题2: Are there at-risk student groups who need extra support?
    识别需要额外支持的高风险学生群体
    """
    
    print("\n" + "="*60)
    print("问题2: 课程支持策略分析")
    print("="*60)
    
    results = {}
    
    #1.定义高风险学生 (平均分 < 60)
    
    if 'average_score' in df.columns:
       df['is_at_risk'] = df['average_score'] < 60
       at_risk_count = df['is_at_risk'].sum()
       at_risk_percentage = at_risk_count / len(df) * 100
    
    print("\n高风险学生统计：")
    print(f"高风险学生人数：{at_risk_count}人")
    print(f"高风险人数百分比：{at_risk_percentage:.1f}%")
    
    results['at_risk_count'] = at_risk_count
    results['at_risk_percentage'] = at_risk_percentage
    
    #2.高风险学生特征分析
    
    print("\n高风险学生特征分析：")
    
    #按性别分析
    
    if 'gender' in df.columns and 'is_at_risk' in df.columns:
        gender_risk = df.groupby('gender')['is_at_risk'].mean()*100
        print("\n按性别分布的风险比例：")
        for gender, risk in gender_risk.items():
            print(f"{gender}: {risk:.1f}%")
        
    #按教育背景分析
    
    if 'preveducation' in df.columns and 'is_at_risk' in df.columns:
        preveducation_risk = df.groupby('preveducation')['is_at_risk'].mean()*100
        print("\n按教育背景分布的风险比例：")
        for preveducation, risk in preveducation_risk.items():
            print(f"{preveducation}:{risk:.1f}%")
    
    #按居住地分析
    
    if 'residence' in df.columns and 'is_at_risk' in df.columns:
        residence_risk = df.groupby('residence')['is_at_risk'].mean()*100
        print("\n按居住地分布的风险比例：")
        for residence, risk in residence_risk.items():
            print(f"{residence}:{risk:.1f}%")
    
    # 3. 高风险学生预测模型
    
    print("\n高风险学生预测模型:")
    
    # 准备特征
    
    feature_cols = []
    if 'entryexam' in df.columns:
       feature_cols.append('entryexam')
    if 'age' in df.columns:
       feature_cols.append('age')
    if 'studyhours' in df.columns:
       feature_cols.append('studyhours')
    
    categorical_col = ['gender','preveducation','country','residence']
    for col in categorical_col:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            feature_cols.append(f'{col}_encoded')
    
    if len(feature_cols) >= 2 and 'is_at_risk' in df.columns:
        X = df[feature_cols]
        y = df['is_at_risk'].astype(int)
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        
        lr_clf = LogisticRegression(max_iter=1000,random_state=42)
        lr_clf.fit(X_train,y_train)
        y_pred = lr_clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"逻辑回归模型准确率: {accuracy:.3f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        
        rf_clf = RandomForestClassifier(n_estimators=100,random_state=42)
        rf_clf.fit(X_train,y_train)
        y_pred_rf = rf_clf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        
        print(f"\n随机森林模型准确率: {accuracy_rf:.3f}")
        print("随机森林特征重要性:")
        rf_importance = pd.DataFrame({'feature': feature_cols,'importance': rf_clf.feature_importances_}).sort_values('importance', ascending=False)
        print(rf_importance.round(3))
        
        results['lr_accuracy'] = accuracy
        results['rf_accuracy'] = accuracy_rf
        results['rf_feature_importance'] = rf_importance
        
    #4.支持计划设计
    print("\n💡 支持计划设计:")
    print("1. 目标群体: 平均成绩低于60分的学生")
    print("2. 支持措施:")
    print("   - 个性化辅导课程 (每周2小时)")
    print("   - 学习技巧工作坊")
    print("   - 同伴导师计划")
    print("   - 额外学习资源提供")
    print("3. 预期成果:")
    print("   - 高风险学生成绩提升20%")
    print("   - 高风险学生比例减少30%")
    print("4. 关键绩效指标 (KPIs):")
    print("   - 高风险学生平均成绩提升")
    print("   - 高风险学生数量减少")
    print("   - 学生满意度调查得分")
    
    return results

# ====== 问题3: 资源分配与ROI ======

def missions_allocation(df):
    """
    问题3: Resource Allocation & Program ROI
    学生细分与资源分配策略
    """
    
    print("\n" + "="*60)
    print("问题3:资源分配与ROI分析")
    print("="*60)
    
    results = {}
    
    #1.学生细分(聚类分析)
    
    print("\n学生细分分析 (聚类):")
    
    #选择聚类特征
    
    cluster_features = []
    if 'average_score' in df.columns:
        cluster_features.append('average_score')
    if 'entryexam' in df.columns:
        cluster_features.append('entryexam')
    if 'studyhours' in df.columns:
        cluster_features.append('studyhours')
        
    if len(cluster_features) >= 2:
        
        #标准化特征
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[cluster_features])
       
       #使用肘部法则确定最佳K值
       
        inertias = []
        K_range = range(1, 6)
        for k in K_range:
            kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        #可视化肘部法则
        
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('聚类数量 (K)')
        plt.ylabel('内聚力 (Inertia)')
        plt.title('肘部法则: 确定最佳聚类数量')
        plt.xticks(K_range)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../visualizations/elbow_method.png', dpi=300, bbox_inches='tight')
        
        #选择K=3 
        
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        #分析每个聚类
        
        cluster_analysis = df.groupby('cluster').agg({'average_score': 'mean','entryexam': 'mean','studyhours': 'mean','risk_level': lambda x: (x == 'high_risk').mean() * 100}).round(2)
        
        print(f"\n将学生分为{optimal_k}个群体:")
        print(cluster_analysis)
        
        cluster_names = {0: '表现优秀学生',1: '中等表现学生', 2: '需要支持学生'}
        df['segment'] = df['cluster'].map(cluster_names)
        
        #可视化聚类结果
        
        if len(cluster_features) >= 2:
            plt.figure(figsize=(10, 8))
            
            #选择前两个特征进行可视化
            
            x_feature, y_feature = cluster_features[0], cluster_features[1]
            
            scatter = plt.scatter(df[x_feature], df[y_feature], c=df['cluster'], cmap='viridis', alpha=0.7)
            plt.xlabel(x_feature)
            plt.ylabel(y_feature)
            plt.title('学生聚类可视化')
            plt.colorbar(scatter, label='聚类')
            
            #标记聚类中心
            
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
            plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X')
            
            plt.tight_layout()
            plt.savefig('../visualizations/cluster_analysis.png', dpi=300, bbox_inches='tight')
            print("聚类分析图表已保存")
        
        results['clusters'] = cluster_analysis
        results['cluster_centers'] = centers
        
    #2.资源分配策略
        
    print("\n资源分配策略:")
   
    if 'segment' in df.columns:
        segment_counts = df['segment'].value_counts()
       
        print("基于学生细分的结果分配:")
        for segment, count in segment_counts.items():
            percentage = count / len(df) * 100
            print(f"\n{segment} ({count}人, {percentage:.1f}%):")
           
            if '需要支持' in segment:
                print(" - 资源分配: 高投入 (40%总资源)")
                print(" - 支持措施: 个性化辅导、额外练习、学习小组")
                print(" - 预期ROI: 高 (每投入1元，预期回报2.5元)")
            elif '中等表现' in segment:
                print(" - 资源分配: 中等投入 (35%总资源)")
                print(" - 支持措施: 工作坊、在线资源、定期反馈")
                print(" - 预期ROI: 中等 (每投入1元，预期回报1.8元)")
            else:
                print(" - 资源分配: 基础投入 (25%总资源)")
                print(" - 支持措施: 自主学习材料、进阶课程")
                print(" - 预期ROI: 稳定 (每投入1元，预期回报1.3元)")
        
    
    #3.ROI分析
    
    print("\nROI分析:")
   
    #假设数据
    
    interventions = ['个性化辅导', '学习工作坊', '在线资源', '同伴导师']
    costs_per_student = [2000, 800, 300, 1500]  # 元/学生
    expected_improvements = [15, 8, 5, 10]  # 预期成绩提升百分比
    student_counts = [25, 40, 60, 20]  # 各措施目标学生数
   
    roi_data = []
    for i, (intervention, cost, improvement, count) in enumerate(zip(interventions, costs_per_student, expected_improvements, student_counts)):
        total_cost = cost * count
       
        #假设每提高1分价值500元
        
        total_benefit = improvement * 500 * count
        roi = (total_benefit - total_cost) / total_cost if total_cost > 0 else 0
       
        roi_data.append({
           '干预措施': intervention,
           '人均成本': f'¥{cost:,}',
           '目标学生数': count,
           '总成本': f'¥{total_cost:,}',
           '预期提分(%)': improvement,
           '总效益': f'¥{total_benefit:,}',
           '投资回报率': f'{roi:.2%}'
       })
   
    roi_df = pd.DataFrame(roi_data)
    print(roi_df.to_string(index=False))
   
    #可视化ROI分析
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
   
    #左图: 成本效益对比
    
    x = range(len(interventions))
    width = 0.35
   
    costs = [c * n for c, n in zip(costs_per_student, student_counts)]
    benefits = [imp * 500 * n for imp, n in zip(expected_improvements, student_counts)]
   
    ax1.bar([i - width/2 for i in x], costs, width, label='总成本', color=COLORS[0])
    ax1.bar([i + width/2 for i in x], benefits, width, label='总效益', color=COLORS[1])
    ax1.set_xlabel('干预措施')
    ax1.set_ylabel('金额 (元)')
    ax1.set_title('不同干预措施的成本效益分析')
    ax1.set_xticks(x)
    ax1.set_xticklabels(interventions, rotation=45, ha='right')
    ax1.legend()
   
    #右图: ROI对比
    
    rois = [(b - c) / c for b, c in zip(benefits, costs)]
    bars = ax2.bar(x, rois, color=COLORS[2:6])
    ax2.set_xlabel('干预措施')
    ax2.set_ylabel('投资回报率 (ROI)')
    ax2.set_title('不同干预措施的投资回报率')
    ax2.set_xticks(x)
    ax2.set_xticklabels(interventions, rotation=45, ha='right')
   
    #添加数值标签
    
    for bar, roi in zip(bars, rois):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height, f'{roi:.1%}', ha='center', va='bottom')
   
    plt.tight_layout()
    plt.savefig('../visualizations/roi_analysis.png', dpi=300, bbox_inches='tight')
    print("\nROI分析图表已保存")
   
    results['roi_analysis'] = roi_df
   
    return results

# ====== 额外挑战 ======

def missions_bonus(df,previous_results):
    """
    额外挑战: 如果只能实施一个干预措施
    """
    
    print("\n" + "="*60)
    print("额外挑战: 如果只能实施一个干预措施")
    print("="*60)
    
    #基于之前的分析选择最佳干预措施
    
    if 'roi_analysis' in previous_results:
       roi_df = previous_results['roi_analysis']
       
       # 找到ROI最高的措施
       
       roi_values = []
       for roi_str in roi_df['投资回报率']:
           
           # 从字符串中提取ROI值
           
            roi_value = float(roi_str.strip('%')) / 100
            roi_values.append(roi_value)
            
       best_idx = np.argmax(roi_values)
       best_intervention = roi_df.iloc[best_idx]
       
       print(f"\n推荐: {best_intervention['干预措施']}")
       print("\n数据支持:")
       print(f"1.最高ROI:{best_intervention['投资回报率']}")
       print("2.成本效益比最优")
       print(f"3.预期效果:平均提分{best_intervention['预期提分(%)']}%")
        
       print("\n实施细节:")
       print(f"-目标群体:{best_intervention['目标学生数']} 名学生")
       print(f"-总成本:{best_intervention['总成本']}")
       print(f"-预期总效益:{best_intervention['总效益']}")
        
       print("\n理由:")
       print("1.基于数据驱动决策:ROI分析显示此项措施回报率最高")
       print("2.可扩展性:易于实施和推广")
       print("3.目标明确:针对最需要帮助的学生群体")
       print("4. 可持续性:长期效果和知识转移")
        
       return best_intervention
    
    return None

# ==================== 生成分析报告 ====================

def generate_report(df, results_q1, results_q2, results_q3, best_intervention):
    """生成分析报告"""
    
    import os
    os.makedirs('../reports', exist_ok=True)
    
    report_content = f"""
# BI学生数据分析报告

## 项目概述
- 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- 数据规模: {df.shape[0]} 名学生
- 高风险学生比例: {results_q2.get('at_risk_percentage', 0):.1f}%

## 1. 招生优化分析
### 主要发现
- 入学考试预测能力: R² = {results_q1.get('entryexam_r2', 0):.3f}
- 最重要的预测特征: {results_q1.get('feature_importance', pd.DataFrame()).iloc[0]['feature'] if 'feature_importance' in results_q1 and len(results_q1['feature_importance']) > 0 else 'N/A'}

### 建议
- 保留入学考试作为主要筛选工具
- 结合教育背景和学习时长进行综合评估
- 建立预测模型优化录取决策

## 2. 课程支持策略
### 高风险学生分析
- 高风险学生数量: {results_q2.get('at_risk_count', 0)} 人
- 高风险学生比例: {results_q2.get('at_risk_percentage', 0):.1f}%

### 支持计划
- 目标群体: 平均成绩低于60分的学生
- 关键措施: 个性化辅导、学习工作坊、额外资源
- 预期成果: 高风险学生成绩提升20%

## 3. 资源分配与ROI
### 学生细分
将学生分为{len(results_q3.get('clusters', pd.DataFrame()))}个群体，实施差异化支持策略。

### ROI分析
最佳干预措施: {best_intervention['干预措施'] if best_intervention is not None else 'N/A'}
投资回报率: {best_intervention['投资回报率'] if best_intervention is not None else 'N/A'}

## 4. 总结与建议
### 立即行动
1. 实施{best_intervention['干预措施'] if best_intervention is not None else '推荐措施'}，重点关注高风险学生
2. 优化录取流程，结合多种评估指标
3. 建立学生表现监控系统

### 长期规划
1. 持续收集数据优化分析模型
2. 扩展支持计划覆盖更多学生
3. 建立数据驱动的教育决策文化
"""
    report_path = '../reports/full_analysis_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n完整分析报告已保存至: {report_path}")

# ==================== 主程序 ====================

def main():
    """主函数"""
    
    print("="*60)
    print("BI学生数据分析项目")
    print("="*60)
    
    # 创建输出目录
    
    import os
    os.makedirs('../visualizations', exist_ok=True)
    os.makedirs('../reports', exist_ok=True)
    
    # 1. 加载数据
    
    df = load_data()
    if df is None:
        return
    
    # 2. 探索性数据分析
    
    df = exploration_data_analysis(df)
    
    # 3. 问题1: 招生优化
    
    results_q1 = missions_recruit(df)
    
    # 4. 问题2: 课程支持策略
    
    results_q2 = missions_support(df)
    
    # 5. 问题3: 资源分配与ROI
    
    results_q3 = missions_allocation(df)
    
    # 6. 额外挑战
    
    best_intervention = missions_bonus(df, results_q3)
    
    # 7. 生成报告
    
    generate_report(df, results_q1, results_q2, results_q3, best_intervention)
    
    print("\n" + "="*60)
    print("数据分析完成!")
    print("="*60)
    print("生成的文件:")
    print(" - visualizations/ - 所有分析图表")
    print(" - reports/ - 分析报告")
    print("\n下一步建议:")
    print("  1. 查看生成的可视化图表")
    print("  2. 阅读分析报告")

if __name__ == "__main__":
    main()
    
    
    
      
        
    
    


 
