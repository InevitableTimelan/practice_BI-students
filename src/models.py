# -*- coding: utf-8 -*-
"""
models.py - 机器学习模型定义
"""

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import pandas as pd
import numpy as np

def train_regression_model(X, y, model_type='linear'):
    """训练回归模型"""
    
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    model.fit(X, y)
    return model

def train_classification_model(X, y, model_type='logistic'):
    """训练分类模型"""
    
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test, task='regression'):
    """评估模型性能"""
    
    y_pred = model.predict(X_test)
    
    if task == 'regression':
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return {'r2': r2, 'mse': mse, 'predictions': y_pred}
    elif task == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {'accuracy': accuracy, 'report': report, 'predictions': y_pred}
    else:
        raise ValueError(f"未知任务类型: {task}")

def get_feature_importance(model, feature_names):
    """获取特征重要性"""
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df
