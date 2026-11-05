# -*- coding: utf-8 -*-
"""
Word2Vec + XGBoost 文本分类训练模块
===================================

功能：读取预处理阶段产出的向量化数据（.npy），训练 XGBoost 分类器。

完整流程：
    1. 读取向量化后的 numpy 数组（data_preprocessing.py 产出）
    2. RandomizedSearchCV 随机搜索最优参数（除 n_estimators 外）
    3. 用最优参数 + early_stopping 在 dev 集上训练最终模型
    4. 测试集评估 + 保存模型

与 01_rf 的核心区别：
    ┌──────────────┬─────────────────────┬──────────────────────┐
    │              │ 01_rf               │ 02_xgb               │
    ├──────────────┼─────────────────────┼──────────────────────┤
    │ 特征          │ TF-IDF 稀疏向量       │ Word2Vec 密集向量    │
    │ 维度          │ 几万维（大部分是0）    │ 128维（每维都有值）    │
    │ 语义          │ 无（词是独立ID）      │ 有（近义词向量相近）    │
    │ 模型          │ 随机森林（Bagging）   │ XGBoost（Boosting）   │
    │ 特征工程复杂度  │ 低（直接算TF-IDF）    │ 中（需训练Word2Vec）   │
    └──────────────┴─────────────────────┴──────────────────────┘

依赖：
    - numpy：读取向量化数据
    - scikit-learn：评估指标、随机搜索
    - xgboost：XGBoost 分类器
    - joblib：模型序列化
"""

import os
import joblib
import numpy as np

from config import Config
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import xgboost as xgb

# ============================================================
# 加载配置
# ============================================================
config = Config()
search_jobs = int(os.environ.get('XGB_SEARCH_JOBS', '-1'))
model_jobs = int(os.environ.get('XGB_MODEL_JOBS', '-1'))
search_n_iter = int(os.environ.get('XGB_SEARCH_N_ITER', '20'))


# ============================================================
# 主程序入口
# ============================================================
# 预处理（data_preprocessing.py）已经完成：
#   1. jieba 分词 → CSV
#   2. Word2Vec 训练 + 向量化 → .npy 文件
# 本脚本直接读取向量化后的 numpy 数组，只负责模型训练和评估。
if __name__ == '__main__':
    # ========================================================
    # 1- 读取向量化后的数据（data_preprocessing.py 产出）
    # ========================================================
    print("读取向量化数据...")
    X_train = np.load(config.vector_train_datapath)
    y_train = np.load(config.vector_train_label_datapath)
    X_dev   = np.load(config.vector_dev_datapath)
    y_dev   = np.load(config.vector_dev_label_datapath)
    X_test  = np.load(config.vector_test_datapath)
    y_test  = np.load(config.vector_test_label_datapath)

    print(f"训练集：{X_train.shape[0]} 条，特征维度：{X_train.shape[1]}")
    print(f"验证集：{X_dev.shape[0]} 条")
    print(f"测试集：{X_test.shape[0]} 条")

    # ========================================================
    # 2- 阶段一：RandomizedSearchCV 随机搜索最优参数
    # ========================================================
    #
    # 策略：仅使用 train 做交叉验证搜索，dev 只用于最终 early stopping。
    #       这样避免同一个 dev 集同时参与调参和早停。
    #
    # XGBoost 核心思想：Boosting（提升）
    #   每棵新树都在修正之前所有树的残差（错误），逐步逼近真实值。
    #   与随机森林的 Bagging（并行、独立训练）不同，Boosting 是串行的。
    #
    # Bagging vs Boosting：
    #   ┌──────────┬──────────────────┬─────────────────────┐
    #   │          │ Bagging（RF）     │ Boosting（XGBoost） │
    #   ├──────────┼──────────────────┼─────────────────────┤
    #   │ 树的关系  │ 并行、独立        │ 串行、依赖          │
    #   │ 关注点   │ 减少方差          │ 减少偏差            │
    #   │ 过拟合   │ 天然抗过拟合      │ 需要正则化控制      │
    #   │ 训练速度 │ 快（可并行）      │ 慢（必须串行）      │
    #   │ 精度     │ 一般              │ 通常更高            │
    #   └──────────┴──────────────────┴─────────────────────┘

    X_search = X_train
    y_search = y_train
    print(f"\n搜索数据（train）：{len(X_search)} 条")

    # 2.1 计算类别数
    num_class = len(set(y_search))

    # 2.2 定义基础模型（搜索用，n_estimators 固定，不参与搜索）
    base_model = xgb.XGBClassifier(
        n_estimators=500,        # 搜索阶段固定，最终由 early_stopping 决定
        objective='multi:softmax',
        num_class=num_class,
        eval_metric='mlogloss',
        random_state=88,
        verbosity=0,             # 搜索时静默，避免日志爆炸
        n_jobs=model_jobs,
    )

    # 2.3 定义超参数搜索空间
    # 根据上一轮结果（最优：subsample=0.6, reg_lambda=0.5, reg_alpha=1, max_depth=8,
    # learning_rate=0.1, gamma=0, colsample_bytree=0.6）缩小搜索范围。
    # n_estimators 不参与搜索，最终由阶段二的 early_stopping 自动决定。
    # max_depth 上次最优 8 触顶，重点向深的方向搜。
    param_distributions = {
        'max_depth': [8, 10, 12, 15],                    # 树深度：上次 8 触顶，往上搜
        'learning_rate': [0.05, 0.1, 0.15],              # 学习率：围绕 0.1 微调
        'subsample': [0.5, 0.6, 0.7, 0.8],               # 数据采样：围绕 0.6
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],        # 特征采样：围绕 0.6
        'gamma': [0, 0.1, 0.2],                          # 分裂阈值：0 最优，留小范围
        'reg_alpha': [0.5, 1, 2],                        # L1 正则：围绕 1
        'reg_lambda': [0.3, 0.5, 1, 2],                  # L2 正则：围绕 0.5
    }

    # 2.4 创建随机搜索对象
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=search_n_iter,
        scoring='f1_macro',        # 多分类用宏平均 F1
        cv=2,                      # 2 折交叉验证
        n_jobs=search_jobs,
        verbose=10,                # 输出每组参数进度
        random_state=88,           # 固定随机种子
    )

    # 2.5 执行随机搜索
    print(f"\n类别数：{num_class}")
    print("=" * 50)
    print("阶段一：随机搜索（RandomizedSearchCV）")
    print(f"随机采样 {search_n_iter} 组参数，每组 2 折交叉验证")
    print(f"实际训练次数：{search_n_iter} × 2 = {search_n_iter * 2} 次")
    print("=" * 50)

    random_search.fit(X_search, y_search)

    # 2.6 输出搜索结果
    print("\n" + "=" * 50)
    print("随机搜索结果")
    print("=" * 50)
    print(f"最优参数组合：{random_search.best_params_}")
    print(f"最优交叉验证得分（{random_search.scoring}）：{random_search.best_score_:.4f}")

    # ========================================================
    # 3- 阶段二：用最优参数 + early_stopping 训练最终模型
    # ========================================================
    #
    # 为什么搜索完还要重新训练？
    #   RandomizedSearchCV 的 best_estimator_ 是在全部 X_search 上 refit 的，
    #   没有用 early_stopping。XGBoost 支持 early_stopping_rounds 参数，
    #   可以在验证集指标不再提升时自动停止，防止过拟合、节省时间。
    #
    # early_stopping_rounds 原理：
    #   每训练一棵树，在验证集上算一次 eval_metric（mlogloss）。
    #   如果连续 N 轮验证集指标没有提升，就停止训练。
    #   配合大的 n_estimators（如 2000），让模型自动找到最优树数。
    #
    # 参数决策：
    #   - n_estimators 设很大（2000），让 early_stopping 自动决定在哪停
    #   - 其他参数用搜索出的最优值

    print("\n" + "=" * 50)
    print("阶段二：用最优参数 + early_stopping 训练最终模型")
    print("=" * 50)

    best_params = random_search.best_params_.copy()
    # 搜索出的 n_estimators 只是参考值，最终由 early_stopping 自动决定
    best_params.pop('n_estimators', None)

    model = xgb.XGBClassifier(
        **best_params,
        n_estimators=2000,           # 设很大，让 early_stopping 自动决定在哪停
        objective='multi:softmax',
        num_class=num_class,
        eval_metric='mlogloss',
        random_state=88,
        verbosity=1,
        n_jobs=model_jobs,
        early_stopping_rounds=50,    # 连续 50 轮不提升就停止
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],   # 用 dev 集监控早停
        verbose=True,
    )

    print(f"\n最优树数（early_stopping 自动确定）：{model.best_iteration}")
    print(f"最优验证集 mlogloss：{model.best_score:.4f}")

    # ========================================================
    # 4- 阶段三：测试集评估
    # ========================================================
    print("\n" + "=" * 50)
    print("阶段三：测试集评估")
    print("=" * 50)

    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"宏平均精确率（Precision Macro）：{precision:.4f}")
    print(f"宏平均召回率（Recall Macro）：    {recall:.4f}")
    print(f"宏平均 F1 值（F1 Macro）：        {f1:.4f}")

    print("\n详细分类报告：")
    print(classification_report(y_test, y_pred, digits=4))

    # ========================================================
    # 5- 保存模型
    # ========================================================
    print("=" * 50)
    print("保存模型...")

    # XGBoost 支持两种保存方式：
    #   - joblib.dump()：Python 标准方式，与 sklearn 兼容，推理服务用此加载
    #   - model.save_model()：XGBoost 原生格式，跨语言
    joblib.dump(model, config.xgb_model_path)
    print(f"XGBoost 模型已保存到：{config.xgb_model_path}")

    print("\n训练完成！")
    print(f"最终 F1 Macro：{f1:.4f}")
    print(f"对比 01_rf（TF-IDF + RF）：约 0.73")
    print(f"提升：{(f1 - 0.73) * 100:.1f} 个百分点")