# -*- coding: utf-8 -*-
"""
随机森林文本分类训练模块
========================
功能：使用 TF-IDF + 随机森林（Random Forest）对中文新闻标题进行多分类。

完整流程：
    1. 读取预处理后的 CSV 数据
    2. 加载停用词表
    3. TF-IDF 特征提取（将文本转为数值向量）
    4. 划分训练集和测试集
    5. 随机搜索（RandomizedSearchCV）：在训练集上随机采样搜索最优超参数
    6. 模型训练（使用随机搜索最优参数）
    7. 模型评估（精确率、召回率、F1值、分类报告）
    8. 保存训练好的模型

核心算法简介：
    - TF-IDF（Term Frequency-Inverse Document Frequency）：
      衡量一个词对一篇文章的重要程度。词频（TF）越高、出现在越多文档中的词
      其 IDF 越低（即区分度越低），TF-IDF = TF × IDF。
    - 随机森林（Random Forest）：
      基于 Bagging 的集成学习算法，由多棵决策树组成，每棵树在随机抽样的
      数据和特征上训练，最终通过投票（分类）或平均（回归）得到结果。
      优点：抗过拟合能力强、能处理高维数据、可评估特征重要性。

依赖：
    - pandas：数据处理
    - scikit-learn：机器学习（TF-IDF、随机森林、评估指标）
    - joblib：模型序列化（比 pickle 更适合大规模 numpy 数组）
    - matplotlib：可视化超参数搜索过程
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 配置中文字体，解决 matplotlib 中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

from config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# 基础配置信息
# ============================================================
config = Config()


# ============================================================
# 主程序入口
# ============================================================
if __name__ == '__main__':
    # ========================================================
    # 1- 读取预处理好的数据
    # ========================================================
    df = pd.read_csv(config.process_train_datapath)
    # print(df.head())  # 查看前 5 行，确认数据加载正确

    # ========================================================
    # 2- 提取特征数据和目标值
    # ========================================================
    # words 列：分词后用空格连接的文本，如 "苹果 非常 好吃"
    words = df["words"]
    # label 列：分类标签（整数或字符串均可）
    label = df["label"]

    # ========================================================
    # 3- 特征工程：用 TF-IDF 来表示词向量
    # ========================================================

    # --- 3.1 加载停用词 ---
    # 停用词是指在文本中频繁出现但对分类没有贡献的词汇，
    # 如"的"、"了"、"在"、"是"等。去除它们可以：
    #   1. 减少特征维度，降低计算开销
    #   2. 减少噪声，提高模型泛化能力
    with open(config.stopwords_datapath, encoding="UTF-8") as f:
        # f.read().split()：读取全部内容并按空白字符（空格、换行等）分割
        stop_words_list = f.read().split()
        # print(stop_words_list)

    # --- 3.2 先划分文本，再仅用训练文本拟合 TF-IDF ---
    #
    # TfidfVectorizer 常用参数详解：
    #   - stop_words : {'english'}, list, or None, default=None
    #       停用词列表。传入 'english' 使用内置英文停用词。
    #   - max_df : float or int, default=1.0
    #       文档频率上限。float 表示比例（如 0.8 表示出现在超过 80% 文档中的词被忽略），
    #       int 表示绝对文档数。用于过滤过于常见的词。
    #   - min_df : float or int, default=1
    #       文档频率下限。float 表示比例，int 表示绝对文档数。
    #       用于过滤只在极少数文档中出现的罕见词。
    #   - max_features : int, default=None
    #       只保留 TF-IDF 值最高的前 N 个词。有效控制特征维度。
    #   - ngram_range : tuple (min_n, max_n), default=(1, 1)
    #       n-gram 范围。(1,1) 只使用单个词（unigram），
    #       (1,2) 同时使用单个词和相邻两个词的组合（bigram）。
    #   - analyzer : {'word', 'char', 'char_wb'}, default='word'
    #       特征分析单元。'word' 按词分析，'char' 按字符分析。
    #       中文场景下 'char' 有时比 'word' 效果更好，因为不需要分词。
    #   - token_pattern : str, default=r'(?u)\b\w\w+\b'
    #       用于识别"词"的正则表达式。默认匹配至少两个字符的单词。
    #       中文场景有时需要改为 r'(?u)\b\w+\b' 以保留单字词。
    #   - sublinear_tf : bool, default=False
    #       是否对 TF 做 sublinear scaling（1 + log(TF)），
    #       可削弱高频词的影响。
    #   - norm : {'l1', 'l2'}, default='l2'
    #       归一化方式。l2 是欧几里得归一化，l1 是曼哈顿归一化。
    tf_obj = TfidfVectorizer(stop_words=stop_words_list)

    words_train, words_test, y_train, y_test = train_test_split(
        words,
        label,
        test_size=0.2,
        random_state=88,
        shuffle=True,
        stratify=label,
    )

    # --- 3.3 对文本内容进行向量化处理 ---
    # fit_transform() = fit() + transform()
    #   - fit()：学习词汇表（构建 word -> index 的映射），计算 IDF 值
    #   - transform()：将文本转换为 TF-IDF 稀疏矩阵
    #
    # 返回值 x 是 scipy.sparse.csr_matrix（稀疏矩阵），
    # 只存储非零值，大幅节省内存。
    x_train = tf_obj.fit_transform(words_train)
    x_test = tf_obj.transform(words_test)

    """
    TF-IDF 处理后的稀疏矩阵数据解释：
        Coords	Values
        (0, 818)	0.4145423492151699

        1- 0：该词所在的句子索引（第 0 条样本）
        2- 818：该词在词汇表中的索引（第 818 个词）
        3- 0.4145423492151699：该词的 TF-IDF 值

    补充知识点：
        - tf_obj.get_feature_names_out()：获取所有特征名称列表
        - tf_obj.vocabulary_：获取词汇表 {词: 索引} 字典
        - tf_obj.idf_：获取每个词的 IDF 值数组
        - x.shape：返回 (样本数, 特征数)，即 (n_samples, n_features)
    """
    # print("处理后的特征名称：", tf_obj.get_feature_names_out())
    # print("处理后的词汇表：", tf_obj.vocabulary_)
    # print("处理后的词汇表大小：", len(tf_obj.vocabulary_))
    # print("处理后的词向量形状：", x.shape)

    # ========================================================
    # 4- 数据已在 TF-IDF 拟合前完成分层划分
    # ========================================================
    #
    # train_test_split() 常用参数详解：
    #   - *arrays : sequence of indexables
    #       要划分的数据集，可以传入多个数组（特征 + 标签），
    #       它们会被同步划分。
    #   - test_size : float or int, default=None
    #       测试集比例（如 0.2 表示 20% 数据作为测试集）或绝对数量。
    #       train_size 和 test_size 只需指定一个，另一个自动计算。
    #   - train_size : float or int, default=None
    #       训练集比例或绝对数量。
    #   - random_state : int, default=None
    #       随机种子。固定后每次划分结果一致，保证实验可复现。
    #       常用值：42、88、2024 等任意整数。
    #   - shuffle : bool, default=True
    #       是否在划分前随机打乱数据。一般设为 True 避免数据顺序偏差。
    #   - stratify : array-like, default=None
    #       分层抽样。传入标签数组，使训练集和测试集的类别分布一致。
    #       对于类别不平衡的数据集非常重要！
    #       例如：train_test_split(x, y, stratify=y) 保证各集合中类别比例相同。
    # ========================================================
    # 5- 随机搜索（RandomizedSearchCV）：在训练集上调参
    # ========================================================
    #
    # 本质：从参数分布中随机采样 N 组参数组合，通过交叉验证找到最优参数。
    # 相比网格搜索，随机搜索不遍历所有组合，而是固定采样次数，
    # 因此在参数空间很大时效率远高于网格搜索。
    # 注意：随机搜索只在训练集（x_train, y_train）上进行，
    #       测试集（x_test）全程不参与，只在最终评估时使用。
    #
    # RandomizedSearchCV 常用参数详解：
    #   - estimator : estimator object
    #       基础模型（estimator），如 RandomForestClassifier()。
    #   - param_distributions : dict or list of dicts
    #       超参数搜索空间。与 GridSearchCV 的 param_grid 不同，
    #       这里不仅支持列表，还支持 scipy 的分布对象（如 randint、uniform），
    #       从分布中随机采样，而不是遍历固定值。
    #       例如：
    #       {'n_estimators': [100, 200, 300]}          → 从列表中随机选
    #       {'n_estimators': randint(100, 500)}         → 从 100~500 中随机选
    #       {'max_depth': randint(5, 50)}               → 从 5~50 中随机选
    #   - n_iter : int, default=10
    #       随机采样的参数组合数。这是最关键的区别：网格搜索遍历所有组合，
    #       随机搜索只尝试 n_iter 组。n_iter 越大搜索越充分，但越耗时。
    #       推荐：先设小值（如 20~50）快速找到大致方向，再缩小范围精调。
    #   - scoring : str, callable, list, tuple or dict, default=None
    #       模型评估指标。同 GridSearchCV。
    #   - cv : int, cross-validation generator or iterable, default=None
    #       交叉验证折数。数据量大时用 cv=3 即可，小数据用 5。
    #   - n_jobs : int, default=None
    #       并行线程数。-1 使用全部 CPU 核心。
    #       随机搜索和网格搜索一样，各组参数间完全独立，非常适合并行。
    #   - verbose : int, default=0
    #       日志详细程度。
    #   - refit : bool, default=True
    #       是否用最优参数在全量训练集上重新训练。默认 True。
    #   - random_state : int, default=None
    #       随机种子。固定后可复现随机采样结果。
    #   - return_train_score : bool, default=False
    #       是否同时返回训练集评分。
    #
    # 补充知识点——为什么随机搜索比网格搜索高效：
    #   网格搜索对所有参数组合均匀采样，但很多参数对最终效果影响不大，
    #   浪费了大量时间在"不重要"的组合上。随机搜索在每个重要维度上
    #   都能探索更多不同的值，因此在相同迭代次数下通常能找到更好的解。
    #   论文《Random Search for Hyper-Parameter Optimization》证明：
    #   随机搜索在相同预算下通常优于网格搜索。

    # 5.1 定义基础模型
    base_model = RandomForestClassifier(random_state=88)

    # 5.2 定义超参数搜索空间（参数分布）
    # 注意：只有 10 个标签，树不需要太深就能区分，深度过大只会拖慢速度。
    # min_samples_split/leaf 调大，通过剪枝进一步减少树的大小。
    param_distributions = {
        'n_estimators': [150, 170, 250],          # 树的数量（10 标签不需要太多树
        'max_depth': [20, 25, 30],                # 树的最大深度（10 标签，浅树足够）
        'min_samples_split': [20, 30, 50],        # 内部节点最小分裂样本数
        'min_samples_leaf': [16],                 # 叶子节点最小样本数
        'max_features': ['sqrt'],                 # 每棵树随机选取的特征数
    }

    # 5.3 创建随机搜索对象
    random_search = RandomizedSearchCV(
        estimator=base_model,               # 基础模型
        param_distributions=param_distributions,  # 参数搜索空间
        n_iter=20,                          # 随机采样 20 组参数组合
        scoring='f1_macro',                 # 多分类用宏平均 F1 作为评估指标
        cv=2,                               # 2 折交叉验证（数据量大，2 折足够）
        n_jobs=-1,                          # 使用所有 CPU 核心并行搜索
        verbose=10,                         # 输出每组参数的搜索进度
        random_state=88,                    # 固定随机种子，保证可复现
    )

    # 5.4 在训练集上执行随机搜索
    print("=" * 50)
    print("开始随机搜索（RandomizedSearchCV），在训练集上寻找最优参数...")
    print(f"从 {3 * 4 * 3 * 3 * 2} 种可能组合中随机采样 {20} 组，每组进行 {2} 折交叉验证")
    print(f"实际训练次数：{20} × {2} = {20 * 2} 次")
    print("=" * 50)

    # 注意：RandomizedSearchCV 是阻塞调用，tqdm 无法介入其内部循环。
    # 因此这里不用 tqdm 包装，而是通过 verbose=10 让 sklearn 自身输出进度：
    #   [CV] END ... n_estimators=200, max_depth=20 ...; score=0.852 total time=12.3s
    # 每完成一组参数就会打印一行，可以看到实时进度。
    random_search.fit(x_train, y_train)

    # 5.5 输出随机搜索最优结果
    print("\n" + "=" * 50)
    print("随机搜索结果")
    print("=" * 50)
    print(f"最优参数组合：{random_search.best_params_}")
    print(f"最优交叉验证得分（{random_search.scoring}）：{random_search.best_score_:.4f}")

    # 5.6 查看所有采样结果的排名（可选）
    # cv_results = pd.DataFrame(random_search.cv_results_)
    # print("所有采样结果排名（按得分降序）：")
    # print(cv_results[['params', 'mean_test_score', 'rank_test_score']].head(10))

    # 5.6 可视化：超参数搜索过程
    #     将 cv_results_ 转为 DataFrame，绘制两张图：
    #       图1：所有参数组合的得分排名柱状图（最直观）
    #       图2：每个超参数与得分的散点图（看趋势和最优值）
    cv_results = pd.DataFrame(random_search.cv_results_)
    params_df = pd.DataFrame(cv_results['params'].tolist())
    params_df['mean_test_score'] = cv_results['mean_test_score']
    params_df['rank'] = cv_results['rank_test_score']

    # 将参数组合格式化为可读字符串（用于柱状图标签）
    params_df['param_label'] = params_df.apply(
        lambda row: f"n={row['n_estimators']} d={row['max_depth']} "
                    f"sp={row['min_samples_split']} lf={row['min_samples_leaf']} "
                    f"f={row['max_features']}",
        axis=1
    )

    # ================================================================
    # 图1：得分排名柱状图 —— 一眼看出哪些参数组合效果好
    # ================================================================
    # 按得分降序排列，颜色渐变：得分越高颜色越深
    sorted_df = params_df.sort_values('mean_test_score', ascending=True)
    colors = plt.cm.YlOrRd(0.3 + 0.7 * (sorted_df['mean_test_score'] - sorted_df['mean_test_score'].min())
                            / (sorted_df['mean_test_score'].max() - sorted_df['mean_test_score'].min()))

    fig1, ax1 = plt.subplots(figsize=(16, 8))
    bars = ax1.barh(range(len(sorted_df)), sorted_df['mean_test_score'], color=colors, edgecolor='#333', linewidth=0.5)
    ax1.set_yticks(range(len(sorted_df)))
    ax1.set_yticklabels(sorted_df['param_label'], fontsize=9)
    ax1.set_xlabel('F1 Macro Score', fontsize=13)
    ax1.set_title('随机搜索：所有参数组合得分排名', fontsize=15, fontweight='bold')
    # 标注最优和最差
    ax1.annotate(f'最优 {sorted_df["mean_test_score"].iloc[-1]:.4f}',
                 xy=(sorted_df['mean_test_score'].iloc[-1], len(sorted_df) - 1),
                 fontsize=11, color='red', fontweight='bold', va='center')
    ax1.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('hyperparameter_ranking.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("排名柱状图已保存为 hyperparameter_ranking.png")

    # ================================================================
    # 图2：各参数对得分的影响 —— 看趋势，找规律
    # ================================================================
    param_info = [
        ('n_estimators', '树的数量', [100, 150, 200]),
        ('max_depth', '树的最大深度', [8, 10, 12, 15]),
        ('min_samples_split', '最小分裂样本数', [20, 50, 100]),
        ('min_samples_leaf', '叶子最小样本数', [8, 16, 32]),
    ]

    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    best = random_search.best_params_

    for idx, (param, title, xticks) in enumerate(param_info):
        ax = axes[idx]
        # 散点图：每个点一组参数组合，加一点随机抖动避免重叠
        jitter = (pd.Series(xticks).diff().mean() or 1) * 0.05
        x_jittered = params_df[param] + np.random.uniform(-jitter, jitter, len(params_df))
        ax.scatter(x_jittered, params_df['mean_test_score'],
                   c='#5B9BD5', s=80, alpha=0.6, edgecolors='white', linewidth=0.5)
        # 最优值高亮
        ax.scatter([best[param]], [random_search.best_score_],
                   c='red', s=250, marker='*', edgecolors='darkred', linewidth=1,
                   zorder=5, label=f'最优={best[param]}')
        ax.set_xticks(xticks)
        ax.set_xlabel(title, fontsize=13)
        ax.set_ylabel('F1 Macro Score', fontsize=13)
        ax.set_title(f'{title} 对得分的影响', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('随机搜索：超参数与交叉验证得分的关系', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('hyperparameter_scatter.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("参数影响散点图已保存为 hyperparameter_scatter.png")

    # 补充：max_features 对比图（单独一张小图，两个柱状图对比）
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    feat_scores = params_df.groupby('max_features')['mean_test_score'].agg(['mean', 'std'])
    bars = ax3.bar(feat_scores.index, feat_scores['mean'],
                   yerr=feat_scores['std'], color=['#5B9BD5', '#ED7D31'],
                   edgecolor='#333', linewidth=0.8, capsize=8, width=0.4)
    # 在柱子上标注数值
    for bar, val in zip(bars, feat_scores['mean']):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                 f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1 Macro Score', fontsize=13)
    ax3.set_title('max_features 对得分的影响', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('hyperparameter_max_features.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("特征选择对比图已保存为 hyperparameter_max_features.png")

    # 5.7 获取最优模型（RandomizedSearchCV 默认 refit=True，已用最优参数重新训练）
    model = random_search.best_estimator_

    # ========================================================
    # 6- 模型训练（使用随机搜索最优参数）
    # ========================================================
    # 注意：由于 RandomizedSearchCV 的 refit=True（默认），
    # best_estimator_ 已经是用最优参数在全量训练集上训练好的模型，
    # 无需再次调用 fit()。以下代码展示如果不使用搜索时的手动训练方式。
    #
    # RandomForestClassifier 常用参数详解：
    #
    #   --- 核心参数 ---
    #   - n_estimators : int, default=100
    #       决策树的数量（弱学习器个数）。
    #       数量越多，模型越稳定，泛化能力越强，但训练时间线性增长。
    #       推荐范围：100~1000。不是越多越好，到达一定数量后收益递减。
    #
    #   - criterion : {'gini', 'entropy', 'log_loss'}, default='gini'
    #       分裂质量衡量标准。
    #       'gini'：基尼不纯度，计算快。
    #       'entropy'：信息增益，与 ID3/C4.5 算法一致。
    #       'log_loss'：与 'entropy' 相同（新版本名称）。
    #
    #   - max_depth : int, default=None
    #       树的最大深度。None 表示不限制，节点会一直分裂直到纯或
    #       达到 min_samples_split。限制深度是防止过拟合的有效手段。
    #       推荐范围：5~30，根据数据量调整。
    #
    #   - min_samples_split : int or float, default=2
    #       内部节点再分裂所需的最小样本数。
    #       int：绝对数量；float：比例（如 0.01 表示 1%）。
    #       增大此值可以防止过拟合。
    #
    #   - min_samples_leaf : int or float, default=1
    #       叶子节点最少样本数。int 为绝对数量，float 为比例。
    #       增大此值可以平滑模型，防止过拟合。
    #
    #   - max_features : {'sqrt', 'log2'}, int or float, default='sqrt'
    #       每棵树随机选取的特征数量。
    #       'sqrt'：sqrt(n_features)，分类问题默认值。
    #       'log2'：log2(n_features)。
    #       None：使用全部特征（即 Bagging）。
    #       减少此值可以增加树之间的多样性，降低过拟合。
    #
    #   --- 并行与性能 ---
    #   - n_jobs : int, default=None
    #       并行线程数。-1 使用所有 CPU 核心，None 表示 1 个。
    #       注意：RandomForestClassifier 的并行是 embarrassingly parallel，
    #       每棵树独立训练，因此加速效果接近线性。
    #
    #   - verbose : int, default=0
    #       日志详细程度。0 不输出，>0 输出训练过程信息。
    #
    #   --- 样本权重 ---
    #   - class_weight : dict, 'balanced', 'balanced_subsample', or None
    #       类别权重。'balanced' 自动按类别样本数反比加权，
    #       对类别不平衡问题非常有效。
    #       'balanced_subsample' 类似，但基于每棵树的 bootstrap 样本计算。
    #
    #   - bootstrap : bool, default=True
    #       是否使用 bootstrap 抽样（有放回抽样）构建每棵树的训练集。
    #       True 是标准的随机森林，False 则使用全部样本。
    #
    #   - oob_score : bool, default=False
    #       是否使用袋外样本（out-of-bag）评估模型。
    #       开启后可通过 model.oob_score_ 获取 OOB 准确率，
    #       相当于免费的交叉验证，不需要额外划分验证集。
    #
    #   --- 随机种子 ---
    #   - random_state : int, default=None
    #       随机种子，保证结果可复现。
    # 如果不使用网格搜索，手动训练的方式如下（已注释）：
    # model = RandomForestClassifier(n_estimators=100)
    # model.fit(x_train, y_train)

    # ========================================================
    # 7- 模型评估
    # ========================================================
    y_pred = model.predict(x_test)

    """
    多分类评估指标 average 参数详解：
        - 'binary'：仅适用于二分类，返回正类的指标值
        - 'macro'：宏平均。先计算每个类别的指标，再取算术平均。
          每个类别权重相同，不受样本数量影响。
          适用场景：各类别样本量均衡，或所有类别同等重要。
        - 'micro'：微观平均。将所有类别的 TP/FP/FN 汇总后计算全局指标。
          结果等于 Accuracy（准确率）。
          适用场景：样本分布不均衡，希望大类别主导评估结果。
        - 'weighted'：加权平均。先计算每个类别的指标，再按样本数加权平均。
          适用场景：样本分布不均衡，但希望考虑每个类别的表现。
        - None：返回每个类别各自的指标值（数组形式）。

    补充知识点——混淆矩阵相关概念：
        - TP（True Positive）：预测为正，实际为正
        - FP（False Positive）：预测为正，实际为负（误报）
        - FN（False Negative）：预测为负，实际为正（漏报）
        - TN（True Negative）：预测为负，实际为负

        - 精确率（Precision）= TP / (TP + FP)
          预测为正的样本中真正为正的比例。关注"预测准不准"。
        - 召回率（Recall）= TP / (TP + FN)
          实际为正的样本中被正确预测的比例。关注"找得全不全"。
        - F1值 = 2 × Precision × Recall / (Precision + Recall)
          精确率和召回率的调和平均，综合衡量模型性能。
          取值范围 [0, 1]，越接近 1 越好。
          调和平均比算术平均更严格，只有两者都高时 F1 才高。
    """
    print("=" * 50)
    print("模型评估结果")
    print("=" * 50)
    print(f"精确率（Precision）：{precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"召回率（Recall）：   {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1值（F1-Score）：   {f1_score(y_test, y_pred, average='macro'):.4f}")

    # classification_report：生成详细的分类报告，包含每个类别的
    # precision、recall、f1-score 和 support（样本数）
    print("\n分类报告（Classification Report）：")
    print(classification_report(y_test, y_pred))

    # ========================================================
    # 8- 保存训练好的模型
    # ========================================================
    #
    # 知识点：joblib vs pickle
    #   - pickle 是 Python 标准库的序列化工具，适合通用 Python 对象。
    #   - joblib 是专门为科学计算优化的序列化库，对大型 numpy 数组
    #     的序列化更高效（内部使用高效的压缩和序列化协议）。
    #   - 对于 scikit-learn 模型，推荐使用 joblib，因为：
    #     1. 序列化/反序列化速度更快
    #     2. 文件体积更小
    #     3. 支持内存映射（memory-mapping），加载大模型时更高效
    #
    # pickle 的等效写法（备选方案）：
    #     with open(config.rf_model_path, mode="wb") as f:
    #         pickle.dump(model, f)
    #     with open(config.tfidf_model_path, mode="wb") as f:
    #         pickle.dump(tf_obj, f)

    joblib.dump(model, config.rf_model_path)
    joblib.dump(tf_obj, config.tfidf_model_path)

    print("\n" + "=" * 50)
    print(f"随机森林模型已保存至：{config.rf_model_path}")
    print(f"TF-IDF向量器已保存至：{config.tfidf_model_path}")
    print("=" * 50)