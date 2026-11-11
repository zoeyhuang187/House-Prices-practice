# House Prices project
基于 Kaggle "House Prices: Advanced Regression Techniques" 数据集的房价预测项目。使用 XGBoost 对 79 个房屋特征进行建模，预测房屋售价（SalePrice）。

## 项目概览

| 项目 | 内容 |
| :--- | :--- |
| 任务类型 | 回归（预测连续数值） |
| 数据集 | Ames Housing Dataset，1460 行训练 / 1459 行测试，79 个特征 |
| 目标变量 | `SalePrice`（美元） |
| 最终模型 | XGBoost（梯度提升树） |
| 评估指标 | 5 折交叉验证 R² |
| **最终结果** | **R² = 0.9041 ± 0.0078** |

## 数据集

数据来自 Kaggle 竞赛 [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)。

- `train.csv`：1460 行，包含 79 个特征 + 目标列 `SalePrice`
- `test.csv`：1459 行，仅包含特征
- `data_description.txt`：字段说明文档

特征涵盖房屋面积、质量评级、地下室、车库、社区、建造年份等。数据中包含数值型、有序分类型、名义分类型和缺失值，适合练习完整的特征工程流程。

## 方法

### 1. 目标变换

`SalePrice` 呈明显右偏（均值 180,921，中位数 163,000，最大值 755,000）。对目标做 `log1p` 变换，压缩右尾，减少极端高价房对损失函数的主导。

训练时使用 `log1p(SalePrice)`，预测时用 `expm1` 还原到原始价格尺度，评估指标均在原始尺度上计算。

### 2. 特征工程

| 处理 | 说明 |
| :--- | :--- |
| **有序编码** | 16 个等级字段（如 `ExterQual`、`KitchenQual`、`BsmtQual`）按业务顺序映射为整数（Ex=5, Gd=4, TA=3, Fa=2, Po=1, NA=0） |
| **独热编码** | 24 个名义字段（如 `Neighborhood`、`MSZoning`、`SaleType`）做独热编码 |
| **房龄构造** | `房龄 = YrSold - YearBuilt`，`改建房龄 = YrSold - YearRemodAdd` |
| **车库特征** | `有车库` 指示列 + `车库房龄`（无车库填 -1） |
| **缺失处理** | `LotFrontage` 填中位数并保留缺失指示列；`MasVnrArea` 填中位数；`MasVnrType` 填众数；`NA` 表示"无此设施"的字段映射为 0 |
| **特征裁剪** | 独热后特征膨胀到 230+ 列，基于 XGBoost 特征重要性裁剪到 80 列，控制过拟合 |

### 3. 模型

```python
XGBRegressor(
    n_estimators=452,        # 来自实验阶段早停的最优树数
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=5,
    reg_alpha=0.5,           # L1 正则
    reg_lambda=2.0,          # L2 正则
    subsample=0.7,
    colsample_bytree=0.6,
    random_state=42
)
最终模型在全部 1460 行训练数据上训练，n_estimators 固定为实验阶段早停得到的最优值（452），不再使用早停（全量训练无独立验证集）。

结果
5 折交叉验证
指标	均值	标准差
R²	0.9041	0.0078
R² 标准差仅 0.0078，说明模型在不同折之间表现稳定，不是单次切分的偶然结果。

如何运行
环境依赖
python >= 3.8
pandas
numpy
scikit-learn
xgboost >= 1.6

运行步骤
下载数据并放到项目根目录：

train.csv

test.csv

data_description.txt

运行特征工程与训练脚本：
python train.py

生成提交文件：
# 输出 submission.csv，格式为 Id, SalePrice

.
├── train.py                  # 特征工程 + 训练 + 预测
├── top_features.txt          # 实验阶段选出的 80 个特征列名
├── train.csv                 # 训练数据
├── test.csv                  # 测试数据
├── data_description.txt      # 字段说明
├── submission.csv            # 生成的提交文件
└── README.md

局限与后续改进
当前局限
特征裁剪基于单次模型的重要性：80 列的选择来自一次训练的特征重要性，可能存在少量信息损失，未做多轮稳定性验证。

未做模型融合：仅使用 XGBoost 单模型，未尝试与线性模型或 LightGBM 的加权融合。

中位数填充在 train/test 上分别计算：严格来说应使用训练集中位数填充测试集，当前实现在此数据集上影响可忽略。

可能的改进方向
模型融合：XGBoost + Ridge 或 XGBoost + LightGBM 的加权平均，预期可提升 0.005~0.01 R²。

特征工程补充：构造 TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF、TotalBath 等组合特征。

贝叶斯超参数优化：用 Optuna 替代手动调参，在更大参数空间内搜索。

目标编码：对 Neighborhood 等高基数分类字段尝试目标编码，替代独热编码。

作者
[黄超]


