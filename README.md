# FoldSafe-HybridMDA（seed 2027）

NNSFMDA 参考：Yang et al., *Journal of Molecular Biology* 437(12), 169086 (2025), DOI: 10.1016/j.jmb.2025.169086。

该版本在原 FoldSafe-TreeMDA 的 133 维折内特征基础上增加了 NNSFMDA-inspired 矩阵分支，并削弱随机森林：

- ExtraTrees：主分支，最终权重不低于 0.65；
- Histogram Gradient Boosting：互补监督分支；
- NNSFMDA-inspired：折内相似度传播、有界核范数矩阵补全与单层全局注意力分支；
- RandomForest：弱辅助分支，树数从 400 降至 120、限制深度，最终权重不超过 0.10。

为保证新增矩阵分支确实参与最终模型、同时避免较弱的独立矩阵分支压制监督分支，NNSFMDA-inspired 权重预先限定在 0.05–0.10；这项约束在查看外层测试标签前固定。其余权重仍只由内层验证集选择。

融合权重不是用外层测试集选择的。每个外层折中重新划分内层训练/验证集，删除内层验证正边，重新构造全部特征和 NNSFMDA-inspired 分数，再在内层验证集搜索权重。之后才在未触碰的外层测试集评价。

## 运行

```powershell
python FoldSafe_HybridMDA_Seed2027.py `
  --data ".\data\MDAD" `
  --brmda-view ".\data\mdad_brmda_view.npz" `
  --output ".\results_seed2027"
```

脚本依赖同目录的 `FoldSafe_TreeMDA_Seed2027.py`，用于复用已经审计的 MDAD 加载、折分和 133 维折内特征构造。

这里使用“NNSFMDA-inspired”而非“NNSFMDA 原代码”这一名称：公开论文明确给出有界核范数补全与简化 Transformer 的总体结构，但本项目没有论文作者的训练源码。实现采用显式 `[0,1]` 投影的奇异值阈值迭代，并在补全后的异构网络上使用一个确定性的全局注意力层。分支内部使用 0.65 相似度传播、0.25 有界补全和 0.10 注意力微调，以避免较弱的注意力近似覆盖稳定的折内传播信号。

## 结果文件

- `summary.json`：每个分支和融合模型的五折均值、标准差与 pooled 指标；
- `fold_metrics.csv`：各折指标、权重和泄漏检查；
- `predictions.csv`：全部外层 out-of-fold 预测；
- `inner_weight_grid.csv`：仅由内层验证集计算的权重搜索记录；
- `roc_pr_curves.png`：融合模型、ExtraTrees 和 NNSFMDA-inspired 分支的 ROC/PR 曲线。

## 解释限制

结果使用 1:1 平衡伪负样本和 pair-wise transductive CV。未知关联并非经实验确认的真阴性；该结果不能直接外推到“全部未知对”或完全未见药物/微生物的冷启动任务。代码以真实运行结果为准，不人为保证 0.98。

## DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23099723.svg)](https://doi.org/10.5281/zenodo.23099723)

Archived release (v1.0.0): https://doi.org/10.5281/zenodo.23099723
