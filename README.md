# FASTE：多层特征融合与对抗训练的方面级情感三元组抽取

FASTE 是一套面向方面级情感三元组抽取（Aspect Sentiment Triplet Extraction, ASTE）的 span-based NLP 框架，当前按 **Expert Systems with Applications（ESWA）** 投稿版本整理。项目聚焦结构化情感抽取中的边界识别、关系组合和情感极性联合建模问题。

ASTE 的目标是从句子中同时抽取：

```text
Aspect Term + Opinion Term + Sentiment Polarity
```

这比普通情感分类更复杂，因为模型不仅要判断情感倾向，还要同时识别 aspect 边界、opinion 边界，并把两者正确配对。

## 研究问题

现有 ASTE 方法常见两类问题：

- 生成式方法灵活，但边界控制和复现实验稳定性较弱；
- 传统判别式 span 方法过度依赖最后一层 encoder 表示，容易丢失浅层句法和边界线索。

FASTE 的设计重点是保留判别式方法的可控解码优势，同时通过多层特征融合和对抗训练提升边界鲁棒性。

## 方法设计

### MLFF：多层特征融合

代码中使用 `MLFF_LAYERS = [3, 7, 11]` 融合低层、中层和高层表示。低层更敏感于词法与边界，中层更适合局部短语组合，高层提供全局语义。多层融合可以缓解只依赖最后一层导致的边界信息弱化问题。

### AT-FGM：对抗训练

项目在 embedding 空间加入 FGM 扰动，让模型在局部噪声和边界模糊样本上保持稳定。对 ASTE 这类边界敏感任务而言，轻微 token 表示扰动可能导致 span 偏移，对抗训练可以提升模型鲁棒性。

### Span-Based 解码

FASTE 通过 span representation 构造候选实体和候选关系，相比纯生成式抽取更容易控制输出格式，也便于复现、评估和错误定位。

## 工程亮点

- 单文件主流程覆盖数据读取、batch 构造、模型训练、预测、评测和资源统计；
- 内置主实验、消融实验和敏感性分析入口；
- 支持 t-SNE 表征分析和论文图表生成；
- 代码中保留 `SpanInstance`、`BatchLoader`、`SpanRepresentation`、`SpanASTEModel` 等清晰模块；
- 结果包包含日志、图表和补充实验材料，便于论文复现和结果审计。

## 结果摘要

在 ASTE-Data-V2 多个领域上，FASTE 在非生成式系统中取得有竞争力的结果，其中 `14lap` 数据集 F1 达到 **66.60**。这说明多层表示和对抗训练能有效提升 span 边界和关系组合质量。

## 仓库结构

| 文件 | 说明 |
| --- | --- |
| `aste.py` | 主训练、预测、评测、消融和敏感性分析流程 |
| `add_ablation.py` | 补充消融实验 |
| `extract_tsne.py` | 表征可视化数据抽取 |
| `create_image.py` | 论文图表生成 |
| `paper_results.zip` | 实验日志与结果包 |
| `sensitivity_analysis.pdf` | 敏感性分析图 |
| `tsne_visualization.pdf` | t-SNE 可视化图 |

## 运行方式

```bash
python aste.py --mode main --datasets lap14 res14
```

运行消融或敏感性分析：

```bash
python aste.py --mode ablation --datasets lap14
python aste.py --mode sensitivity --datasets lap14
```

实际数据路径由 `aste.py` 中的 `DATA_ROOT` 和 `DATASET_CONFIGS` 控制。

## 项目状态

- 论文状态：ESWA 投稿版本；
- 代码状态：主实验、消融、敏感性分析、可视化和结果包已整理；
- 许可协议：MIT License。
