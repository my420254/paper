# FASTE: Span-Based Aspect Sentiment Triplet Extraction via Multi-Level Feature Fusion and Adversarial Training

## 中文项目介绍

FASTE 是我围绕方面级情感三元组抽取（Aspect Sentiment Triplet Extraction, ASTE）完成的 span-based 信息抽取工作，当前按 **Expert Systems with Applications（ESWA）** 投稿材料组织。ESWA 是人工智能应用方向的高影响力期刊，常见分区口径较高；最终分区和录用状态以当年官方结果为准。

这个任务的目标不是只判断一句话情感，而是同时抽取：

```text
Aspect Term + Opinion Term + Sentiment Polarity
```

难点在于边界识别、跨词搭配、情感极性判断要同时正确。很多生成式方法看起来灵活，但在实体边界和稳定复现上容易出错；传统 span-based 方法又容易过度依赖最后一层 encoder 表示，导致浅层句法和边界线索丢失。

## 核心方法

- **MLFF 多层特征融合**：融合中间层和高层表示，让模型同时保留语义、句法和边界线索，提升 span 边界定位能力。
- **AT-FGM 对抗训练**：在 embedding 空间引入受控扰动，提升模型对局部噪声和边界模糊样本的鲁棒性。
- **span-based 解码**：相比纯生成式抽取，更强调结构化边界、可控解码和结果复现。

## 面试展示重点

- **任务复杂度**：ASTE 同时涉及实体抽取、关系组合和情感分类，是比普通情感分类更复杂的结构化 NLP 任务。
- **模型设计**：不是简单套 BERT，而是针对“最后层语义强但边界弱”的缺陷做多层特征融合。
- **实验表现**：在 ASTE-Data-V2 多个领域上取得强结果，其中 14lap F1 达到 **66.60**，在非生成式系统中具备竞争力。
- **工程完整性**：包含训练、评测、t-SNE 表征分析、图表生成和 paper result package，能支撑论文写作和复现实验。
- **可讲难点**：span 数量膨胀、负样本极多、边界错一位即判错、不同领域数据分布差异明显、生成式和判别式方法的评测口径不同。

## 技术关键词

`PyTorch` · `Transformer` · `BERT` · `ASTE` · `Span-based Extraction` · `Adversarial Training` · `Feature Fusion` · `NLP`

This repository contains the official implementation of **FASTE**, a span-based ASTE framework.

## What this work solves

Most span-based ASTE models rely too heavily on the final encoder layer, which drops boundary-sensitive syntactic cues and causes span mismatch.

FASTE addresses that by combining:

- **Multi-Level Feature Fusion (MLFF)**: recovers lower-level syntactic and boundary information from intermediate layers
- **Adversarial Training (AT-FGM)**: stabilizes optimization when local noise makes the loss landscape irregular

## My contribution

- Designed the model structure and training recipe
- Built the evaluation / visualization pipeline
- Prepared the paper figures, error analysis, and experimental comparison package

## Why it matters

- Stronger boundary alignment than generation-style ASTE systems
- Compact enough for real-time deployment
- Built for reproducible evaluation, not only for paper numbers

## Main result

On ASTE-Data-V2, FASTE achieves state-of-the-art results among non-generative systems across multiple benchmark domains, including **66.60 F1 on 14lap**.

## Repository contents

- `aste.py`: main training and inference pipeline
- `extract_tsne.py`: representation analysis
- `create_image.py`: figure generation
- `paper_results.zip`: logs and result package

## Status

ESWA submission version.
