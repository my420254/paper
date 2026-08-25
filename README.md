# FASTE: Span-Based Aspect Sentiment Triplet Extraction via Multi-Level Feature Fusion and Adversarial Training

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

