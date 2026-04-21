# ALoHA-Rec: Asymmetric-aware Low-rank Hierarchical Adaptation for Multi-Domain Recommendation

This repository contains the official implementation of ALoHA-Rec.

## Data Preparation

- Amazon: <http://jmcauley.ucsd.edu/data/amazon/>, place at `./data/amazon_5_core/amazon.csv`
- Douban: <https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewsideinformation>, place at `./data/douban/douban_sample.csv`
- KuaiRand-1K: <https://kuairand.com/>, place at `./data/kuairand/kuairand_sample.csv`

## Document Structure

```
scripts/
  amazon_main.py            entry script for Amazon
  douban_main.py            entry script for Douban
  kuairand_main.py          entry script for KuaiRand
src/
  basic/
    layers.py               embeddings, MLP, and extractors (FCN, DCN, DeepFM, xDeepFM, AutoInt)
    features.py             dense, sparse, and sequence feature definitions
    callback.py             early stopping
  models/multi_domain/
    adls.py                 ALoHA-Rec model
    sharebottom.py          SharedBottom backbone
    epnet.py                EPNet backbone
  trainers/
    adls_trainer.py         two-stage trainer
    ctr_trainer.py          standard trainer for single-task baselines
  utils/data.py             dataset and dataloader utilities
requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/amazon_main.py --extractor deepfm
```

## Acknowledgement

This codebase is built on top of the [Scenario-Wise-Rec](https://github.com/Xiaopengli1/Scenario-Wise-Rec) benchmark. We thank the authors for open-sourcing their framework.
