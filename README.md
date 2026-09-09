# FineLPDP

Fine-grained (line-level) software defect prediction via multi-task learning.

## 1. Dataset

```bash
git clone https://github.com/awsm-research/line-level-defect-prediction
```

## 2. Environment

```bash
conda create -n finelpdp python=3.10 -y
conda activate finelpdp
pip install -r requirements.txt
```

## 3. Training

```bash
python FineLPDP_train.py -dataset activemq
```

Omit `-dataset` to train on all 9 projects.

## 4. Prediction

```bash
python FineLPDP_prediction_within.py
```
