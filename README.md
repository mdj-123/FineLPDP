# FineLPDP

Fine-grained (line-level) software defect prediction via multi-task learning.

## 1. Dataset

```bash
git clone https://github.com/awsm-research/line-level-defect-prediction
```

## 2. CodeBERT

Download the CodeBERT checkpoint and put it in `./codebert-base` (default path):

```bash
git lfs install
git clone https://huggingface.co/microsoft/codebert-base
```

Or use the Hugging Face model id directly with `-model_name_or_path microsoft/codebert-base`.

## 3. Environment

```bash
conda create -n finelpdp python=3.10 -y
conda activate finelpdp
pip install -r requirements.txt
```

## 4. Training

```bash
python FineLPDP_train.py -dataset activemq
```

Omit `-dataset` to train on all 9 projects.

## 5. Prediction

```bash
python FineLPDP_prediction_within.py
```
