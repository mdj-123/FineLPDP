
import copy
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
from my_util import *
from FineLPDP import FineLPDPModel


class InputFeatures(object):
    def __init__(self, input_ids, label, line_label):
        self.input_ids = input_ids
        self.label = label
        self.line_label = torch.FloatTensor(line_label)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, datasets, labels, line_labels):
        self.examples = []
        labels = torch.FloatTensor(labels)
        for dataset, label, line_label in zip(datasets, labels, line_labels):
            dataset_ids = [convert_examples_to_features(item, tokenizer, args) for item in dataset]
            self.examples.append(InputFeatures(dataset_ids, label, line_label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), self.examples[i].label, self.examples[i].line_label


def convert_examples_to_features(item, tokenizer, args):
    code = ' '.join(item)
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return source_ids


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    return [data_list for data_list in batch]


def prototype_contrastive_loss(line_features, line_labels_padded, line_mask,
                                prototypes, margin=1.0, device='cuda'):

    features = line_features[line_mask]                     # (N_valid, D)
    labels = line_labels_padded[line_mask].long().to(device)  # (N_valid,)

    # L2-normalize for scale-invariant distances
    features_norm = F.normalize(features, p=2, dim=-1)       # (N_valid, D)
    prototypes_norm = F.normalize(prototypes, p=2, dim=-1)   # (2, D)

    pos_mask = (labels == 1)
    neg_mask = (labels == 0)

    total_loss = torch.tensor(0.0, device=device)
    n_components = 0

    # ---- 1. Pull toward own prototype ----
    if pos_mask.sum() > 0:
        pos_feats = features_norm[pos_mask]
        pos_proto = prototypes_norm[1].unsqueeze(0)          # (1, D)
        pull_pos = ((pos_feats - pos_proto) ** 2).sum(dim=-1).mean()
        total_loss = total_loss + pull_pos
        n_components += 1

    if neg_mask.sum() > 0:
        neg_feats = features_norm[neg_mask]
        neg_proto = prototypes_norm[0].unsqueeze(0)
        pull_neg = ((neg_feats - neg_proto) ** 2).sum(dim=-1).mean()
        total_loss = total_loss + pull_neg
        n_components += 1

    # ---- 2. Push away from opposite prototype ----
    if pos_mask.sum() > 0:
        pos_feats = features_norm[pos_mask]
        neg_proto = prototypes_norm[0].unsqueeze(0)
        dist_pos_to_neg = ((pos_feats - neg_proto) ** 2).sum(dim=-1)   # (N_pos,)
        push_pos = F.relu(margin - dist_pos_to_neg).mean()
        total_loss = total_loss + push_pos
        n_components += 1

    if neg_mask.sum() > 0:
        neg_feats = features_norm[neg_mask]
        pos_proto = prototypes_norm[1].unsqueeze(0)
        dist_neg_to_pos = ((neg_feats - pos_proto) ** 2).sum(dim=-1)
        push_neg = F.relu(margin - dist_neg_to_pos).mean()
        total_loss = total_loss + push_neg
        n_components += 1

    # ---- 3. Push the two prototypes apart ----
    proto_dist = ((prototypes_norm[0] - prototypes_norm[1]) ** 2).sum()
    proto_push = F.relu(margin - proto_dist)
    total_loss = total_loss + proto_push
    n_components += 1

    return total_loss / max(n_components, 1)


def train_model(args, dataset_name):

    train_rel = [all_train_releases[dataset_name]]
    valid_rel = all_eval_releases[dataset_name][0]

    train_dfs = [get_df(version) for version in train_rel]
    train_df = pd.concat(train_dfs, ignore_index=True)
    valid_df = get_df(valid_rel)

    train_code3d, train_label, train_line_label = get_code3d_and_label(train_df, True, args.max_train_LOC)
    valid_code3d, valid_label, valid_line_label = get_code3d_and_label(valid_df, True, args.max_train_LOC)

    total_lines = sum(len(ll) for ll in train_line_label)
    pos_lines = sum(sum(ll) for ll in train_line_label)
    neg_lines = total_lines - pos_lines
    line_pos_weight = torch.tensor([neg_lines / max(pos_lines, 1)])

    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    x_train_vec = TextDataset(tokenizer, args, train_code3d, train_label, train_line_label)
    x_valid_vec = TextDataset(tokenizer, args, valid_code3d, valid_label, valid_line_label)

    train_labels_tensor = torch.FloatTensor(train_label)
    class_counts = torch.bincount(train_labels_tensor.long())
    sample_weights = 1.0 / class_counts[train_labels_tensor.long()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_dl = DataLoader(x_train_vec, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    valid_dl = DataLoader(x_valid_vec, shuffle=False, batch_size=args.batch_size, drop_last=False,
                          collate_fn=collate_fn)

    codebert = model_class.from_pretrained(args.model_name_or_path, config=config)
    codebert.to(args.device)
    for param in codebert.parameters():
        param.requires_grad = False

    model = FineLPDPModel(
        line_dim=768,
        line_gru_dim=256,
        line_num_layers=1,
        dropout=0.1,
        device=args.device,
    ).to(args.device)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    line_criterion = nn.BCEWithLogitsLoss(pos_weight=line_pos_weight.to(args.device))
    sig = nn.Sigmoid()

    best_auc = 0
    best_model = None

    results = {
        'train_loss': [], 'val_loss': [],
        'val_auc': [], 'val_ba': [], 'val_mcc': [],
        'val_recall': [], 'val_effort': [], 'val_ifa': []
    }

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_losses = []

        for step, batch in tqdm(enumerate(train_dl), total=len(train_dl), desc=f'Train Loop'):
            inputs = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            line_labels = [item[2] for item in batch]
            labels = torch.tensor(labels)

            max_lines = max(ll.size(0) for ll in line_labels)
            line_labels_padded = torch.zeros(len(line_labels), max_lines)
            line_mask = torch.zeros(len(line_labels), max_lines, dtype=torch.bool)
            for i, ll in enumerate(line_labels):
                n = ll.size(0)
                line_labels_padded[i, :n] = ll
                line_mask[i, :n] = True

            cov_inputs = []
            with torch.no_grad():
                for item in inputs:
                    codeemb = codebert(item.to(args.device), attention_mask=item.to(args.device).ne(1))
                    cov_inputs.append(codeemb.pooler_output)


            file_output, line_output, _, file_features, raw_line_features = model(
                cov_inputs, return_line_features=True
            )

            file_loss = criterion(file_output.reshape(len(labels)), labels.to(args.device))

            line_loss = line_criterion(line_output[line_mask],
                                       line_labels_padded[line_mask].to(args.device))


            proto_loss = prototype_contrastive_loss(
                raw_line_features,
                line_labels_padded,
                line_mask,
                model.prototypes,
                margin=args.proto_margin,
                device=args.device,
            )

            loss = file_loss + args.line_loss_weight * line_loss + args.proto_loss_weight * proto_loss

            train_losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        results['train_loss'].append(np.mean(train_losses))

        model.eval()
        val_losses = []
        outputs, outputs_labels = [], []
        all_line_probs, all_line_labels = [], []

        with torch.no_grad():
            for batch in tqdm(valid_dl, total=len(valid_dl), desc='Valid Loop'):
                inputs = [item[0] for item in batch]
                labels = [item[1] for item in batch]
                line_labels = [item[2] for item in batch]
                labels = torch.tensor(labels)

                max_lines = max(ll.size(0) for ll in line_labels)
                line_labels_padded = torch.zeros(len(line_labels), max_lines)
                line_mask = torch.zeros(len(line_labels), max_lines, dtype=torch.bool)
                for i, ll in enumerate(line_labels):
                    n = ll.size(0)
                    line_labels_padded[i, :n] = ll
                    line_mask[i, :n] = True

                cov_inputs = []
                for item in inputs:
                    codeemb = codebert(item.to(args.device), attention_mask=item.to(args.device).ne(1))
                    cov_inputs.append(codeemb.pooler_output)

                file_output, line_output, sent_lengths, file_features, raw_line_features = model(
                    cov_inputs, return_line_features=True
                )
                outputs.append(sig(file_output))
                outputs_labels.append(labels)

                file_loss = criterion(file_output.reshape(len(labels)), labels.to(args.device))

                line_loss = line_criterion(line_output[line_mask],
                                           line_labels_padded[line_mask].to(args.device))

                proto_loss = prototype_contrastive_loss(
                    raw_line_features,
                    line_labels_padded,
                    line_mask,
                    model.prototypes,
                    margin=args.proto_margin,
                    device=args.device,
                )

                val_loss = file_loss + args.line_loss_weight * line_loss + args.proto_loss_weight * proto_loss

                for j, length in enumerate(sent_lengths):
                    probs = sig(line_output[j, :length]).cpu().numpy()
                    true = line_labels_padded[j, :length].cpu().numpy().astype(int)
                    all_line_probs.extend(probs)
                    all_line_labels.extend(true)

                val_losses.append(val_loss.item())
                torch.cuda.empty_cache()

        results['val_loss'].append(np.mean(val_losses))

        y_prob = torch.cat(outputs).cpu().numpy()
        y_gt = torch.cat(outputs_labels).cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)

        results['val_auc'].append(roc_auc_score(y_gt, y_prob))
        results['val_ba'].append(balanced_accuracy_score(y_gt, y_pred))
        results['val_mcc'].append(matthews_corrcoef(y_gt, y_pred))

        if len(all_line_probs) > 0:
            sorted_indices = np.argsort(all_line_probs)[::-1]
            sorted_labels = np.array(all_line_labels)[sorted_indices]
            total_pos = np.sum(all_line_labels)

            top_k = int(len(sorted_labels) * 0.2)
            recall = np.sum(sorted_labels[:top_k]) / max(total_pos, 1)
            results['val_recall'].append(recall)

            target = int(0.2 * total_pos)
            effort = np.where(np.cumsum(sorted_labels) >= target)[0]
            effort_ratio = (effort[0] + 1) / len(sorted_labels) if len(effort) > 0 else 1.0
            results['val_effort'].append(effort_ratio)

            first_pos = np.where(sorted_labels == 1)[0]
            ifa = first_pos[0] if len(first_pos) > 0 else len(sorted_labels)
            results['val_ifa'].append(ifa)

        print(f"Epoch {epoch:2d} | "
              f"Train Loss: {results['train_loss'][-1]:.4f} | "
              f"Val Loss: {results['val_loss'][-1]:.4f} | "
              f"AUC: {results['val_auc'][-1]:.4f} | "
              f"BA: {results['val_ba'][-1]:.4f} | "
              f"MCC: {results['val_mcc'][-1]:.4f} | "
              f"Recall@20%: {results['val_recall'][-1]:.4f} | "
              f"Effort@20%: {results['val_effort'][-1]:.4f} | "
              f"IFA: {results['val_ifa'][-1]:.0f}")

        if results['val_auc'][-1] >= best_auc:
            best_auc = results['val_auc'][-1]
            best_model = copy.deepcopy(model)

        results_df = pd.DataFrame(results)
        os.makedirs(args.loss_dir, exist_ok=True)
        results_df.to_csv(args.loss_dir + dataset_name + '-results.csv', index=False)

        os.makedirs(os.path.join(args.save_model_dir, dataset_name), exist_ok=True)
        torch.save(best_model.state_dict(), os.path.join(args.save_model_dir + dataset_name + '/', 'epoch-' + str(epoch) + '-model.pth'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_lvl_gt', type=str, default='datasets/preprocessed_data/')
    parser.add_argument('-save_model_dir', type=str, default='output/model/FineLPDP/')
    parser.add_argument('-loss_dir', type=str, default='output/loss/FineLPDP/')
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-num_epochs', type=int, default=20)
    parser.add_argument('-max_grad_norm', type=int, default=5)
    parser.add_argument('-max_train_LOC', type=int, default=900)
    parser.add_argument('-dropout', type=float, default=0.2)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-line_loss_weight', type=float, default=0.5)
    parser.add_argument('-proto_loss_weight', type=float, default=0.1,
                        help='weight for prototype contrastive loss')
    parser.add_argument('-proto_margin', type=float, default=1.0,
                        help='minimum sq-L2 distance for push/separation in prototype loss')
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-model_type', type=str, default='roberta')
    parser.add_argument('-model_name_or_path', type=str, default='./codebert-base')
    parser.add_argument('-config_name', type=str, default=None)
    parser.add_argument('-tokenizer_name', type=str, default='./codebert-base')
    parser.add_argument('-cache_dir', type=str, default=None)
    parser.add_argument('-block_size', type=int, default=75)
    parser.add_argument('-do_lower_case', action='store_true')
    parser.add_argument('-dataset', type=str, default=None, help='Specific dataset to run (default: all)')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    if args.dataset:
        dataset_names = [args.dataset]
    else:
        dataset_names = list(all_releases.keys())

    for dataset_name in dataset_names:
        print(f'\n{"#" * 60}')
        print(f'Running (prototype contrastive loss) on {dataset_name}')
        print(f'{"#" * 60}')
        train_model(args, dataset_name)


if __name__ == "__main__":
    main()
