"""
Kuairand experiment entry for ALoHA-Rec (ADLS) with multiple backbones.

Dataset: Kuairand (5 domains based on tab: 0, 1, 2, 3, 4)
Features:
  - Dense: follow_user_num, fans_user_num, friend_user_num, register_days
  - Sparse: user_id, video_id, and other categorical features
Target: is_click (binary CTR prediction)

Supported Backbones:
- fcn: Fully Connected Network (MLP)
- deepfm: DeepFM (FM + Deep Network)
- dcn: Deep & Cross Network
- xdeepfm: xDeepFM (CIN + Deep Network)
- autoint: AutoInt (Self-Attention + Deep Network)
- epnet: EPNet (Scenario-aware Gating + Deep Network)

Usage:
    python kuairand_main.py --backbone all
    python kuairand_main.py --backbone fcn,deepfm,dcn
    python kuairand_main.py --backbone epnet --k_layers 2 --k_experts 2
"""

import sys
sys.path.append("..")
import os
import time
import json
import csv
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm
import math

from src.basic.features import DenseFeature, SparseFeature
from src.models.multi_domain.adls import ADLS
from src.trainers.adls_trainer import ALoHATrainer

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def convert_numeric(val):
    """Forced conversion to int."""
    return int(val)


def load_kuairand(data_path='./data/kuairand'):
    """Load Kuairand multi-domain dataset."""
    data = pd.read_csv(os.path.join(data_path, 'kuairand.csv'))

    valid_tabs = [0, 1, 2, 3, 4]
    data = data[data["tab"].apply(lambda x: x in valid_tabs)]
    data.reset_index(drop=True, inplace=True)

    data.rename(columns={'tab': "domain_indicator"}, inplace=True)
    domain_num = data.domain_indicator.nunique()

    col_names = data.columns.to_list()

    dense_features = ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]
    useless_features = ["play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time"]
    target = "is_click"

    sparse_features = [col for col in col_names if col not in dense_features and
                       col not in useless_features and col not in [target, 'domain_indicator']]

    for feature in dense_features:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_features:
        data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])

    for feature in useless_features:
        if feature in data.columns:
            del data[feature]

    lbe_domain = LabelEncoder()
    data['domain_indicator'] = lbe_domain.fit_transform(data['domain_indicator'])

    for feat in tqdm(sparse_features, desc="Encoding sparse features"):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=int(data[feature_name].max())+1, embed_dim=16)
                   for feature_name in sparse_features]

    y = data[target].astype(np.float32)
    X = data.drop(columns=[target])

    return dense_feas, sparse_feas, X, y, domain_num


class CTRDataset(Dataset):
    """CTR prediction dataset with domain indicator."""

    def __init__(self, X_df, y, feature_defs):
        self.X = X_df.reset_index(drop=True)
        self.y = np.asarray(y).astype(np.float32)
        self.feats = feature_defs
        self.names = [f.name for f in feature_defs]
        assert 'domain_indicator' in self.X.columns

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        x_dict = {}
        for f in self.feats:
            v = row[f.name]
            if hasattr(f, 'vocab_size'):
                x_dict[f.name] = torch.tensor(int(v), dtype=torch.long)
            else:
                x_dict[f.name] = torch.tensor(float(v), dtype=torch.float32)
        x_dict["domain_indicator"] = torch.tensor(int(row['domain_indicator']), dtype=torch.long)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return x_dict, label


class DomainBalancedBatchSampler(Sampler):
    """Ensures each batch contains samples from multiple domains."""

    def __init__(self, domain_ids, batch_size, m_domains=5, drop_last=True):
        self.batch_size = int(batch_size)
        self.m = int(m_domains)
        self.drop_last = bool(drop_last)

        self.indices = torch.arange(len(domain_ids)).tolist()
        buckets = {}
        for i, d in enumerate(domain_ids):
            buckets.setdefault(int(d), []).append(i)

        self.domains = sorted(buckets.keys())
        self.buckets = {k: torch.tensor(v, dtype=torch.long) for k, v in buckets.items()}
        self.N = len(self.indices)
        self.num_batches = (self.N // self.batch_size) if drop_last else math.ceil(self.N / self.batch_size)

        self.ptr = {}
        for k in self.domains:
            perm = torch.randperm(len(self.buckets[k]))
            self.buckets[k] = self.buckets[k][perm]
            self.ptr[k] = 0

    def __iter__(self):
        B = self.batch_size
        m = min(self.m, len(self.domains))

        for _ in range(self.num_batches):
            chosen = torch.randperm(len(self.domains))[:m].tolist()
            per = max(1, B // m)
            batch = []

            for ci in chosen:
                k = self.domains[ci]
                need = per
                while need > 0:
                    if self.ptr[k] >= len(self.buckets[k]):
                        perm = torch.randperm(len(self.buckets[k]))
                        self.buckets[k] = self.buckets[k][perm]
                        self.ptr[k] = 0
                    batch.append(int(self.buckets[k][self.ptr[k]]))
                    self.ptr[k] += 1
                    need -= 1

            rem = B - len(batch)
            j = 0
            while rem > 0:
                k = self.domains[chosen[j % m]]
                if self.ptr[k] >= len(self.buckets[k]):
                    perm = torch.randperm(len(self.buckets[k]))
                    self.buckets[k] = self.buckets[k][perm]
                    self.ptr[k] = 0
                batch.append(int(self.buckets[k][self.ptr[k]]))
                self.ptr[k] += 1
                rem -= 1
                j += 1

            yield batch

    def __len__(self):
        return self.num_batches


SUPPORTED_BACKBONES = ['fcn', 'deepfm', 'dcn', 'xdeepfm', 'autoint', 'epnet']

BACKBONE_CONFIGS = {
    'fcn': {'fcn_dims': [256, 128, 64]},
    'deepfm': {'deep_dims': [256, 128, 64]},
    'dcn': {'num_cross_layers': 3, 'deep_dims': [256, 128, 64]},
    'xdeepfm': {'cin_layer_sizes': [128, 128], 'deep_dims': [256, 128, 64]},
    'autoint': {'num_heads': 2, 'num_attention_layers': 2, 'deep_dims': [256, 128, 64]},
    'epnet': {'fcn_dims': [128, 64, 32], 'tower_dims': []},
}


def parse_backbones(arg_str):
    """Parse backbone argument."""
    s = arg_str.strip().lower()
    if s == 'all':
        return SUPPORTED_BACKBONES
    items = [t.strip() for t in s.split(',') if t.strip()]
    return [it for it in items if it in SUPPORTED_BACKBONES] or ['fcn']


def run_one_backbone(backbone, features, domain_num, train_loader, val_loader, test_loader,
                     device, args, experiment_root, result_filepath):
    """Run experiment with one backbone."""
    print(f"\n{'='*80}")
    print(f"Training backbone: {backbone.upper()}")
    print(f"{'='*80}")

    artifact_dir = os.path.join(experiment_root, backbone)
    os.makedirs(artifact_dir, exist_ok=True)

    backbone_config = dict(BACKBONE_CONFIGS[backbone])
    tower_dims = backbone_config.pop('tower_dims', [16])

    model = ADLS(
        features=features,
        domain_num=domain_num,
        backbone_type=backbone,
        num_experts=domain_num,
        lora_rank=args.rank,
        lora_alpha=1.0,
        num_lora_layers=args.num_lora_layers,
        k_layers=args.k_layers,
        k_experts=args.k_experts,
        train_lora_in_stage1=False,
        stage1_lora_mode='none',
        ema_decay=args.ema_decay,
        beta=args.beta,
        bottom_params={
            "dims": backbone_config.get('deep_dims', backbone_config.get('fcn_dims', [256, 128, 64])),
            "dropout": 0.2, "activation": "relu"
        },
        tower_params={"dims": tower_dims, "dropout": 0.2, "activation": "relu"},
        **{k: v for k, v in backbone_config.items() if k not in ['deep_dims', 'fcn_dims']}
    ).to(device)

    trainer = ALoHATrainer(
        model=model, dataset_name=f"kuairand_{backbone}", device=device,
        save_dir='./checkpoints', verbose=True,
        stage1_config=dict(
            epochs=30,
            lr=1e-3,
            weight_decay=1e-5,
            patience=5,
            gradient_clip=1.0,
            label_smoothing=0.0,
            gaba_update_freq=10,
            gaba_compute_freq=100,
        ),
        stage2_config=dict(
            epochs=50,
            lr_inter_router=5e-4,
            lr_gate=1e-3,
            lr_intra_router=5e-4,
            lr_lora=5e-4,
            weight_decay=1e-5,
            patience=8,
            lambda_mutual=0.1,
            lambda_sep=0.05,
            lambda_dir=0.1,
            gradient_clip=1.0,
            label_smoothing=0.0,
        )
    )

    t0 = time.time()
    results = trainer.run_full_training(train_loader, val_loader, test_loader, result_filepath, backbone, artifact_dir)
    elapsed = time.time() - t0

    summary = {'backbone': backbone, 'elapsed_sec': elapsed, 'stage1': results['stage1'],
               'stage2': results['stage2'], 'improvement': results['improvement']}
    with open(os.path.join(artifact_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nBackbone {backbone} completed in {elapsed:.1f}s")
    print(f"  Stage1 Test AUC: {results['stage1'].get('test_auc', 0):.6f}")
    print(f"  Stage2 Test AUC: {results['stage2'].get('test_auc', 0):.6f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='ALoHA-Rec Kuairand Experiment')
    parser.add_argument('--data_path', type=str, default='./data/kuairand')
    parser.add_argument('--backbone', type=str, default='all')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--num_lora_layers', type=int, default=3)
    parser.add_argument('--k_layers', type=int, default=2)
    parser.add_argument('--k_experts', type=int, default=2)
    parser.add_argument('--ema_decay', type=float, default=0.9, help='EMA decay for gradient smoothing')
    parser.add_argument('--beta', type=float, default=0.9, help='EMA decay for per-domain self-reliance')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--m_domains', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    print("Loading Kuairand dataset...")
    dense_feas, sparse_feas, X, y, domain_num = load_kuairand(args.data_path)
    features = dense_feas + sparse_feas

    print(f"\nDataset Info:")
    print(f"  Total samples: {len(y)}")
    print(f"  Domains: {domain_num}")
    print(f"  Dense features: {len(dense_feas)}")
    print(f"  Sparse features: {len(sparse_feas)}")
    print(f"  Positive ratio: {y.mean():.4f}")

    idx = np.arange(len(y))
    np.random.shuffle(idx)
    n_train, n_val = int(len(idx) * 0.8), int(len(idx) * 0.1)
    X_train, y_train = X.iloc[idx[:n_train]], y.iloc[idx[:n_train]]
    X_val, y_val = X.iloc[idx[n_train:n_train+n_val]], y.iloc[idx[n_train:n_train+n_val]]
    X_test, y_test = X.iloc[idx[n_train+n_val:]], y.iloc[idx[n_train+n_val:]]

    print(f"\nData Split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    train_ds = CTRDataset(X_train, y_train, features)
    val_ds = CTRDataset(X_val, y_val, features)
    test_ds = CTRDataset(X_test, y_test, features)

    train_sampler = DomainBalancedBatchSampler(
        X_train['domain_indicator'].astype(int).tolist(),
        args.batch_size,
        min(args.m_domains, domain_num),
        True
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_root = os.path.join(args.output_dir, f'kuairand_aloha_rec_{timestamp}')
    os.makedirs(experiment_root, exist_ok=True)
    result_filepath = os.path.join(experiment_root, 'all_results.txt')

    with open(result_filepath, 'w') as f:
        f.write(f"ALoHA-Rec Kuairand Experiment - {datetime.now()}\n{'='*80}\n")
        f.write("Configuration:\n")
        f.write(f"  LoRA Rank: {args.rank}\n")
        f.write(f"  Num LoRA Layers: {args.num_lora_layers}\n")
        f.write(f"  K_layers: {args.k_layers}\n")
        f.write(f"  K_experts: {args.k_experts}\n")
        f.write(f"  EMA decay: {args.ema_decay}\n")
        f.write(f"  Beta (self-reliance EMA): {args.beta}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Device: {args.device}\n")
        f.write("\n" + "="*80 + "\n\n")

    selected = parse_backbones(args.backbone)
    print(f"\nRunning {len(selected)} backbone(s): {selected}")

    all_summaries = []
    for bb in selected:
        try:
            summary = run_one_backbone(
                bb, features, domain_num, train_loader, val_loader, test_loader,
                args.device, args, experiment_root, result_filepath
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"[ERROR] {bb}: {e}")
            import traceback
            traceback.print_exc()

    with open(os.path.join(experiment_root, 'all_backbones_summary.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2, default=str)

    csv_path = os.path.join(experiment_root, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['backbone', 'elapsed_sec', 'stage1_test_auc', 'stage2_test_auc', 'improvement'])
        for s in all_summaries:
            writer.writerow([
                s['backbone'],
                f"{s['elapsed_sec']:.1f}",
                f"{s['stage1'].get('test_auc', 0):.6f}",
                f"{s['stage2'].get('test_auc', 0):.6f}",
                f"{s['improvement'].get('test_auc', 0):+.6f}"
            ])

    print(f"\n{'='*80}")
    print(f"Experiment completed! Results saved to: {experiment_root}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()