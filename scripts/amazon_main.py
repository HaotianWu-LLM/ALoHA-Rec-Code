import os
import sys
import csv
import argparse
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from src.basic.features import DenseFeature, SparseFeature
from src.utils.data import DataGenerator
from src.models.multi_domain import ADLS
from src.trainers.adls_trainer import ALoHATrainer


def get_amazon_data_dict(data_path):
    data = pd.read_csv(os.path.join(data_path, 'amazon.csv'))
    domain_num = 3
    col_names = data.columns.values.tolist()
    dense_cols = []
    sparse_cols = [c for c in col_names if c not in dense_cols and c not in ['label', 'domain_indicator']]

    if dense_cols:
        sca = MinMaxScaler()
        data[dense_cols] = sca.fit_transform(data[dense_cols])
    for feat in tqdm(sparse_cols):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in sparse_cols]

    domain_vocab = int(data['domain_indicator'].max() + 1)
    domain_feat = SparseFeature('domain_indicator', vocab_size=domain_vocab, embed_dim=16)

    y = data["label"]
    del data["label"]
    x = data
    return dense_feas, sparse_feas, domain_feat, x, y, domain_num


def get_extractor_hparams(framework, extractor):
    """Per-extractor overrides. Non-FCN extractors have ~10-100x more parameters
    than FCN and need tighter regularization + earlier stopping to avoid
    overfitting on top of PEP gating, which is already a strong personalization
    mechanism."""
    if extractor == 'fcn':
        if framework == 'sharedbottom':
            return {
                'bottom_dims': [128], 'bottom_dropout': 0.0,
                'tower_dims': [8],
                'lr': 1e-3, 'weight_decay': 1e-5, 'patience': 5,
            }
        else:
            return {
                'bottom_dims': [128, 64, 32], 'bottom_dropout': 0.0,
                'tower_dims': None,
                'lr': 1e-3, 'weight_decay': 1e-5, 'patience': 5,
            }

    if framework == 'sharedbottom':
        return {
            'bottom_dims': [64, 32], 'bottom_dropout': 0.0,
            'tower_dims': [8],
            'lr': 1e-3, 'weight_decay': 1e-4, 'patience': 2,
        }
    else:
        return {
            'bottom_dims': [64, 32], 'bottom_dropout': 0.0,
            'tower_dims': None,
            'lr': 1e-3, 'weight_decay': 1e-4, 'patience': 2,
        }


def run_single(framework, extractor, dataset_path, epoch, batch_size,
               device, save_dir, seed, stage2_epoch,
               lr_override=None, wd_override=None, patience_override=None):
    torch.manual_seed(seed)
    dataset_name = "amazon_5_core"

    dense_feas, sparse_feas, domain_feat, x, y, domain_num = get_amazon_data_dict(dataset_path)

    hp = get_extractor_hparams(framework, extractor)
    lr = lr_override if lr_override is not None else hp['lr']
    weight_decay = wd_override if wd_override is not None else hp['weight_decay']
    patience = patience_override if patience_override is not None else hp['patience']

    if framework == 'sharedbottom':
        features = dense_feas + sparse_feas
        bottom_params = {"dims": hp['bottom_dims'], "dropout": hp['bottom_dropout']}
        tower_params = {"dims": hp['tower_dims']}
    elif framework == 'epnet':
        features = [domain_feat] + sparse_feas + dense_feas
        bottom_params = {"dims": hp['bottom_dims'], "dropout": hp['bottom_dropout']}
        tower_params = None
    else:
        raise ValueError(f"Unknown framework: {framework}")

    print(f"[{framework}/{extractor}] bottom_dims={hp['bottom_dims']} "
          f"dropout={hp['bottom_dropout']} lr={lr} wd={weight_decay} "
          f"patience={patience}")

    dg = DataGenerator(x, y)
    train_dl, val_dl, test_dl = dg.generate_dataloader(
        split_ratio=[0.8, 0.1], batch_size=batch_size
    )

    model = ADLS(
        features=features, domain_num=domain_num,
        framework=framework, extractor=extractor,
        bottom_params=bottom_params, tower_params=tower_params,
        lora_rank=8, num_experts=4,
    )

    trainer = ALoHATrainer(
        model=model, data_set_type=dataset_name, device=device, model_path=save_dir,
        stage1_lr=lr, stage1_weight_decay=weight_decay,
        stage1_n_epoch=epoch, stage1_earlystop_patience=patience,
        stage1_scheduler_step=4, stage1_scheduler_gamma=0.95,
        stage2_n_epoch=stage2_epoch,
    )
    stage1_metrics, stage2_metrics = trainer.run_full_training(
        train_dl, val_dl, test_dl, batch_size=batch_size
    )

    print(f"\n[{framework}/{extractor}] Stage1:", stage1_metrics)
    print(f"[{framework}/{extractor}] Stage2:", stage2_metrics)

    out_csv = f"adls_{framework}_{extractor}_{dataset_name}_{seed}.csv"
    with open(out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['framework', 'extractor', 'seed', 'stage',
                         'auc', 'log', 'auc0', 'log0', 'auc1', 'log1', 'auc2', 'log2'])
        for stage_name, m in [('stage1', stage1_metrics), ('stage2', stage2_metrics)]:
            da = m.get('domain_auc', [None] * domain_num)
            dl = m.get('domain_logloss', [None] * domain_num)
            writer.writerow([framework, extractor, str(seed), stage_name,
                             m.get('test_auc'), m.get('test_logloss'),
                             da[0] if len(da) > 0 else None, dl[0] if len(dl) > 0 else None,
                             da[1] if len(da) > 1 else None, dl[1] if len(dl) > 1 else None,
                             da[2] if len(da) > 2 else None, dl[2] if len(dl) > 2 else None])
    return stage1_metrics, stage2_metrics


def main():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/amazon_5_core")
    parser.add_argument('--framework', default='sharedbottom',
                        choices=['sharedbottom', 'epnet', 'all'])
    parser.add_argument('--extractor', default='fcn',
                        choices=['fcn', 'dcn', 'deepfm', 'xdeepfm', 'autoint', 'all'])
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--stage2_epoch', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Override extractor-specific lr when set.')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Override extractor-specific weight_decay when set.')
    parser.add_argument('--patience', type=int, default=None,
                        help='Override extractor-specific earlystop patience when set.')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./ckpt')
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()

    frameworks = ['sharedbottom', 'epnet'] if args.framework == 'all' else [args.framework]
    extractors = ['fcn', 'dcn', 'deepfm', 'xdeepfm', 'autoint'] if args.extractor == 'all' else [args.extractor]

    for fw in frameworks:
        for ex in extractors:
            print(f"\n{'=' * 60}\nRunning framework={fw}, extractor={ex}\n{'=' * 60}")
            run_single(
                framework=fw, extractor=ex,
                dataset_path=args.dataset_path,
                epoch=args.epoch,
                batch_size=args.batch_size,
                device=args.device, save_dir=args.save_dir,
                seed=args.seed, stage2_epoch=args.stage2_epoch,
                lr_override=args.learning_rate,
                wd_override=args.weight_decay,
                patience_override=args.patience,
            )


if __name__ == '__main__':
    main()