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


def run_single(framework, extractor, dataset_path, epoch, lr, batch_size,
               weight_decay, device, save_dir, seed, stage2_epoch):
    torch.manual_seed(seed)
    dataset_name = "amazon_5_core"

    dense_feas, sparse_feas, domain_feat, x, y, domain_num = get_amazon_data_dict(dataset_path)

    if framework == 'sharedbottom':
        features = dense_feas + sparse_feas
        bottom_params = {"dims": [128]}
        tower_params = {"dims": [8]}
    elif framework == 'epnet':
        features = [domain_feat] + sparse_feas + dense_feas
        bottom_params = {"dims": [128, 64, 32]}
        tower_params = None
    else:
        raise ValueError(f"Unknown framework: {framework}")

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
        stage1_n_epoch=epoch, stage1_earlystop_patience=5,
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
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
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
                epoch=args.epoch, lr=args.learning_rate,
                batch_size=args.batch_size, weight_decay=args.weight_decay,
                device=args.device, save_dir=args.save_dir,
                seed=args.seed, stage2_epoch=args.stage2_epoch,
            )


if __name__ == '__main__':
    main()