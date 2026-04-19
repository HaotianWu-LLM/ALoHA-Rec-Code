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


def get_kuairand_data_dict(data_path):
    data = pd.read_csv(os.path.join(data_path, 'kuairand_sample.csv'))
    data = data[data["tab"].apply(lambda x: x in [1, 0, 4, 2, 6])]
    data.reset_index(drop=True, inplace=True)
    data.rename(columns={'tab': "domain_indicator"}, inplace=True)
    domain_num = int(data.domain_indicator.nunique())

    col_names = data.columns.to_list()
    dense_cols = ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]
    useless_features = ["play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time"]
    scenario_features = ["domain_indicator"]
    sparse_cols = [c for c in col_names if c not in dense_cols and
                   c not in useless_features and c not in ['is_click', 'domain_indicator']]

    for feature in dense_cols:
        data[feature] = data[feature].apply(lambda x: int(x))
    if dense_cols:
        sca = MinMaxScaler()
        data[dense_cols] = sca.fit_transform(data[dense_cols])

    for feature in useless_features:
        if feature in data.columns:
            del data[feature]
    for feature in scenario_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])
    for feature in tqdm(sparse_cols):
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])

    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].nunique(), embed_dim=16) for col in sparse_cols]
    domain_vocab = int(data['domain_indicator'].max() + 1)
    domain_feat = SparseFeature('domain_indicator', vocab_size=domain_vocab, embed_dim=16)

    y = data["is_click"]
    del data["is_click"]
    x = data
    return dense_feas, sparse_feas, domain_feat, x, y, domain_num


def run_single(framework, extractor, dataset_path, epoch, lr, batch_size,
               weight_decay, device, save_dir, seed, stage2_epoch):
    torch.manual_seed(seed)
    dataset_name = "kuairand"
    dense_feas, sparse_feas, domain_feat, x, y, domain_num = get_kuairand_data_dict(dataset_path)

    if framework == 'sharedbottom':
        features = dense_feas + sparse_feas
        bottom_params = {"dims": [128]}
        tower_params = {"dims": [64, 32]}
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
        stage1_n_epoch=epoch, stage1_earlystop_patience=4,
        stage1_scheduler_step=4, stage1_scheduler_gamma=0.75,
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
        header = ['framework', 'extractor', 'seed', 'stage', 'auc', 'log']
        for d in range(domain_num):
            header += [f'auc{d}', f'log{d}']
        writer.writerow(header)
        for stage_name, m in [('stage1', stage1_metrics), ('stage2', stage2_metrics)]:
            da = m.get('domain_auc', [None] * domain_num)
            dl = m.get('domain_logloss', [None] * domain_num)
            row = [framework, extractor, str(seed), stage_name,
                   m.get('test_auc'), m.get('test_logloss')]
            for d in range(domain_num):
                row += [da[d] if d < len(da) else None, dl[d] if d < len(dl) else None]
            writer.writerow(row)
    return stage1_metrics, stage2_metrics


def main():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/kuairand")
    parser.add_argument('--framework', default='sharedbottom', choices=['sharedbottom', 'epnet', 'all'])
    parser.add_argument('--extractor', default='fcn', choices=['fcn', 'dcn', 'deepfm', 'xdeepfm', 'autoint', 'all'])
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