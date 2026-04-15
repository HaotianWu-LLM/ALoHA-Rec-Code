import sys

sys.path.append("..")
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scenario_wise_rec.basic.features import DenseFeature, SparseFeature
from scenario_wise_rec.trainers import CTRTrainer
from scenario_wise_rec.utils.data import DataGenerator
from scenario_wise_rec.models.multi_domain import Star, MMOE, PLE, SharedBottom, AdaSparse, Sarnet, M2M, AdaptDHM, \
    EPNet, PPNet, M3oE, HamurSmall


def get_healthcare_data_dict(data_path='./data/healthcare'):
    # Load data
    data = pd.read_csv(data_path + '/healthcare.csv')

    # Domain settings
    # Disease_Domain: 0 (Diabetes), 1 (Blood Pressure), 2 (Overweight)
    domain_num = 3
    data.rename(columns={'Disease_Domain': 'domain_indicator'}, inplace=True)

    # Feature Classification based on NHANES
    target = "MCQ220"

    # Categorical features (NHANES Demographics)
    sparse_cols = ['gender', 'race']

    # Continuous features (Body measures, HEI scores, LS7 scores)
    # Exclude target, domain, and sparse columns to get dense columns
    exclude_cols = [target, 'domain_indicator'] + sparse_cols
    dense_cols = [col for col in data.columns if col not in exclude_cols]

    # Preprocessing
    # 1. Dense Features: MinMaxScaler
    for feature in dense_cols:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_cols:
        sca = MinMaxScaler()
        data[dense_cols] = sca.fit_transform(data[dense_cols])

    # 2. Sparse Features: LabelEncoder
    for feat in tqdm(sparse_cols):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))

    # Feature Objects
    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].nunique() + 1, embed_dim=16) for col in sparse_cols]

    y = data[target]
    del data[target]
    x = data
    return dense_feas, sparse_feas, x, y, domain_num


def get_healthcare_data_dict_adasparse(data_path='./data/healthcare'):
    data = pd.read_csv(data_path + '/healthcare.csv')

    domain_num = 3
    scenario_fea_num = 1
    data.rename(columns={'Disease_Domain': 'domain_indicator'}, inplace=True)

    target = "MCQ220"
    scenario_cols = ['domain_indicator']
    sparse_cols = ['gender', 'race']

    exclude_cols = [target] + scenario_cols + sparse_cols
    dense_cols = [col for col in data.columns if col not in exclude_cols]

    # Preprocessing
    for feature in dense_cols:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_cols:
        sca = MinMaxScaler()
        data[dense_cols] = sca.fit_transform(data[dense_cols])

    for feat in tqdm(sparse_cols):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    print("scenario_cols:%d sparse cols:%d dense cols:%d" % (len(scenario_cols), len(sparse_cols), len(dense_cols)))

    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].nunique() + 1, embed_dim=16) for col in sparse_cols]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_cols]

    y = data[target]
    del data[target]
    x = data

    return (dense_feas, sparse_feas, scenario_feas, scenario_fea_num,
            x, y, domain_num)


def get_healthcare_data_dict_ppnet(data_path='./data/healthcare'):
    data = pd.read_csv(data_path + '/healthcare.csv')

    domain_num = 3
    scenario_fea_num = 1
    data.rename(columns={'Disease_Domain': 'domain_indicator'}, inplace=True)

    target = "MCQ220"
    scenario_cols = ['domain_indicator']

    # PPNet usually requires some ID features, but healthcare data is not user-item interaction based.
    # We will treat sparse features as ID-like features or keep sparse separate if strictly following architecture.
    # In Amazon/Douban baseline, user/item IDs are separated.
    # Here we don't have high-cardinality IDs (SEQN was dropped).
    # We will treat 'gender' and 'race' as standard sparse features, and leave id_cols empty or
    # use sparse features as id_features if the model strictly demands it.
    # Looking at PPNet implementation, it takes id_features separately.
    # We will simulate this by assigning sparse_cols to id_cols to ensure pipeline consistency,
    # or keep them empty if model allows. Let's assign sparse to sparse and keep IDs empty if possible,
    # BUT PPNet usually uses IDs for gating. Let's use 'race' as a proxy for a group ID if needed,
    # but strictly speaking, standard sparse features are fine.
    # However, to strictly align with the function signature in amazon_ppnet:

    id_cols = []  # No high cardinality IDs
    sparse_cols = ['gender', 'race']

    exclude_cols = [target] + scenario_cols + sparse_cols + id_cols
    dense_cols = [col for col in data.columns if col not in exclude_cols]

    for feature in dense_cols:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_cols:
        sca = MinMaxScaler()
        data[dense_cols] = sca.fit_transform(data[dense_cols])

    for feat in tqdm(sparse_cols):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # PPNet helper in amazon baseline encodes ID cols. We have none, but let's keep the loop structure.
    for feat in tqdm(id_cols):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    print("scenario_cols:%d sparse cols:%d dense cols:%d id cols:%d" % (
        len(scenario_cols), len(sparse_cols), len(dense_cols), len(id_cols)))

    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].nunique() + 1, embed_dim=16) for col in sparse_cols]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_cols]
    id_feas = [SparseFeature(col, vocab_size=data[col].nunique() + 1, embed_dim=16) for col in id_cols]

    y = data[target]
    del data[target]
    x = data
    return (dense_feas, sparse_feas, scenario_feas, id_feas, scenario_fea_num,
            x, y, domain_num)


def convert_numeric(val):
    """
    Forced conversion
    """
    try:
        return float(val)
    except:
        return 0.0


def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    torch.manual_seed(seed)
    dataset_name = "Healthcare"

    if model_name in ["adasparse", "m2m", "adaptdhm", "epnet"]:
        (dense_feas, sparse_feas, scenario_feas, scenario_fea_num, x, y,
         domain_num) = get_healthcare_data_dict_adasparse(
            dataset_path)
    elif model_name == "ppnet":
        (dense_feas, sparse_feas, scenario_feas, id_feas, scenario_fea_num, x, y,
         domain_num) = get_healthcare_data_dict_ppnet(dataset_path)
    else:
        dense_feas, sparse_feas, x, y, domain_num = get_healthcare_data_dict(dataset_path)

    dg = DataGenerator(x, y)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.8, 0.1],
                                                                               batch_size=batch_size)
    if model_name == "star":
        model = Star(dense_feas + sparse_feas, domain_num, fcn_dims=[128, 64, 32], aux_dims=[32])
    elif model_name == "SharedBottom":
        model = SharedBottom(dense_feas + sparse_feas, domain_num, bottom_params={"dims": [128]},
                             tower_params={"dims": [8]})
    elif model_name == "MMOE":
        model = MMOE(dense_feas + sparse_feas, domain_num, n_expert=domain_num, expert_params={"dims": [16]},
                     tower_params={"dims": [8]})
    elif model_name == "PLE":
        model = PLE(dense_feas + sparse_feas, domain_num, n_level=1, n_expert_specific=2, n_expert_shared=1,
                    expert_params={"dims": [16]}, tower_params={"dims": [8]})
    elif model_name == "adasparse":
        model = AdaSparse(sce_features=scenario_feas, agn_features=sparse_feas, form='Fusion', epsilon=1e-2, alpha=1.0,
                          delta_alpha=1e-4, mlp_params={"dims": [32, 32], "dropout": 0.2, "activation": "relu"})
    elif model_name == "sarnet":
        model = Sarnet(features=sparse_feas, domain_num=domain_num, domain_shared_expert_num=8,
                       domain_specific_expert_num=2)
    elif model_name == "m2m":
        model = M2M(features=sparse_feas + scenario_feas, domain_feature=scenario_feas, domain_num=domain_num,
                    num_experts=4, expert_output_size=16,
                    transformer_dims={"num_encoder_layers": 2, "num_decoder_layers": 2, "dim_feedforward": 16})
    elif model_name == "adaptdhm":
        model = AdaptDHM(features=sparse_feas + scenario_feas, fcn_dims=[64, 64], cluster_num=3, beta=0.9,
                         device=device)
    elif model_name == "epnet":
        model = EPNet(sce_features=scenario_feas, agn_features=sparse_feas + dense_feas, fcn_dims=[128, 64, 32])
    elif model_name == "ppnet":
        model = PPNet(id_features=id_feas, agn_features=sparse_feas + dense_feas + scenario_feas, domain_num=domain_num,
                      fcn_dims=[128, 64, 32])
    elif model_name == "m3oe":
        model = M3oE(features=dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[128, 64, 64, 32], expert_num=4,
                     exp_d=1, exp_t=1, bal_d=1, bal_t=1, device=device)
    elif model_name == "hamur":
        model = HamurSmall(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128], hyper_dims=[64], k=35)

    ctr_trainer = CTRTrainer(model, dataset_name, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},
                             n_epoch=epoch, earlystop_patience=5, device=device, model_path=save_dir,
                             scheduler_params={"step_size": 4, "gamma": 0.95})

    ctr_trainer.fit(train_dataloader, val_dataloader)

    domain_logloss, domain_auc, logloss, auc = ctr_trainer.evaluate_multi_domain_loss(ctr_trainer.model,
                                                                                      test_dataloader, domain_num)
    print(f'test auc: {auc} | test logloss: {logloss}')
    for d in range(domain_num):
        print(f'test domain {d} auc: {domain_auc[d]} | test domain {d} logloss: {domain_logloss[d]}')

    # Return results for the caller function
    return model_name, seed, auc, logloss, domain_auc, domain_logloss


if __name__ == '__main__':
    import argparse
    import warnings
    import os

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/healthcare")
    parser.add_argument('--model_name', default='star')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--save_dir', default='./chkpt')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run_all', action='store_true', help='Run all baseline models with multiple seeds')

    args = parser.parse_args()

    # Define all baseline models
    all_models = ["star", "SharedBottom", "MMOE", "PLE", "adasparse", "sarnet",
                  "m2m", "adaptdhm", "epnet", "ppnet", "m3oe", "hamur"]

    # Define seeds to use
    all_seeds = [0, 150, 368, 2025, 10000]

    # Output file for all results
    results_file = "all_healthcare_experiments_results.txt"

    if args.run_all:
        # Open results file
        with open(results_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("HEALTHCARE MULTI-DOMAIN EXPERIMENTS RESULTS\n")
            f.write("=" * 80 + "\n\n")

        # Run all models with all seeds
        for model_name in all_models:
            print(f"\n{'=' * 50}\nRunning model: {model_name}\n{'=' * 50}")

            # Write model header to results file
            with open(results_file, "a") as f:
                f.write(f"MODEL: {model_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Seed':<8} {'AUC':<10} {'LogLoss':<10} " +
                        f"{'AUC-D0 (Diabetes)':<18} {'Log-D0':<10} " +
                        f"{'AUC-D1 (HBP)':<18} {'Log-D1':<10} " +
                        f"{'AUC-D2 (Overweight)':<18} {'Log-D2':<10}\n")

            # Run model with each seed
            for seed in all_seeds:
                print(f"Running {model_name} with seed {seed}")
                try:
                    model_name_result, seed_result, auc, logloss, domain_auc, domain_logloss = main(
                        args.dataset_path, model_name, args.epoch, args.learning_rate,
                        args.batch_size, args.weight_decay, args.device,
                        args.save_dir, seed
                    )

                    # Write result to file
                    with open(results_file, "a") as f:
                        f.write(f"{seed:<8} {auc:<10.4f} {logloss:<10.4f} " +
                                f"{domain_auc[0]:<18.4f} {domain_logloss[0]:<10.4f} " +
                                f"{domain_auc[1]:<18.4f} {domain_logloss[1]:<10.4f} " +
                                f"{domain_auc[2]:<18.4f} {domain_logloss[2]:<10.4f}\n")
                except Exception as e:
                    print(f"Failed to run {model_name} with seed {seed}: {e}")
                    with open(results_file, "a") as f:
                        f.write(f"{seed:<8} FAILED: {str(e)}\n")

            # Add separator after each model
            with open(results_file, "a") as f:
                f.write("\n\n")
    else:
        # Run single experiment with specified parameters
        main(args.dataset_path, args.model_name, args.epoch, args.learning_rate,
             args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed)