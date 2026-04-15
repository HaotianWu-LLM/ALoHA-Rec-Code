"""
ALoHA-Rec Two-Stage Trainer

Changes vs previous version:
  1. _collect_domain_gradients: parameter-space gradients from last backbone layer
     (paper Eq.4 requires nabla_W L_m, not nabla_h L_m)
  2. train_stage1: per-domain averaged loss (paper Eq.3)
  3. train_stage1: optimizer only includes Stage 1 relevant params
     (excludes LoRA experts, routers, etc.)
"""

import os
import time
import json
import copy
import math
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from ..basic.callback import EarlyStopper


class LabelSmoothingBCELoss(nn.Module):

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets_smooth)


class CosineAnnealingWarmupScheduler:

    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_max, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.warmup_epochs > 0 and self.current_epoch <= self.warmup_epochs:
            lr = self.lr_min + (self.lr_max - self.lr_min) * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self):
        if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
            return self.lr_min + (self.lr_max - self.lr_min) * (self.current_epoch / max(self.warmup_epochs, 1))
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))


def _compute_multidomain_loss(criterion, logits, y, domain_ids):
    """Paper Eq.(3): per-domain averaged loss L_pre = (1/M) sum_m E[l(f(x,m), y)]."""
    loss = torch.tensor(0.0, device=logits.device)
    n_domains = 0
    for d in torch.unique(domain_ids):
        mask = (domain_ids == d)
        if mask.sum() > 0:
            loss = loss + criterion(logits[mask], y[mask].float())
            n_domains += 1
    return loss / max(n_domains, 1)


class ALoHATrainer:

    def __init__(self, model, dataset_name, device='cpu',
                 stage1_config=None, stage2_config=None,
                 save_dir='./checkpoints', verbose=True):
        self.model = model.to(device)
        self.dataset_name = dataset_name
        self.device = device
        self.save_dir = save_dir
        self.verbose = verbose
        os.makedirs(save_dir, exist_ok=True)

        default_stage1 = dict(
            epochs=50,
            lr=1e-3,
            lr_min=1e-6,
            warmup_epochs=0,
            weight_decay=1e-5,
            patience=10,
            gradient_clip=1.0,
            label_smoothing=0.0,
            gaba_update_freq=10,
            gaba_compute_freq=100,
        )
        self.stage1_config = {**default_stage1, **(stage1_config or {})}

        default_stage2 = dict(
            epochs=50,
            lr_inter_router=5e-4,
            lr_intra_router=5e-4,
            lr_lora=5e-4,
            lr_gate=1e-3,
            weight_decay=1e-5,
            patience=10,
            lambda_dir=0.1,
            gradient_clip=1.0,
            label_smoothing=0.0,
        )
        self.stage2_config = {**default_stage2, **(stage2_config or {})}

        self.criterion = nn.BCEWithLogitsLoss()
        self.global_step = 0
        self.best_R_benefit = None
        self.best_val_auc_stage1 = 0.0

        self.history = {
            'stage1': {'train_loss': [], 'val_auc': [], 'val_loss': [], 'lr': []},
            'stage2': {'train_loss': [], 'val_auc': [], 'val_loss': [],
                       'l_dir': []}
        }

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        all_targets, all_preds = [], []

        for x_dict, y in dataloader:
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
            y = y.to(self.device)
            logits = self.model(x_dict)
            probs = torch.sigmoid(logits)
            all_targets.extend(y.cpu().tolist())
            all_preds.extend(probs.cpu().tolist())

        if len(set(all_targets)) < 2:
            return 0.5, 1.0

        auc = roc_auc_score(all_targets, all_preds)
        ll = log_loss(all_targets, all_preds, labels=[0, 1])
        return float(auc), float(ll)

    @torch.no_grad()
    def evaluate_by_domain(self, dataloader):
        self.model.eval()
        domain_data = {}

        for x_dict, y in dataloader:
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
            y = y.to(self.device)
            domain_ids = x_dict["domain_indicator"].long()
            logits = self.model(x_dict)
            probs = torch.sigmoid(logits)

            for d in torch.unique(domain_ids):
                d_int = int(d.item())
                mask = (domain_ids == d)
                if d_int not in domain_data:
                    domain_data[d_int] = {'targets': [], 'preds': []}
                domain_data[d_int]['targets'].extend(y[mask].cpu().tolist())
                domain_data[d_int]['preds'].extend(probs[mask].cpu().tolist())

        results = {}
        for d, data in sorted(domain_data.items()):
            if len(data['targets']) > 0 and len(set(data['targets'])) > 1:
                auc = roc_auc_score(data['targets'], data['preds'])
                ll = log_loss(data['targets'], data['preds'], labels=[0, 1])
            else:
                auc, ll = 0.5, 1.0
            results[d] = {'auc': float(auc), 'logloss': float(ll), 'samples': len(data['targets'])}
        return results

    def _collect_domain_gradients(self, x_dict, y, criterion=None):
        """Collect per-domain PARAMETER-SPACE gradients for GABA (paper Eq.4).

        EPNet: uses backbone's shared output_layer (matching standalone).
        Others: uses domain towers (matching their Stage 1 forward path).
        """
        if criterion is None:
            criterion = self.criterion

        domain_ids = x_dict["domain_indicator"].long()
        domain_gradients = {}

        was_training = self.model.training
        self.model.eval()

        last_layer = self.model.backbone.backbone_layers[-1]
        target_params = [p for p in last_layer.parameters()]
        if not target_params:
            self.model.train(was_training)
            return domain_gradients

        input_emb = self.model.embedding(x_dict, self.model.features, squeeze_dim=True)
        emb_3d = self.model.embedding(x_dict, self.model.features, squeeze_dim=False)
        if input_emb.dim() == 1:
            input_emb = input_emb.unsqueeze(0)
        if emb_3d.dim() == 2:
            emb_3d = emb_3d.unsqueeze(0)

        is_epnet = (self.model.backbone_type == 'epnet')

        if is_epnet:
            # EPNet: backbone's own output_layer
            logits = self.model.backbone.forward_with_lora_experts(
                input_emb, None, mode='none',
                emb_3d=emb_3d, return_features=False, domain_id=domain_ids
            )
        else:
            # Others: backbone features → domain towers
            features, aux_logit = self.model.backbone.forward_with_lora_experts(
                input_emb, None, mode='none',
                emb_3d=emb_3d, return_features=True, domain_id=domain_ids
            )

        unique_domains = torch.unique(domain_ids)
        for idx, d in enumerate(unique_domains):
            mask = (domain_ids == d)
            if mask.sum() < 2:
                continue
            d_int = int(d.item())
            if d_int >= self.model.domain_num:
                continue

            if is_epnet:
                loss = criterion(logits[mask], y[mask].float())
            else:
                tower_out = self.model.domain_towers[d_int](features[mask]).squeeze(-1)
                d_logits = tower_out + aux_logit[mask]
                loss = criterion(d_logits, y[mask].float())

            if torch.isnan(loss):
                continue

            is_last = (idx == len(unique_domains) - 1)
            grads = torch.autograd.grad(
                loss, target_params,
                retain_graph=not is_last,
                create_graph=False
            )
            domain_grad = torch.cat([g.detach().flatten() for g in grads])

            if not torch.isnan(domain_grad).any() and torch.norm(domain_grad) > 1e-10:
                domain_gradients[d_int] = domain_grad

        self.model.train(was_training)
        return domain_gradients

    def train_stage1(self, train_loader, val_loader=None, test_loader=None, artifact_dir=None):
        print("\n" + "="*70)
        print("STAGE 1: Pre-training with GABA Mechanism")
        print("="*70)

        self.model.set_training_stage(1)

        # Only optimize Stage 1 relevant parameters
        # (embedding + backbone + domain_towers, excludes LoRA/routers/gates)
        stage1_params = self.model.get_stage1_params()
        optimizer = torch.optim.AdamW(
            stage1_params,
            lr=self.stage1_config['lr'],
            weight_decay=self.stage1_config['weight_decay']
        )

        scheduler = CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_epochs=self.stage1_config['warmup_epochs'],
            total_epochs=self.stage1_config['epochs'],
            lr_max=self.stage1_config['lr'],
            lr_min=self.stage1_config['lr_min']
        )

        criterion = LabelSmoothingBCELoss(smoothing=self.stage1_config['label_smoothing'])
        self._init_ema_gradients(train_loader, criterion=criterion)

        gaba_update_freq = self.stage1_config['gaba_update_freq']
        gaba_compute_freq = self.stage1_config['gaba_compute_freq']

        self.best_val_auc_stage1 = 0.0
        self.best_R_benefit = None
        best_model_weights = None
        patience_counter = 0

        for epoch in range(self.stage1_config['epochs']):
            self.model.train()
            total_loss = 0.0
            batch_count = 0
            current_lr = scheduler.get_lr()

            pbar = tqdm.tqdm(train_loader, desc=f"S1 Epoch {epoch+1}/{self.stage1_config['epochs']}")
            for batch_idx, (x_dict, y) in enumerate(pbar):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)

                if batch_idx % gaba_update_freq == 0:
                    domain_grads = self._collect_domain_gradients(x_dict, y, criterion=criterion)
                    for d, grad in domain_grads.items():
                        self.model.update_ema_gradients(d, grad)

                optimizer.zero_grad()
                logits = self.model(x_dict)

                # Paper Eq.(3): per-domain averaged loss
                domain_ids = x_dict["domain_indicator"].long()
                loss = _compute_multidomain_loss(criterion, logits, y, domain_ids)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    stage1_params,
                    self.stage1_config['gradient_clip']
                )
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                if batch_idx > 0 and batch_idx % gaba_compute_freq == 0:
                    R = self.model.compute_benefit_matrix_from_ema()
                    self.model.set_benefit_matrix(R)

                pbar.set_postfix(loss=f"{total_loss/batch_count:.4f}", lr=f"{current_lr:.2e}")

            scheduler.step()

            R_epoch = self.model.compute_benefit_matrix_from_ema()
            self.model.set_benefit_matrix(R_epoch)

            val_auc, val_loss = 0.5, 1.0
            if val_loader is not None:
                val_auc, val_loss = self.evaluate(val_loader)

                if val_auc > self.best_val_auc_stage1:
                    self.best_val_auc_stage1 = val_auc
                    self.best_R_benefit = R_epoch.clone()
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                    if self.verbose:
                        print(f"  New best Val AUC: {val_auc:.6f}")
                else:
                    patience_counter += 1

                if self.verbose:
                    print(f"  Val AUC: {val_auc:.6f}, Val Loss: {val_loss:.6f}")

            self.history['stage1']['train_loss'].append(total_loss / batch_count)
            self.history['stage1']['val_auc'].append(val_auc)
            self.history['stage1']['val_loss'].append(val_loss)
            self.history['stage1']['lr'].append(current_lr)

            if patience_counter >= self.stage1_config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)

        if self.best_R_benefit is not None:
            self.model.set_benefit_matrix(self.best_R_benefit)

        self._print_benefit_matrix_summary(self.best_R_benefit, title="Stage 1 Best R_benefit (GABA)")
        self._print_ema_gradient_diagnostics()

        metrics = {'stage': 'stage1'}
        if val_loader is not None:
            va, vl = self.evaluate(val_loader)
            metrics['val_auc'] = va
            metrics['val_logloss'] = vl
            metrics['val_domain_metrics'] = self.evaluate_by_domain(val_loader)

        if test_loader is not None:
            ta, tl = self.evaluate(test_loader)
            metrics['test_auc'] = ta
            metrics['test_logloss'] = tl
            metrics['test_domain_metrics'] = self.evaluate_by_domain(test_loader)

        if artifact_dir:
            os.makedirs(artifact_dir, exist_ok=True)
            with open(os.path.join(artifact_dir, 'stage1_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            torch.save(self.best_R_benefit, os.path.join(artifact_dir, 'R_benefit.pt'))

        return metrics

    def _init_ema_gradients(self, train_loader, criterion=None):
        self.model.eval()
        for x_dict, y in train_loader:
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
            y = y.to(self.device)
            domain_grads = self._collect_domain_gradients(x_dict, y, criterion=criterion)
            if domain_grads:
                for d, grad in domain_grads.items():
                    self.model.update_ema_gradients(d, grad)
                break
        self.model.train()

    def _print_benefit_matrix_summary(self, R, title="Benefit Matrix R"):
        if R is None:
            return
        print("\n" + "-"*60)
        print(f"{title}:")
        print("-"*60)
        R_np = R.detach().cpu().numpy()
        D = R_np.shape[0]

        header = "       " + "  ".join([f"D{j:>5d}" for j in range(D)])
        print(header)
        for i in range(D):
            row = f"  D{i:<3d}" + "  ".join([f"{R_np[i, j]:7.4f}" for j in range(D)])
            print(row)

        print(f"\n  Diagonal (self-reliance): {np.diag(R_np).round(4)}")
        print(f"  Row sums: {R_np.sum(axis=1).round(4)}")
        off_diag = R_np[~np.eye(D, dtype=bool)]
        if len(off_diag) > 0:
            print(f"  Off-diag  mean={off_diag.mean():.4f}  max={off_diag.max():.4f}  min={off_diag.min():.4f}")
        print("-"*60 + "\n")

    def _print_ema_gradient_diagnostics(self):
        if not self.model.ema_initialized or self.model.ema_gradients is None:
            print("[GABA] EMA gradients not initialized.")
            return

        ema = self.model.ema_gradients
        D = self.model.domain_num
        counts = self.model.gradient_count

        print("-"*60)
        print("EMA Gradient Diagnostics:")
        print("-"*60)
        print(f"  Gradient dimension: {ema.shape[1]}")
        for d in range(D):
            norm = torch.norm(ema[d]).item()
            print(f"  Domain {d}: ||g||={norm:.6f}, update_count={int(counts[d].item())}")

        print(f"\n  Pairwise cosine similarities:")
        header = "       " + "  ".join([f"D{j:>5d}" for j in range(D)])
        print(header)
        for i in range(D):
            row_vals = []
            for j in range(D):
                ni, nj = torch.norm(ema[i]), torch.norm(ema[j])
                if ni > 1e-10 and nj > 1e-10:
                    cos = torch.dot(ema[i], ema[j]) / (ni * nj)
                    row_vals.append(f"{cos.item():7.4f}")
                else:
                    row_vals.append(f"    N/A")
            print(f"  D{i:<3d}" + "  ".join(row_vals))
        print("-"*60 + "\n")

    def _print_and_save_gated_R(self, artifact_dir=None):
        R_gated = self.model.compute_gated_R()
        self._print_benefit_matrix_summary(
            R_gated, title="Stage 2 Effective Gated R  (R_benefit * softplus(gate))"
        )
        gate_np = self.model.relation_gate_logits.detach().cpu().numpy()
        D = gate_np.shape[0]
        print("-"*60)
        print("Relation Gate Logits (before softplus):")
        header = "       " + "  ".join([f"D{j:>5d}" for j in range(D)])
        print(header)
        for i in range(D):
            row = f"  D{i:<3d}" + "  ".join([f"{gate_np[i, j]:7.4f}" for j in range(D)])
            print(row)
        print("-"*60 + "\n")

        if artifact_dir:
            torch.save(R_gated, os.path.join(artifact_dir, 'R_gated_stage2.pt'))
            torch.save(self.model.relation_gate_logits.detach().cpu(),
                       os.path.join(artifact_dir, 'gate_logits_stage2.pt'))

        return R_gated

    def _write_matrix_to_file(self, f, R, title):
        R_np = R.detach().cpu().numpy()
        D = R_np.shape[0]
        f.write(f"\n{title}:\n")
        header = "       " + "  ".join([f"D{j:>5d}" for j in range(D)])
        f.write(header + "\n")
        for i in range(D):
            row = f"  D{i:<3d}" + "  ".join([f"{R_np[i, j]:7.4f}" for j in range(D)])
            f.write(row + "\n")

    def _build_domain_dataloaders(self, train_loader):
        domain_samples = {}

        for x_dict, y in train_loader:
            domain_ids = x_dict["domain_indicator"]
            for i in range(len(y)):
                d = int(domain_ids[i].item())
                if d not in domain_samples:
                    domain_samples[d] = {'x': {k: [] for k in x_dict.keys()}, 'y': []}
                for k in x_dict.keys():
                    domain_samples[d]['x'][k].append(x_dict[k][i])
                domain_samples[d]['y'].append(y[i])

        domain_loaders = {}
        if hasattr(train_loader, 'batch_size') and train_loader.batch_size is not None:
            batch_size = train_loader.batch_size
        elif hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'batch_size'):
            batch_size = train_loader.batch_sampler.batch_size
        else:
            batch_size = 2048

        for d, data in domain_samples.items():
            x_tensor = {k: torch.stack(v) for k, v in data['x'].items()}
            y_tensor = torch.stack(data['y'])
            dataset = torch.utils.data.TensorDataset(
                *[x_tensor[k] for k in sorted(x_tensor.keys())],
                y_tensor
            )
            sorted_keys = sorted(x_tensor.keys())
            domain_loaders[d] = {
                'loader': torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False),
                'keys': sorted_keys
            }

        return domain_loaders

    def _set_phase_w_trainable(self):
        is_epnet = (self.model.backbone_type == 'epnet')
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            elif 'embedding' in name and 'domain' not in name:
                param.requires_grad = False
            elif 'domain_towers' in name:
                param.requires_grad = is_epnet
            elif 'inter_layer_router' in name:
                param.requires_grad = True
            elif 'intra_layer_router' in name:
                param.requires_grad = False
            elif 'lora_' in name.lower() or 'lora_experts' in name:
                param.requires_grad = False
            elif 'relation_gate_logits' in name:
                param.requires_grad = True
            elif 'domain_embeddings' in name:
                param.requires_grad = True
            elif 'layer_positions' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _set_phase_h_trainable(self):
        is_epnet = (self.model.backbone_type == 'epnet')
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            elif 'embedding' in name and 'domain' not in name:
                param.requires_grad = False
            elif 'domain_towers' in name:
                param.requires_grad = is_epnet
            elif 'inter_layer_router' in name:
                param.requires_grad = False
            elif 'intra_layer_router' in name:
                param.requires_grad = True
            elif 'lora_' in name.lower() or 'lora_experts' in name:
                param.requires_grad = True
            elif 'relation_gate_logits' in name:
                param.requires_grad = False
            elif 'domain_embeddings' in name:
                param.requires_grad = False
            elif 'layer_positions' in name:
                param.requires_grad = False
            else:
                param.requires_grad = False

    def _create_phase_w_optimizer(self):
        param_groups = []

        # EPNet: towers are trainable in Stage 2
        tower_params = [p for n, p in self.model.named_parameters()
                        if 'domain_towers' in n and p.requires_grad]
        if tower_params:
            param_groups.append({'params': tower_params,
                                'lr': self.stage2_config['lr_lora']})

        inter_router_params = [p for n, p in self.model.named_parameters()
                               if 'inter_layer_router' in n and p.requires_grad]
        if inter_router_params:
            param_groups.append({'params': inter_router_params,
                                'lr': self.stage2_config['lr_inter_router']})

        gate_params = [p for n, p in self.model.named_parameters()
                       if 'relation_gate_logits' in n and p.requires_grad]
        if gate_params:
            param_groups.append({'params': gate_params,
                                'lr': self.stage2_config['lr_gate']})

        shared_params = [p for n, p in self.model.named_parameters()
                         if ('domain_embeddings' in n or 'layer_positions' in n) and p.requires_grad]
        if shared_params:
            param_groups.append({'params': shared_params,
                                'lr': self.stage2_config['lr_inter_router']})

        if param_groups:
            return torch.optim.AdamW(param_groups, weight_decay=self.stage2_config['weight_decay'])
        return None

    def _create_phase_h_optimizer(self):
        param_groups = []

        # EPNet: towers are trainable in Stage 2
        tower_params = [p for n, p in self.model.named_parameters()
                        if 'domain_towers' in n and p.requires_grad]
        if tower_params:
            param_groups.append({'params': tower_params,
                                'lr': self.stage2_config['lr_lora']})

        intra_router_params = [p for n, p in self.model.named_parameters()
                               if 'intra_layer_router' in n and p.requires_grad]
        if intra_router_params:
            param_groups.append({'params': intra_router_params,
                                'lr': self.stage2_config['lr_intra_router']})

        lora_params = [p for n, p in self.model.named_parameters()
                       if ('lora_' in n.lower() or 'lora_experts' in n) and p.requires_grad]
        if lora_params:
            param_groups.append({'params': lora_params,
                                'lr': self.stage2_config['lr_lora']})

        if param_groups:
            return torch.optim.AdamW(param_groups, weight_decay=self.stage2_config['weight_decay'])
        return None

    def train_stage2(self, train_loader, val_loader=None, test_loader=None, artifact_dir=None):
        print("\n" + "="*70)
        print("STAGE 2: Fine-tuning with Hierarchical Routing")
        print("  Alternating: Phase-W (odd epochs) / Phase-H (even epochs)")
        print("  Phase-H: Domain-independent training")
        print("="*70)

        self.model.set_training_stage(2)

        if self.best_R_benefit is not None:
            self.model.set_benefit_matrix(self.best_R_benefit)

        # EPNet: towers were unused in Stage 1, initialize from backbone output_layer
        if self.model.backbone_type == 'epnet':
            self.model.init_domain_towers_from_backbone()
            print("  EPNet: initialized domain towers from backbone output_layer")

        domain_loaders = self._build_domain_dataloaders(train_loader)
        print(f"  Built domain-specific loaders for {len(domain_loaders)} domains")

        criterion = LabelSmoothingBCELoss(smoothing=self.stage2_config['label_smoothing'])

        lam_d = self.stage2_config['lambda_dir']

        best_val_auc = 0.0
        best_model_weights = None
        patience_counter = 0

        self._set_phase_w_trainable()
        opt_phase_w = self._create_phase_w_optimizer()

        self._set_phase_h_trainable()
        opt_phase_h = self._create_phase_h_optimizer()

        for epoch in range(self.stage2_config['epochs']):
            is_phase_w = (epoch % 2 == 0)
            phase_name = "W" if is_phase_w else "H"

            self.model.train()
            total_task_loss = 0.0
            batch_count = 0

            if is_phase_w:
                self._set_phase_w_trainable()
                optimizer = opt_phase_w

                total_l_dir = 0.0

                pbar = tqdm.tqdm(train_loader, desc=f"S2 Epoch {epoch+1} [Phase-W]")
                for x_dict, y in pbar:
                    x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                    y = y.to(self.device)

                    optimizer.zero_grad()
                    logits = self.model(x_dict)

                    domain_ids = x_dict["domain_indicator"].long()
                    task_loss = _compute_multidomain_loss(criterion, logits, y, domain_ids)

                    domain_emb = self.model.domain_embeddings(domain_ids)
                    zeta, alpha, phi = self.model.hierarchical_routing(domain_emb, domain_ids, return_phi=True)

                    l_dir = self.model.compute_directional_loss(phi, domain_ids)

                    total_loss = task_loss + lam_d * l_dir
                    total_loss.backward()

                    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.stage2_config['gradient_clip'])
                    optimizer.step()

                    total_task_loss += task_loss.item()
                    total_l_dir += l_dir.item()
                    batch_count += 1

                    pbar.set_postfix(
                        loss=f"{total_task_loss/batch_count:.4f}",
                        d=f"{total_l_dir/batch_count:.3f}"
                    )

                self.history['stage2']['l_dir'].append(total_l_dir / batch_count)

            else:
                self._set_phase_h_trainable()
                optimizer = opt_phase_h

                for d, loader_info in domain_loaders.items():
                    loader = loader_info['loader']
                    keys = loader_info['keys']

                    pbar = tqdm.tqdm(loader, desc=f"S2 Epoch {epoch+1} [Phase-H] Domain {d}")
                    for batch in pbar:
                        x_tensors = batch[:-1]
                        y = batch[-1].to(self.device)

                        x_dict = {}
                        for i, k in enumerate(keys):
                            x_dict[k] = x_tensors[i].to(self.device)

                        optimizer.zero_grad()
                        logits = self.model(x_dict)
                        task_loss = criterion(logits, y.float())
                        task_loss.backward()

                        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                        torch.nn.utils.clip_grad_norm_(trainable_params, self.stage2_config['gradient_clip'])
                        optimizer.step()

                        total_task_loss += task_loss.item()
                        batch_count += 1

                        pbar.set_postfix(loss=f"{total_task_loss/batch_count:.4f}")

            avg_loss = total_task_loss / max(batch_count, 1)
            self.history['stage2']['train_loss'].append(avg_loss)

            val_auc, val_loss = 0.5, 1.0
            if val_loader is not None:
                val_auc, val_loss = self.evaluate(val_loader)
                self.history['stage2']['val_auc'].append(val_auc)
                self.history['stage2']['val_loss'].append(val_loss)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                    if self.verbose:
                        print(f"  New best Val AUC: {val_auc:.6f}")
                else:
                    patience_counter += 1

                if self.verbose:
                    print(f"  [Phase-{phase_name}] Val AUC: {val_auc:.6f}, Val Loss: {val_loss:.6f}")

            if patience_counter >= self.stage2_config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)

        self._print_benefit_matrix_summary(
            self.model.R_benefit, title="Stage 2 R_benefit (from Stage 1, frozen)"
        )
        self._print_and_save_gated_R(artifact_dir=artifact_dir)

        metrics = {'stage': 'stage2'}
        if val_loader is not None:
            va, vl = self.evaluate(val_loader)
            metrics['val_auc'] = va
            metrics['val_logloss'] = vl
            metrics['val_domain_metrics'] = self.evaluate_by_domain(val_loader)

        if test_loader is not None:
            ta, tl = self.evaluate(test_loader)
            metrics['test_auc'] = ta
            metrics['test_logloss'] = tl
            metrics['test_domain_metrics'] = self.evaluate_by_domain(test_loader)

        if artifact_dir:
            os.makedirs(artifact_dir, exist_ok=True)
            with open(os.path.join(artifact_dir, 'stage2_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

        return metrics

    def save_all_results(self, stage1_metrics, stage2_metrics, filepath, backbone_name='', config_info=None):
        with open(filepath, 'a') as f:
            f.write("="*80 + "\n")
            f.write(f"ALoHA-Rec Experiment Results\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Backbone: {backbone_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            if config_info:
                f.write("-"*40 + "\n")
                f.write("CONFIGURATION\n")
                f.write("-"*40 + "\n")
                for k, v in config_info.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

            f.write("-"*40 + "\n")
            f.write("STAGE 1 (BASE + GABA) RESULTS\n")
            f.write("-"*40 + "\n")
            if 'val_auc' in stage1_metrics:
                f.write(f"Validation: AUC={stage1_metrics['val_auc']:.6f}, LogLoss={stage1_metrics['val_logloss']:.6f}\n")
            if 'test_auc' in stage1_metrics:
                f.write(f"Test: AUC={stage1_metrics['test_auc']:.6f}, LogLoss={stage1_metrics['test_logloss']:.6f}\n")
            if 'test_domain_metrics' in stage1_metrics:
                f.write("\nPer-Domain Test (Stage 1):\n")
                for d, m in sorted(stage1_metrics['test_domain_metrics'].items()):
                    f.write(f"  Domain {d}: AUC={m['auc']:.6f}, N={m['samples']}\n")

            f.write("\n" + "-"*40 + "\n")
            f.write("STAGE 2 (ALoHA-Rec) RESULTS\n")
            f.write("-"*40 + "\n")
            if 'val_auc' in stage2_metrics:
                f.write(f"Validation: AUC={stage2_metrics['val_auc']:.6f}, LogLoss={stage2_metrics['val_logloss']:.6f}\n")
            if 'test_auc' in stage2_metrics:
                f.write(f"Test: AUC={stage2_metrics['test_auc']:.6f}, LogLoss={stage2_metrics['test_logloss']:.6f}\n")
            if 'test_domain_metrics' in stage2_metrics:
                f.write("\nPer-Domain Test (Stage 2):\n")
                for d, m in sorted(stage2_metrics['test_domain_metrics'].items()):
                    f.write(f"  Domain {d}: AUC={m['auc']:.6f}, N={m['samples']}\n")

            f.write("\n" + "-"*40 + "\n")
            f.write("IMPROVEMENT\n")
            f.write("-"*40 + "\n")
            if 'test_auc' in stage1_metrics and 'test_auc' in stage2_metrics:
                delta = stage2_metrics['test_auc'] - stage1_metrics['test_auc']
                f.write(f"Delta AUC: {delta:+.6f} ({delta/stage1_metrics['test_auc']*100:+.2f}%)\n")

            f.write("\n" + "-"*40 + "\n")
            f.write("BENEFIT MATRICES\n")
            f.write("-"*40 + "\n")
            self._write_matrix_to_file(f, self.model.R_benefit, "R_benefit (Stage 1 GABA)")
            R_gated = self.model.compute_gated_R()
            self._write_matrix_to_file(f, R_gated, "R_gated (Stage 2 effective)")

            f.write("\n" + "="*80 + "\n\n")

        print(f"Results saved to: {filepath}")

    def run_full_training(self, train_loader, val_loader, test_loader,
                          result_filepath, backbone_name='', artifact_dir=None):
        stage1_metrics = self.train_stage1(train_loader, val_loader, test_loader, artifact_dir=artifact_dir)
        stage2_metrics = self.train_stage2(train_loader, val_loader, test_loader, artifact_dir=artifact_dir)

        config_info = {
            'lora_rank': self.model.lora_rank,
            'num_experts': self.model.num_experts,
            'k_layers': self.model.k_layers,
            'k_experts': self.model.k_experts,
            'ema_decay': self.model.ema_decay,
            'stage1_lr': self.stage1_config['lr'],
            'stage1_label_smoothing': self.stage1_config['label_smoothing'],
            'stage2_lr_lora': self.stage2_config['lr_lora'],
            'stage2_lambda_dir': self.stage2_config['lambda_dir'],
        }

        self.save_all_results(stage1_metrics, stage2_metrics, result_filepath,
                             backbone_name=backbone_name, config_info=config_info)

        return {
            'stage1': stage1_metrics,
            'stage2': stage2_metrics,
            'improvement': {
                'test_auc': stage2_metrics.get('test_auc', 0) - stage1_metrics.get('test_auc', 0),
                'test_logloss': stage1_metrics.get('test_logloss', 1) - stage2_metrics.get('test_logloss', 1)
            }
        }