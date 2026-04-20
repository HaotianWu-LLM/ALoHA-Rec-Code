import copy
import time
import os

import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import roc_auc_score, log_loss

from ..basic.callback import EarlyStopper


class ALoHATrainer(object):
    def __init__(
        self,
        model,
        data_set_type,
        device="cpu",
        model_path="./",
        stage1_lr=1e-3,
        stage1_weight_decay=1e-5,
        stage1_n_epoch=100,
        stage1_earlystop_patience=5,
        stage1_scheduler_step=4,
        stage1_scheduler_gamma=0.95,
        gaba_update_freq=10,
        stage2_n_epoch=20,
        stage2_earlystop_patience=5,
        lambda_dir=0.1,
    ):
        self.model = model
        self.data_set_type = data_set_type
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model_path = model_path

        self.stage1_lr = stage1_lr
        self.stage1_weight_decay = stage1_weight_decay
        self.stage1_n_epoch = stage1_n_epoch
        self.stage1_earlystop_patience = stage1_earlystop_patience
        self.stage1_scheduler_step = stage1_scheduler_step
        self.stage1_scheduler_gamma = stage1_scheduler_gamma
        self.gaba_update_freq = gaba_update_freq

        self.stage2_n_epoch = stage2_n_epoch
        self.stage2_earlystop_patience = stage2_earlystop_patience
        self.lambda_dir = lambda_dir

        self.evaluate_fn = roc_auc_score

        self.stage1_R = None
        self.stage2_R = None

    def _print_and_save_R(self, R, stage_tag):
        """Pretty-print R and save as .pt with dataset + stage + timestamp."""
        R_cpu = R.detach().cpu()
        M = R_cpu.shape[0]
        print(f"\n[{stage_tag}] benefit matrix R ({M}x{M}):")
        for i in range(M):
            row = "  " + "  ".join(f"{R_cpu[i, j].item():+.4f}" for j in range(M))
            print(row)

        os.makedirs(self.model_path, exist_ok=True)
        ts = time.strftime('%m_%d_%H_%M', time.localtime())
        fname = f"R_{self.data_set_type}_{stage_tag}_{ts}.pt"
        fpath = os.path.join(self.model_path, fname)
        torch.save(R_cpu, fpath)
        print(f"[{stage_tag}] saved R matrix to: {fpath}")

    def _collect_domain_gradients(self, x_dict, y, criterion):
        was_training = self.model.training
        self.model.eval()
        target_params = list(self.model.framework.backbone_layers[-1].parameters())
        domain_id = x_dict["domain_indicator"].long()
        if domain_id.dim() > 1:
            domain_id = domain_id.squeeze(-1)

        y_pred = self.model(x_dict)
        domain_grads = {}
        unique_domains = torch.unique(domain_id).tolist()
        for i, d in enumerate(unique_domains):
            mask = (domain_id == d)
            if mask.sum() == 0:
                continue
            y_pred_d = y_pred[mask].clamp(1e-7, 1 - 1e-7)
            y_d = y[mask].float()
            loss_d = criterion(y_pred_d, y_d)
            is_last = (i == len(unique_domains) - 1)
            grads = torch.autograd.grad(
                loss_d, target_params,
                retain_graph=not is_last, create_graph=False,
                allow_unused=True
            )
            grad_vec = torch.cat([
                (g.detach().reshape(-1) if g is not None else torch.zeros(p.numel(), device=self.device))
                for g, p in zip(grads, target_params)
            ])
            domain_grads[int(d)] = grad_vec

        self.model.train(was_training)
        return domain_grads

    def train_stage1(self, train_loader, val_loader=None, test_loader=None):
        self.model.set_training_stage(1)
        self.model.to(self.device)

        optimizer = torch.optim.Adam(
            self.model.get_stage1_params(),
            lr=self.stage1_lr,
            weight_decay=self.stage1_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.stage1_scheduler_step, gamma=self.stage1_scheduler_gamma
        )
        criterion = nn.BCELoss()
        early_stopper = EarlyStopper(patience=self.stage1_earlystop_patience)
        early_stopped = False

        for epoch_i in range(self.stage1_n_epoch):
            print('stage1 epoch:', epoch_i)
            self.model.train()
            total_loss = 0.0
            tk0 = tqdm.tqdm(train_loader, desc="train_s1", smoothing=0, mininterval=1.0)
            log_interval = 10
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)

                if self.gaba_update_freq > 0 and i % self.gaba_update_freq == 0:
                    try:
                        domain_grads = self._collect_domain_gradients(x_dict, y, criterion)
                        for d, grad in domain_grads.items():
                            self.model.update_ema_gradients(d, grad)
                    except Exception:
                        pass

                y_pred = self.model(x_dict)
                loss = criterion(y_pred, y.float())
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if (i + 1) % log_interval == 0:
                    tk0.set_postfix(loss=total_loss / log_interval)
                    total_loss = 0

            if epoch_i % scheduler.step_size == 0:
                print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            scheduler.step()

            if val_loader is not None:
                auc, logloss_val = self.evaluate(self.model, val_loader)
                print(f'stage1 epoch:{epoch_i} | val auc: {auc} | val logloss: {logloss_val}')
                if early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f'stage1 validation: best auc: {early_stopper.best_auc}')
                    self.model.load_state_dict(early_stopper.best_weights)
                    early_stopped = True
                    break

        if (not early_stopped) and val_loader is not None and early_stopper.best_weights is not None:
            print(f'stage1 finished without early-stop. Loading best weights '
                  f'(best val auc: {early_stopper.best_auc}) before computing R.')
            self.model.load_state_dict(early_stopper.best_weights)

        try:
            R = self.model.compute_benefit_matrix_from_ema()
            self.model.set_benefit_matrix(R)
            self.stage1_R = R.detach().cpu().clone()
            self._print_and_save_R(R, stage_tag='stage1')
        except Exception as e:
            print(f'[stage1] Failed to compute/save R: {e}')

        metrics = {}
        if val_loader is not None:
            auc, logloss_val = self.evaluate(self.model, val_loader)
            metrics['val_auc'] = auc
            metrics['val_logloss'] = logloss_val
        if test_loader is not None:
            dom_ll, dom_auc, tot_ll, tot_auc = self.evaluate_multi_domain(
                self.model, test_loader, self.model.domain_num
            )
            metrics['test_auc'] = tot_auc
            metrics['test_logloss'] = tot_ll
            metrics['domain_auc'] = dom_auc
            metrics['domain_logloss'] = dom_ll
        return metrics

    def _set_phase_w_trainable(self):
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.inter_layer_router.parameters():
            p.requires_grad = True
        for p in self.model.intra_layer_router.parameters():
            p.requires_grad = True
        self.model.domain_embeddings.weight.requires_grad = True
        self.model.layer_positions.requires_grad = True
        self.model.relation_gate_logits.requires_grad = True

    def _set_phase_h_trainable(self):
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.lora_experts.parameters():
            p.requires_grad = True

    def _create_phase_w_optimizer(self, lr=1e-3, weight_decay=1e-5):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def _create_phase_h_optimizer(self, lr=1e-3, weight_decay=1e-5):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def _build_domain_dataloaders(self, train_loader, batch_size):
        dataset = train_loader.dataset
        domain_indices = {d: [] for d in range(self.model.domain_num)}
        for idx in range(len(dataset)):
            item = dataset[idx]
            if isinstance(item, tuple) and isinstance(item[0], dict):
                did = int(item[0]["domain_indicator"])
                domain_indices[did].append(idx)
        loaders = {}
        for d, idxs in domain_indices.items():
            if len(idxs) == 0:
                continue
            sub = torch.utils.data.Subset(dataset, idxs)
            loaders[d] = torch.utils.data.DataLoader(
                sub, batch_size=batch_size, shuffle=True, drop_last=False
            )
        return loaders

    def train_stage2(self, train_loader, val_loader=None, test_loader=None,
                     batch_size=4096, phase_w_lr=1e-3, phase_h_lr=1e-3,
                     phase_w_weight_decay=1e-5, phase_h_weight_decay=1e-5):
        self.model.set_training_stage(2)
        self.model.to(self.device)

        self._set_phase_w_trainable()
        opt_w = self._create_phase_w_optimizer(phase_w_lr, phase_w_weight_decay)
        self._set_phase_h_trainable()
        opt_h = self._create_phase_h_optimizer(phase_h_lr, phase_h_weight_decay)

        criterion = nn.BCELoss()
        early_stopper = EarlyStopper(patience=self.stage2_earlystop_patience)
        early_stopped = False

        domain_loaders = None
        try:
            domain_loaders = self._build_domain_dataloaders(train_loader, batch_size)
        except Exception:
            domain_loaders = None

        for epoch_i in range(self.stage2_n_epoch):
            is_phase_w = (epoch_i % 2 == 0)
            print(f'stage2 epoch: {epoch_i} | phase: {"W" if is_phase_w else "H"}')

            self.model.train()
            if is_phase_w:
                self._set_phase_w_trainable()
                optimizer = opt_w
                loader = train_loader
                tk0 = tqdm.tqdm(loader, desc="train_s2_W", smoothing=0, mininterval=1.0)
                for i, (x_dict, y) in enumerate(tk0):
                    x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                    y = y.to(self.device)
                    domain_id = x_dict["domain_indicator"].long()
                    if domain_id.dim() > 1:
                        domain_id = domain_id.squeeze(-1)
                    y_pred = self.model(x_dict)
                    loss_task = criterion(y_pred, y.float())
                    loss_dir = self.model.compute_directional_loss(
                        self.model._last_phi, domain_id
                    )
                    loss = loss_task + self.lambda_dir * loss_dir
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                self._set_phase_h_trainable()
                optimizer = opt_h
                if domain_loaders is not None and len(domain_loaders) > 0:
                    for d, dloader in domain_loaders.items():
                        tk0 = tqdm.tqdm(dloader, desc=f"train_s2_H_d{d}", smoothing=0, mininterval=1.0)
                        for i, (x_dict, y) in enumerate(tk0):
                            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                            y = y.to(self.device)
                            y_pred = self.model(x_dict)
                            loss = criterion(y_pred, y.float())
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                else:
                    tk0 = tqdm.tqdm(train_loader, desc="train_s2_H", smoothing=0, mininterval=1.0)
                    for i, (x_dict, y) in enumerate(tk0):
                        x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                        y = y.to(self.device)
                        y_pred = self.model(x_dict)
                        loss = criterion(y_pred, y.float())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

            if val_loader is not None:
                auc, logloss_val = self.evaluate(self.model, val_loader)
                print(f'stage2 epoch:{epoch_i} | val auc: {auc} | val logloss: {logloss_val}')
                if early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f'stage2 validation: best auc: {early_stopper.best_auc}')
                    self.model.load_state_dict(early_stopper.best_weights)
                    early_stopped = True
                    break

        if (not early_stopped) and val_loader is not None and early_stopper.best_weights is not None:
            print(f'stage2 finished without early-stop. Loading best weights '
                  f'(best val auc: {early_stopper.best_auc}) before computing R.')
            self.model.load_state_dict(early_stopper.best_weights)

        for p in self.model.parameters():
            p.requires_grad = True

        try:
            R_gated = self.model.compute_gated_R()
            self.stage2_R = R_gated.detach().cpu().clone()
            self._print_and_save_R(R_gated, stage_tag='stage2')
        except Exception as e:
            print(f'[stage2] Failed to compute/save R: {e}')

        metrics = {}
        if val_loader is not None:
            auc, logloss_val = self.evaluate(self.model, val_loader)
            metrics['val_auc'] = auc
            metrics['val_logloss'] = logloss_val
        if test_loader is not None:
            dom_ll, dom_auc, tot_ll, tot_auc = self.evaluate_multi_domain(
                self.model, test_loader, self.model.domain_num
            )
            metrics['test_auc'] = tot_auc
            metrics['test_logloss'] = tot_ll
            metrics['domain_auc'] = dom_auc
            metrics['domain_logloss'] = dom_ll
        return metrics

    def run_full_training(self, train_loader, val_loader=None, test_loader=None,
                          batch_size=4096):
        stage1_metrics = self.train_stage1(train_loader, val_loader, test_loader)
        stage2_metrics = self.train_stage2(train_loader, val_loader, test_loader, batch_size=batch_size)
        time_now = int(round(time.time() * 1000))
        time_now = time.strftime('%m_%d_%H_%M', time.localtime(time_now / 1000))
        name = f"ADLS_{self.data_set_type}_{time_now}.pth"
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.model_path, name))
        return stage1_metrics, stage2_metrics

    def evaluate(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return self.evaluate_fn(targets, predicts), log_loss(targets, predicts)

    def evaluate_multi_domain(self, model, data_loader, domain_num):
        model.eval()
        targets_all, predicts_all = list(), list()
        tgt_dom = [list() for _ in range(domain_num)]
        prd_dom = [list() for _ in range(domain_num)]
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()
                y = y.to(self.device)
                y_pred = model(x_dict)
                targets_all.extend(y.tolist())
                predicts_all.extend(y_pred.tolist())
                for d in range(domain_num):
                    mask = (domain_id == d)
                    tgt_dom[d].extend(y[mask].tolist())
                    prd_dom[d].extend(y_pred[mask].tolist())
        dom_ll = []
        dom_auc = []
        for d in range(domain_num):
            if tgt_dom[d] and len(set(tgt_dom[d])) > 1:
                dom_ll.append(log_loss(tgt_dom[d], prd_dom[d]))
                dom_auc.append(self.evaluate_fn(tgt_dom[d], prd_dom[d]))
            else:
                dom_ll.append(None)
                dom_auc.append(None)
        tot_ll = log_loss(targets_all, predicts_all) if predicts_all else None
        tot_auc = self.evaluate_fn(targets_all, predicts_all) if predicts_all else None
        return dom_ll, dom_auc, tot_ll, tot_auc