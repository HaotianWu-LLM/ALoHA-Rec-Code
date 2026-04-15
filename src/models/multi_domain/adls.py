"""
ALoHA-Rec: Asymmetric-aware Low-rank Hierarchical Adaptation for Multi-Domain Recommendation

Key design decisions:
  - Routers: single-layer linear (paper Eq.9 / Eq.15)
  - GABA: parameter-space gradients from last backbone layer (paper Eq.4)
  - Stage 1 forward:
      EPNet  → backbone's shared output_layer (matches standalone EPNet exactly)
      Others → domain-specific towers
  - Stage 2: all backbones use domain towers + LoRA routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...basic.layers import EmbeddingLayer, MLP


class LoRAModule(nn.Module):

    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return lora_out * self.scaling


class ADLS(nn.Module):

    def __init__(self, features, domain_num, backbone_type='fcn',
                 bottom_params=None, tower_params=None,
                 num_experts=None, lora_rank=4, lora_alpha=1.0,
                 num_lora_layers=3, k_layers=2, k_experts=2,
                 train_lora_in_stage1=True, stage1_lora_mode='uniform',
                 ema_decay=0.9, beta=0.9,
                 **backbone_kwargs):
        super().__init__()

        self.features = features
        self.domain_num = domain_num
        self.backbone_type = backbone_type.lower()

        if num_experts is None:
            num_experts = domain_num
        self.num_experts = num_experts

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_lora_layers = num_lora_layers
        self.k_layers = k_layers
        self.k_experts = k_experts
        self.train_lora_in_stage1 = train_lora_in_stage1
        self.stage1_lora_mode = stage1_lora_mode
        self.ema_decay = ema_decay
        self.beta = beta

        self.embedding = EmbeddingLayer(features)
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.num_fields = len(features)
        self.embed_dim = features[0].embed_dim if features else 16

        self._init_backbone(bottom_params, tower_params, backbone_kwargs)

        # --- Domain-specific towers ---
        if tower_params is None:
            tower_params = {"dims": [8], "dropout": 0.2, "activation": "relu"}
        tower_dims = tower_params.get("dims", [8])
        tower_dropout = tower_params.get("dropout", 0.2)
        self.domain_towers = nn.ModuleList([
            MLP(self.backbone.feature_dim, output_layer=True,
                dims=tower_dims, dropout=tower_dropout)
            for _ in range(domain_num)
        ])

        lora_configs = self.backbone.get_lora_configs()
        self.num_lora_layers = min(num_lora_layers, len(lora_configs))
        self.lora_configs = lora_configs[:self.num_lora_layers]

        self.domain_hidden_dim = 64
        self.domain_embeddings = nn.Embedding(domain_num, self.domain_hidden_dim)
        nn.init.normal_(self.domain_embeddings.weight, mean=0.0, std=0.01)

        self.layer_position_dim = 32
        self.layer_positions = nn.Parameter(torch.empty(self.num_lora_layers, self.layer_position_dim))
        nn.init.normal_(self.layer_positions, mean=0.0, std=0.01)

        router_input_dim = self.domain_hidden_dim + self.layer_position_dim

        # --- Single-layer routers (paper Eq.9 and Eq.15) ---
        # Inter-layer: s_{l,m} = w_L^T [p_l; h'_m] + b_L
        self.inter_layer_router = nn.Linear(router_input_dim, 1)
        nn.init.xavier_uniform_(self.inter_layer_router.weight)
        nn.init.zeros_(self.inter_layer_router.bias)

        # Intra-layer: r^(e)_{l,m} = (w^(e)_E)^T [p_l; h'_m] + b^(e)_E
        # A single Linear(dim, num_experts) is equivalent to per-expert projections
        self.intra_layer_router = nn.Linear(router_input_dim, num_experts)
        nn.init.xavier_uniform_(self.intra_layer_router.weight)
        nn.init.zeros_(self.intra_layer_router.bias)

        self.lora_experts = nn.ModuleList()
        for cfg in self.lora_configs:
            layer_experts = nn.ModuleList([
                LoRAModule(cfg['in_dim'], cfg['out_dim'], rank=lora_rank, alpha=lora_alpha)
                for _ in range(num_experts)
            ])
            self.lora_experts.append(layer_experts)

        self.register_buffer('R_benefit', torch.eye(domain_num))
        self.register_buffer('ema_gradients', None)
        self.ema_initialized = False
        self.register_buffer('gradient_count', torch.zeros(domain_num))
        self.register_buffer('rho_self', torch.zeros(domain_num))

        self.relation_gate_logits = nn.Parameter(torch.zeros(domain_num, domain_num))

        self.training_stage = 1

    def _init_backbone(self, bottom_params, tower_params, backbone_kwargs):
        from .adls_backbones import (
            FCNBackbone, DeepFMBackbone, DCNBackbone,
            xDeepFMBackbone, AutoIntBackbone, EPNetBackbone
        )

        if bottom_params is None:
            bottom_params = {"dims": [256, 128, 64], "dropout": 0.2}

        deep_dims = bottom_params.get("dims", [256, 128, 64])
        dropout = bottom_params.get("dropout", 0.2)

        common_kwargs = {
            'dropout': dropout,
            'num_fields': self.num_fields,
            'embed_dim': self.embed_dim
        }

        if self.backbone_type in ['fcn', 'mlp']:
            self.backbone = FCNBackbone(
                input_dim=self.input_dim,
                fcn_dims=deep_dims,
                dropout=dropout
            )
        elif self.backbone_type == 'deepfm':
            self.backbone = DeepFMBackbone(
                input_dim=self.input_dim,
                deep_dims=deep_dims,
                **common_kwargs
            )
        elif self.backbone_type == 'dcn':
            self.backbone = DCNBackbone(
                input_dim=self.input_dim,
                num_cross_layers=backbone_kwargs.get('num_cross_layers', 3),
                deep_dims=deep_dims,
                dropout=dropout
            )
        elif self.backbone_type == 'xdeepfm':
            self.backbone = xDeepFMBackbone(
                input_dim=self.input_dim,
                cin_layer_sizes=backbone_kwargs.get('cin_layer_sizes', [128, 128]),
                deep_dims=deep_dims,
                **common_kwargs
            )
        elif self.backbone_type == 'autoint':
            self.backbone = AutoIntBackbone(
                input_dim=self.input_dim,
                deep_dims=deep_dims,
                num_heads=backbone_kwargs.get('num_heads', 2),
                num_attention_layers=backbone_kwargs.get('num_attention_layers', 2),
                **common_kwargs
            )
        elif self.backbone_type == 'epnet':
            self.backbone = EPNetBackbone(
                input_dim=self.input_dim,
                domain_num=self.domain_num,
                fcn_dims=deep_dims,
                dropout=dropout,
                domain_embed_dim=backbone_kwargs.get('domain_embed_dim', 16),
                gate_gamma=backbone_kwargs.get('gate_gamma', 2.0)
            )
        else:
            raise ValueError(f"Unknown backbone type: {self.backbone_type}")

    def set_training_stage(self, stage):
        self.training_stage = stage
        if stage == 2:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.embedding.parameters():
                param.requires_grad = False
            if self.backbone_type != 'epnet':
                # Non-EPNet: towers were trained in Stage 1, freeze them
                for param in self.domain_towers.parameters():
                    param.requires_grad = False
            # EPNet: towers were unused in Stage 1, keep them trainable

    def init_domain_towers_from_backbone(self):
        """Copy backbone output_layer weights to domain towers (EPNet only).
        For EPNet tower_dims=[], both are Linear(feature_dim, 1) so shapes match."""
        src = self.backbone.output_layer
        for tower in self.domain_towers:
            last_linear = None
            if hasattr(tower, 'mlp'):
                for m in tower.mlp.modules():
                    if isinstance(m, nn.Linear):
                        last_linear = m
            if last_linear is not None and last_linear.weight.shape == src.weight.shape:
                last_linear.weight.data.copy_(src.weight.data)
                if last_linear.bias is not None and src.bias is not None:
                    last_linear.bias.data.copy_(src.bias.data)

    def get_stage1_params(self):
        """Return parameters for Stage 1 optimizer."""
        stage1_params = []
        stage1_params.extend(self.embedding.parameters())
        stage1_params.extend(self.backbone.parameters())
        if self.backbone_type != 'epnet':
            # Non-EPNet: towers participate in Stage 1
            stage1_params.extend(self.domain_towers.parameters())
        # EPNet: towers excluded (unused in Stage 1)
        return stage1_params

    def _safe_normalize(self, x, dim=-1, eps=1e-8):
        x_sum = x.sum(dim=dim, keepdim=True)
        x_sum = torch.clamp(x_sum, min=eps)
        return x / x_sum

    def compute_gated_R(self):
        gate = F.softplus(self.relation_gate_logits)
        R_gated = self.R_benefit.detach() * gate
        R_gated = self._safe_normalize(R_gated, dim=1)
        return R_gated

    def forward(self, x):
        domain_id = x["domain_indicator"].long()
        if domain_id.dim() == 0:
            domain_id = domain_id.unsqueeze(0)

        input_emb = self.embedding(x, self.features, squeeze_dim=True)
        emb_3d = self.embedding(x, self.features, squeeze_dim=False)

        if input_emb.dim() == 1:
            input_emb = input_emb.unsqueeze(0)
        if emb_3d.dim() == 2:
            emb_3d = emb_3d.unsqueeze(0)

        if torch.isnan(input_emb).any():
            input_emb = torch.nan_to_num(input_emb, nan=0.0)

        if self.training_stage == 1:
            lora_args = dict(emb_3d=emb_3d, domain_id=domain_id)
            if self.train_lora_in_stage1 and self.stage1_lora_mode == 'uniform':
                lora_args.update(mode='uniform', num_experts=self.num_experts)
                lora_module = self.lora_experts
            else:
                lora_args.update(mode='none')
                lora_module = None

            if self.backbone_type == 'epnet':
                # EPNet: use backbone's own shared output_layer to match standalone
                logits = self.backbone.forward_with_lora_experts(
                    input_emb, lora_module,
                    return_features=False, **lora_args
                )
            else:
                # Other backbones: use domain towers in Stage 1
                features, aux_logit = self.backbone.forward_with_lora_experts(
                    input_emb, lora_module,
                    return_features=True, **lora_args
                )
                B = features.shape[0]
                logits = torch.zeros(B, device=features.device)
                for d in range(self.domain_num):
                    mask = (domain_id == d)
                    if mask.any():
                        tower_out = self.domain_towers[d](features[mask])
                        logits[mask] = tower_out.squeeze(-1)
                logits = logits + aux_logit
        else:
            # Stage 2: all backbones use domain towers + LoRA routing
            domain_emb = self.domain_embeddings(domain_id)
            zeta, alpha = self.hierarchical_routing(domain_emb, domain_id)
            features, aux_logit = self.backbone.forward_with_lora_experts(
                input_emb, self.lora_experts,
                mode='routed', zeta=zeta, alpha=alpha,
                emb_3d=emb_3d, return_features=True, domain_id=domain_id
            )

            B = features.shape[0]
            logits = torch.zeros(B, device=features.device)
            for d in range(self.domain_num):
                mask = (domain_id == d)
                if mask.any():
                    tower_out = self.domain_towers[d](features[mask])
                    logits[mask] = tower_out.squeeze(-1)

            logits = logits + aux_logit

        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0)

        return logits

    def hierarchical_routing(self, domain_emb, domain_ids, return_phi=False):
        if domain_ids.dim() == 0:
            domain_ids = domain_ids.unsqueeze(0)
        if domain_emb.dim() == 1:
            domain_emb = domain_emb.unsqueeze(0)

        B = domain_emb.shape[0]
        device = domain_emb.device

        R_gated = self.compute_gated_R()
        R_expanded = R_gated[domain_ids]
        all_domain_embs = self.domain_embeddings.weight

        h_prime = R_expanded @ all_domain_embs

        layer_scores = []
        for l in range(self.num_lora_layers):
            layer_pos = self.layer_positions[l:l+1].expand(B, -1)
            router_input = torch.cat([h_prime, layer_pos], dim=-1)
            score = self.inter_layer_router(router_input).squeeze(-1)
            layer_scores.append(score)

        layer_scores = torch.stack(layer_scores, dim=1)
        layer_scores = layer_scores - layer_scores.max(dim=-1, keepdim=True)[0]
        phi = F.softmax(layer_scores, dim=-1)

        k = min(self.k_layers, self.num_lora_layers)
        topk_vals, topk_idx = torch.topk(phi, k, dim=-1)

        if self.training:
            mask = torch.zeros_like(phi)
            mask.scatter_(1, topk_idx, 1.0)
            zeta = mask * phi
        else:
            zeta = torch.zeros_like(phi)
            zeta.scatter_(1, topk_idx, topk_vals)

        zeta = self._safe_normalize(zeta, dim=-1)

        alpha_all = []
        for l in range(self.num_lora_layers):
            layer_pos = self.layer_positions[l:l+1].expand(B, -1)
            router_input = torch.cat([h_prime, layer_pos], dim=-1)
            expert_scores = self.intra_layer_router(router_input)

            expert_scores = expert_scores - expert_scores.max(dim=-1, keepdim=True)[0]
            alpha = F.softmax(expert_scores, dim=-1)

            if self.k_experts < self.num_experts:
                topk_vals, topk_idx = torch.topk(alpha, self.k_experts, dim=-1)
                if self.training:
                    mask = torch.zeros_like(alpha)
                    mask.scatter_(1, topk_idx, 1.0)
                    alpha_sparse = mask * alpha
                else:
                    alpha_sparse = torch.zeros_like(alpha)
                    alpha_sparse.scatter_(1, topk_idx, topk_vals)
                alpha_sparse = self._safe_normalize(alpha_sparse, dim=-1)
                alpha_all.append(alpha_sparse)
            else:
                alpha_all.append(alpha)

        alpha_all = torch.stack(alpha_all, dim=1)

        if return_phi:
            return zeta, alpha_all, phi
        return zeta, alpha_all

    def update_ema_gradients(self, domain_id, gradients):
        if isinstance(domain_id, torch.Tensor):
            domain_id = int(domain_id.item())
        else:
            domain_id = int(domain_id)

        device = next(self.parameters()).device

        if not self.ema_initialized or self.ema_gradients is None:
            grad_dim = gradients.shape[0]
            self.ema_gradients = torch.zeros(self.domain_num, grad_dim, device=device)
            self.gradient_count = torch.zeros(self.domain_num, device=device)
            self.rho_self = torch.zeros(self.domain_num, device=device)
            self.ema_initialized = True

        gradients = gradients.to(device)

        if domain_id < self.domain_num:
            self.ema_gradients[domain_id] = (
                self.ema_decay * self.ema_gradients[domain_id] +
                (1 - self.ema_decay) * gradients
            )
            self.gradient_count[domain_id] += 1

    def compute_asymmetric_benefit_score(self, g_target, g_source, eps=1e-8):
        """Paper Eq.(4): rho_{m<-j} = relu(g_j^T g_m / ||g_j||)"""
        g_source_norm = torch.norm(g_source)
        if g_source_norm < eps:
            return torch.tensor(0.0, device=g_target.device)
        projection = torch.dot(g_source, g_target) / (g_source_norm + eps)
        return F.relu(projection)

    def compute_benefit_matrix_from_ema(self):
        """Paper Eq.(4)-(6): Compute R^(0) from EMA gradients."""
        if not self.ema_initialized or self.ema_gradients is None:
            return self.R_benefit

        device = self.ema_gradients.device
        D = self.domain_num
        eps = 1e-8

        rho = torch.zeros(D, D, device=device)

        for m in range(D):
            g_m = self.ema_gradients[m]
            if torch.norm(g_m) < eps:
                continue
            for j in range(D):
                if m != j:
                    g_j = self.ema_gradients[j]
                    if torch.norm(g_j) > eps:
                        rho[m, j] = self.compute_asymmetric_benefit_score(g_m, g_j)

        S = rho.sum(dim=1)
        S_bar = S.mean().item()

        for m in range(D):
            self.rho_self[m] = self.beta * self.rho_self[m] + (1 - self.beta) * S_bar

        R = torch.zeros(D, D, device=device)
        for m in range(D):
            rho_m = self.rho_self[m].item()
            S_m = S[m].item()
            Z_m = rho_m + S_m
            if Z_m > eps:
                R[m, m] = rho_m / Z_m
                for j in range(D):
                    if m != j:
                        R[m, j] = rho[m, j].item() / Z_m
            else:
                R[m, m] = 1.0

        return R

    def set_benefit_matrix(self, R):
        if isinstance(R, torch.Tensor):
            self.R_benefit.copy_(R.detach())
        else:
            self.R_benefit.copy_(torch.tensor(R, device=self.R_benefit.device))

    def get_benefit_matrix(self):
        return self.R_benefit.clone()

    def compute_directional_loss(self, phi, domain_ids):
        """Paper Eq.(13)-(14): Asymmetric routing alignment."""
        unique_domains = torch.unique(domain_ids)
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=phi.device, requires_grad=True)

        R_gated = self.compute_gated_R()
        loss = torch.tensor(0.0, device=phi.device)
        count = 0
        eps = 1e-8

        for i, di in enumerate(unique_domains):
            for j, dj in enumerate(unique_domains):
                if di != dj:
                    mask_i = (domain_ids == di)
                    mask_j = (domain_ids == dj)

                    if mask_i.sum() > 0 and mask_j.sum() > 0:
                        phi_i = phi[mask_i].mean(dim=0)
                        phi_j = phi[mask_j].mean(dim=0)

                        r_ij = R_gated[di, dj]
                        r_ji = R_gated[dj, di]
                        D_ij = F.relu(r_ij - r_ji)

                        if D_ij > 0:
                            phi_i_safe = torch.clamp(phi_i, min=eps, max=1-eps)
                            phi_i_safe = phi_i_safe / phi_i_safe.sum()

                            phi_j_stopped = phi_j.detach()
                            phi_j_stopped = torch.clamp(phi_j_stopped, min=eps, max=1-eps)
                            phi_j_stopped = phi_j_stopped / phi_j_stopped.sum()

                            kl = (phi_i_safe * (phi_i_safe / phi_j_stopped).log()).sum()

                            loss = loss + D_ij * kl
                            count += 1

        if count == 0:
            return torch.tensor(0.0, device=phi.device, requires_grad=True)

        return loss / count

    def initialize_benefit_matrix(self, train_loader, device, max_batches=50):
        """Initialize R^(0) using parameter-space gradients from last backbone layer."""
        self.eval()
        criterion = nn.BCEWithLogitsLoss()

        target_params = [p for p in self.backbone.backbone_layers[-1].parameters()]
        if not target_params:
            self.train()
            return self.R_benefit

        is_epnet = (self.backbone_type == 'epnet')

        for batch_idx, (x_dict, y) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            y = y.to(device)
            domain_ids = x_dict["domain_indicator"].long()

            input_emb = self.embedding(x_dict, self.features, squeeze_dim=True)
            emb_3d = self.embedding(x_dict, self.features, squeeze_dim=False)
            if input_emb.dim() == 1:
                input_emb = input_emb.unsqueeze(0)
            if emb_3d.dim() == 2:
                emb_3d = emb_3d.unsqueeze(0)

            if is_epnet:
                logits = self.backbone.forward_with_lora_experts(
                    input_emb, None, mode='none',
                    emb_3d=emb_3d, return_features=False, domain_id=domain_ids
                )
            else:
                features, aux_logit = self.backbone.forward_with_lora_experts(
                    input_emb, None, mode='none',
                    emb_3d=emb_3d, return_features=True, domain_id=domain_ids
                )

            unique_domains = torch.unique(domain_ids)
            for idx, d in enumerate(unique_domains):
                mask = (domain_ids == d)
                if mask.sum() < 2:
                    continue
                d_int = int(d.item())
                if d_int >= self.domain_num:
                    continue

                if is_epnet:
                    loss = criterion(logits[mask], y[mask].float())
                else:
                    tower_out = self.domain_towers[d_int](features[mask]).squeeze(-1)
                    d_logits = tower_out + aux_logit[mask]
                    loss = criterion(d_logits, y[mask].float())

                if torch.isnan(loss):
                    continue

                is_last = (idx == len(unique_domains) - 1)
                grads = torch.autograd.grad(
                    loss, target_params,
                    retain_graph=not is_last, create_graph=False
                )
                domain_grad = torch.cat([g.detach().flatten() for g in grads])

                if not torch.isnan(domain_grad).any() and torch.norm(domain_grad) > 1e-10:
                    self.update_ema_gradients(d_int, domain_grad)

        R = self.compute_benefit_matrix_from_ema()
        self.set_benefit_matrix(R)
        self.train()
        return R