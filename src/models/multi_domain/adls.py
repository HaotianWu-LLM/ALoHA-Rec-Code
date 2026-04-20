import torch
import torch.nn as nn
import torch.nn.functional as F

from .sharebottom import SharedBottom
from .epnet import EPNet


class _LoRAModule(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(x))


class ADLS(nn.Module):
    def __init__(self, features, domain_num,
                 framework='sharedbottom', extractor='fcn',
                 bottom_params=None, tower_params=None,
                 lora_rank=8, num_experts=4,
                 k_layers=None, k_experts=None,
                 beta=2.0, ema_decay=0.9,
                 stage2_init_seed=42,
                 **extractor_kwargs):
        super().__init__()
        self.features = features
        self.domain_num = domain_num
        self.framework_name = framework.lower()
        self.extractor_name = extractor.lower()
        self.beta = beta
        self.ema_decay = ema_decay

        if self.framework_name == 'sharedbottom':
            bp = bottom_params if bottom_params is not None else {"dims": [128]}
            tp = tower_params if tower_params is not None else {"dims": [8]}
            self.framework = SharedBottom(
                features=features,
                domain_num=domain_num,
                bottom_params=bp,
                tower_params=tp,
                extractor=extractor,
                **extractor_kwargs,
            )
        elif self.framework_name == 'epnet':
            sce_features = [f for f in features if f.name == 'domain_indicator']
            agn_features = [f for f in features if f.name != 'domain_indicator']
            if not sce_features:
                raise ValueError(
                    "ADLS with framework='epnet' requires a SparseFeature named "
                    "'domain_indicator' in `features` to drive the scenario gate."
                )
            bp = bottom_params if bottom_params is not None else {"dims": [128, 64, 32]}
            self.framework = EPNet(
                sce_features=sce_features,
                agn_features=agn_features,
                fcn_dims=bp["dims"],
                extractor=extractor,
                dropout=bp.get("dropout", 0.0),
                **extractor_kwargs,
            )
        else:
            raise ValueError(f"Unknown framework: {framework}")

        cpu_state = torch.get_rng_state()
        cuda_states = None
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
        torch.manual_seed(stage2_init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(stage2_init_seed)

        self._build_stage2_params(lora_rank, num_experts, k_layers, k_experts)

        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

        self.training_stage = 1
        self._lora_enabled = False
        self._routing_cache = None
        self._last_phi = None

        self._hook_handles = []
        for i, linear in enumerate(self.framework.backbone_layers):
            h = linear.register_forward_hook(self._make_lora_hook(i))
            self._hook_handles.append(h)

    def _build_stage2_params(self, lora_rank, num_experts, k_layers, k_experts):
        num_layers = len(self.framework.backbone_layers)
        if num_layers == 0:
            raise RuntimeError(
                "ADLS: framework exposed zero backbone_layers. Check that the "
                "underlying SharedBottom/EPNet built a multi-layer deep network."
            )
        self.num_lora_layers = num_layers
        self.num_experts = num_experts
        self.lora_rank = lora_rank
        self.k_layers = k_layers if k_layers is not None else max(1, num_layers)
        self.k_experts = k_experts if k_experts is not None else max(1, num_experts // 2)

        self.domain_hidden = 64
        self.domain_embeddings = nn.Embedding(self.domain_num, self.domain_hidden)
        nn.init.normal_(self.domain_embeddings.weight, mean=0, std=0.02)
        self.layer_positions = nn.Parameter(torch.randn(num_layers, self.domain_hidden) * 0.02)

        self.inter_layer_router = nn.Sequential(
            nn.Linear(self.domain_hidden * 2, self.domain_hidden), nn.ReLU(),
            nn.Linear(self.domain_hidden, 1)
        )
        self.intra_layer_router = nn.Sequential(
            nn.Linear(self.domain_hidden * 2, self.domain_hidden), nn.ReLU(),
            nn.Linear(self.domain_hidden, num_experts)
        )

        self.lora_experts = nn.ModuleList()
        for linear in self.framework.backbone_layers:
            in_dim, out_dim = linear.in_features, linear.out_features
            experts = nn.ModuleList([
                _LoRAModule(in_dim, out_dim, lora_rank) for _ in range(num_experts)
            ])
            self.lora_experts.append(experts)

        self.relation_gate_logits = nn.Parameter(torch.zeros(self.domain_num, self.domain_num))

        grad_dim = sum(p.numel() for p in self.framework.backbone_layers[-1].parameters())
        self.register_buffer('ema_gradients', torch.zeros(self.domain_num, grad_dim))
        self.register_buffer('ema_initialized', torch.zeros(self.domain_num, dtype=torch.bool))
        self.register_buffer('ema_count', torch.zeros(self.domain_num, dtype=torch.long))
        self.register_buffer('rho_self', torch.zeros(self.domain_num))
        self.register_buffer('R_benefit', torch.eye(self.domain_num))

    def _make_lora_hook(self, layer_idx):
        def hook(module, inputs, output):
            if not self._lora_enabled or self._routing_cache is None:
                return output
            x = inputs[0]
            zeta = self._routing_cache['zeta'][:, layer_idx:layer_idx + 1]
            alpha = self._routing_cache['alpha'][:, layer_idx, :]

            lora_sum = torch.zeros_like(output)
            for e in range(self.num_experts):
                lora_e = self.lora_experts[layer_idx][e](x)
                lora_sum = lora_sum + alpha[:, e:e + 1] * lora_e
            return output + zeta * lora_sum
        return hook

    def set_training_stage(self, stage):
        self.training_stage = int(stage)

    def get_stage1_params(self):
        return list(self.framework.parameters())

    def compute_gated_R(self):
        gate = F.softplus(self.relation_gate_logits)
        R_scaled = self.R_benefit * gate
        row_sum = R_scaled.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return R_scaled / row_sum

    def hierarchical_routing(self, domain_ids, return_phi=False):
        M = self.domain_num
        L = self.num_lora_layers
        E = self.num_experts
        H = self.domain_hidden

        domain_emb_all = self.domain_embeddings.weight
        layer_pos = self.layer_positions

        dom_exp = domain_emb_all.unsqueeze(1).expand(M, L, H)
        pos_exp = layer_pos.unsqueeze(0).expand(M, L, H)
        router_inp = torch.cat([dom_exp, pos_exp], dim=-1)

        zeta_logits = self.inter_layer_router(router_inp).squeeze(-1)
        if self.k_layers < L:
            topk_vals, topk_idx = zeta_logits.topk(self.k_layers, dim=-1)
            zeta_sparse = torch.full_like(zeta_logits, float('-inf'))
            zeta_sparse.scatter_(-1, topk_idx, topk_vals)
            zeta_all = F.softmax(zeta_sparse, dim=-1)
        else:
            zeta_all = F.softmax(zeta_logits, dim=-1)

        alpha_logits = self.intra_layer_router(router_inp)
        if self.k_experts < E:
            topk_vals_a, topk_idx_a = alpha_logits.topk(self.k_experts, dim=-1)
            alpha_sparse = torch.full_like(alpha_logits, float('-inf'))
            alpha_sparse.scatter_(-1, topk_idx_a, topk_vals_a)
            alpha_all = F.softmax(alpha_sparse, dim=-1)
        else:
            alpha_all = F.softmax(alpha_logits, dim=-1)

        R_gated = self.compute_gated_R()
        zeta_agg = R_gated @ zeta_all
        alpha_flat = alpha_all.view(M, L * E)
        alpha_agg = (R_gated @ alpha_flat).view(M, L, E)

        zeta = zeta_agg[domain_ids]
        alpha = alpha_agg[domain_ids]

        if return_phi:
            phi_all = dom_exp + pos_exp
            phi = phi_all[domain_ids]
            return zeta, alpha, phi
        return zeta, alpha, None

    def forward(self, x):
        if self.training_stage == 1:
            self._lora_enabled = False
            self._routing_cache = None
            return self.framework(x)

        domain_id = x["domain_indicator"].long()
        if domain_id.dim() > 1:
            domain_id = domain_id.squeeze(-1)
        zeta, alpha, phi = self.hierarchical_routing(domain_id, return_phi=True)
        self._routing_cache = {'zeta': zeta, 'alpha': alpha}
        self._lora_enabled = True
        try:
            out = self.framework(x)
        finally:
            self._lora_enabled = False
            self._routing_cache = None
        self._last_phi = phi
        return out

    def update_ema_gradients(self, domain_id, grad_vec):
        if not self.ema_initialized[domain_id]:
            self.ema_gradients[domain_id] = grad_vec.detach()
            self.ema_initialized[domain_id] = True
        else:
            self.ema_gradients[domain_id] = (
                self.ema_decay * self.ema_gradients[domain_id]
                + (1.0 - self.ema_decay) * grad_vec.detach()
            )
        self.ema_count[domain_id] += 1
        g = self.ema_gradients[domain_id]
        self.rho_self[domain_id] = g.norm().clamp_min(1e-12)

    def compute_benefit_matrix_from_ema(self):
        M = self.domain_num
        if not bool(self.ema_initialized.all().item()):
            return self.R_benefit
        R = torch.zeros(M, M, device=self.ema_gradients.device)
        for i in range(M):
            for j in range(M):
                if i == j:
                    R[i, j] = 1.0
                    continue
                gi = self.ema_gradients[i]
                gj = self.ema_gradients[j]
                ni = gi.norm().clamp_min(1e-12)
                nj = gj.norm().clamp_min(1e-12)
                cos = (gi * gj).sum() / (ni * nj)
                R[i, j] = F.relu(cos)
        return R

    def set_benefit_matrix(self, R):
        self.R_benefit.copy_(R.detach())

    def compute_directional_loss(self, phi, domain_ids):
        if phi is None:
            return torch.tensor(0.0, device=self.relation_gate_logits.device)
        B = phi.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=phi.device)
        phi_flat = phi.mean(dim=1)
        loss = 0.0
        count = 0
        unique_domains = torch.unique(domain_ids)
        if unique_domains.numel() <= 1:
            return torch.tensor(0.0, device=phi.device)
        R_gated = self.compute_gated_R()
        for i_idx in range(unique_domains.numel()):
            for j_idx in range(i_idx + 1, unique_domains.numel()):
                mi = unique_domains[i_idx].item()
                mj = unique_domains[j_idx].item()
                mask_i = (domain_ids == mi)
                mask_j = (domain_ids == mj)
                if mask_i.sum() == 0 or mask_j.sum() == 0:
                    continue
                mu_i = phi_flat[mask_i].mean(dim=0)
                mu_j = phi_flat[mask_j].mean(dim=0)
                sim = F.cosine_similarity(mu_i.unsqueeze(0), mu_j.unsqueeze(0))
                target = R_gated[mi, mj].detach()
                loss = loss + (sim - target).pow(2).mean()
                count += 1
        if count == 0:
            return torch.tensor(0.0, device=phi.device)
        return loss / count