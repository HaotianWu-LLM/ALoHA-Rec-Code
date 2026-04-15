"""
ALoHA-Rec Backbone Implementations with LoRA insertion points.
Supported Backbones: FCN, DeepFM, DCN, xDeepFM, AutoInt
All backbones return LOGITS (no sigmoid).

Changes:
  - ADLSBackbone base class: added abstract feature_dim property, return_features param
  - Each backbone: added _feature_dim attribute, feature_dim property
  - Each backbone forward_with_lora_experts: added return_features=False param
    When return_features=True, returns (features, aux_logit) instead of logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ADLSBackbone(nn.Module):
    """Base class for ADLS backbones"""
    def __init__(self):
        super().__init__()
        self.backbone_layers = []

    @property
    def feature_dim(self):
        raise NotImplementedError

    def get_lora_configs(self):
        """Return list of dicts with layer_idx, in_dim, out_dim for LoRA insertion"""
        raise NotImplementedError

    def forward_with_lora_experts(self, x, lora_experts, mode='none', zeta=None, alpha=None, num_experts=None,
                                   return_features=False, **kwargs):
        raise NotImplementedError


def _apply_lora(h_base, h_in, lora_experts, lora_idx, mode, zeta, alpha, num_experts):
    """
    Apply LoRA adaptation to base output with numerical stability.
    """
    if lora_experts is None or lora_idx >= len(lora_experts):
        return h_base

    if mode == 'none':
        return h_base

    elif mode == 'uniform':
        lora_sum = torch.zeros_like(h_base)
        for m in range(num_experts):
            lora_out = lora_experts[lora_idx][m](h_in)
            lora_sum = lora_sum + lora_out
        return h_base + lora_sum / num_experts

    elif mode == 'routed':
        layer_weight = zeta[:, lora_idx:lora_idx+1]
        expert_weights = alpha[:, lora_idx, :]

        lora_out = torch.zeros_like(h_base)
        for m in range(len(lora_experts[lora_idx])):
            expert_out = lora_experts[lora_idx][m](h_in)
            lora_out = lora_out + expert_weights[:, m:m+1] * expert_out

        return h_base + layer_weight * lora_out

    return h_base


class FCNBackbone(ADLSBackbone):
    """Fully Connected Network Backbone."""
    
    def __init__(self, input_dim, fcn_dims=None, dropout=0.2, activation='relu'):
        super().__init__()
        if fcn_dims is None:
            fcn_dims = [256, 128, 64]

        self.input_dim = input_dim
        self.fcn_dims = fcn_dims
        self.num_hidden_layers = len(fcn_dims)
        self._feature_dim = fcn_dims[-1]

        dims = [input_dim] + fcn_dims
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.hidden_layers.append(nn.Linear(dims[i], dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(dims[i+1]))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.output_layer = nn.Linear(fcn_dims[-1], 1)
        self._lora_dims = dims
        self.backbone_layers = list(self.hidden_layers)
        self._init_weights()

    @property
    def feature_dim(self):
        return self._feature_dim

    def _init_weights(self):
        for layer in self.hidden_layers:
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def get_lora_configs(self):
        cfgs = []
        for i in range(self.num_hidden_layers):
            cfgs.append({
                'layer_idx': i,
                'in_dim': self._lora_dims[i],
                'out_dim': self._lora_dims[i+1],
                'description': f'Hidden Layer {i+1}'
            })
        return cfgs

    def forward_with_lora_experts(self, x, lora_experts, mode='none', zeta=None, alpha=None, num_experts=None,
                                   return_features=False, **kwargs):
        h = x
        for i in range(self.num_hidden_layers):
            h_in = h
            h = self.hidden_layers[i](h)
            h = _apply_lora(h, h_in, lora_experts, i, mode, zeta, alpha, num_experts)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout_layers[i](h)

        if return_features:
            return h, torch.zeros(h.shape[0], device=h.device)

        logits = self.output_layer(h)
        return logits.squeeze(-1)


class DeepFMBackbone(ADLSBackbone):
    """DeepFM Backbone: FM first-order + FM second-order + Deep Network."""
    
    def __init__(self, input_dim, deep_dims=None, dropout=0.2, num_fields=None, embed_dim=None):
        super().__init__()
        if deep_dims is None:
            deep_dims = [256, 128, 64]

        self.input_dim = input_dim
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.num_hidden_layers = len(deep_dims)
        self._feature_dim = deep_dims[-1]

        self.fm_first_order = nn.Linear(input_dim, 1)

        dims = [input_dim] + deep_dims
        self.deep_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.deep_layers.append(nn.Linear(dims[i], dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(dims[i+1]))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.deep_output = nn.Linear(deep_dims[-1], 1)
        self._lora_dims = dims
        self.backbone_layers = list(self.deep_layers)
        self._init_weights()

    @property
    def feature_dim(self):
        return self._feature_dim

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fm_first_order.weight)
        nn.init.zeros_(self.fm_first_order.bias)
        for layer in self.deep_layers:
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.deep_output.weight)
        nn.init.zeros_(self.deep_output.bias)

    def get_lora_configs(self):
        cfgs = []
        for i in range(self.num_hidden_layers):
            cfgs.append({
                'layer_idx': i,
                'in_dim': self._lora_dims[i],
                'out_dim': self._lora_dims[i+1],
                'description': f'Deep Layer {i+1}'
            })
        return cfgs

    def forward_with_lora_experts(self, x, lora_experts, mode='none', zeta=None, alpha=None, num_experts=None,
                                   return_features=False, emb_3d=None, **kwargs):
        fm_first = self.fm_first_order(x)

        if emb_3d is not None:
            sum_emb = emb_3d.sum(dim=1)
            square_sum = sum_emb ** 2
            sum_square = (emb_3d ** 2).sum(dim=1)
            fm_second = 0.5 * (square_sum - sum_square).sum(dim=1, keepdim=True)
        else:
            fm_second = 0.0

        h = x
        for i in range(self.num_hidden_layers):
            h_in = h
            h = self.deep_layers[i](h)
            h = _apply_lora(h, h_in, lora_experts, i, mode, zeta, alpha, num_experts)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout_layers[i](h)

        if return_features:
            aux = fm_first.squeeze(-1)
            if isinstance(fm_second, torch.Tensor):
                aux = aux + fm_second.squeeze(-1)
            return h, aux

        deep_out = self.deep_output(h)
        logits = fm_first + fm_second + deep_out
        return logits.squeeze(-1)


class DCNBackbone(ADLSBackbone):
    """Deep & Cross Network Backbone."""
    
    def __init__(self, input_dim, num_cross_layers=3, deep_dims=None, dropout=0.2):
        super().__init__()
        if deep_dims is None:
            deep_dims = [256, 128, 64]

        self.input_dim = input_dim
        self.num_cross_layers = num_cross_layers
        self.num_hidden_layers = len(deep_dims)
        self._feature_dim = input_dim + deep_dims[-1]

        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, 1)) for _ in range(num_cross_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_cross_layers)
        ])

        dims = [input_dim] + deep_dims
        self.deep_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.deep_layers.append(nn.Linear(dims[i], dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(dims[i+1]))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.combine_layer = nn.Linear(input_dim + deep_dims[-1], 1)
        self._lora_dims = dims
        self.backbone_layers = list(self.deep_layers)
        self._init_weights()

    @property
    def feature_dim(self):
        return self._feature_dim

    def _init_weights(self):
        for w in self.cross_weights:
            nn.init.xavier_uniform_(w)
        for layer in self.deep_layers:
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.combine_layer.weight)
        nn.init.zeros_(self.combine_layer.bias)

    def get_lora_configs(self):
        cfgs = []
        for i in range(self.num_hidden_layers):
            cfgs.append({
                'layer_idx': i,
                'in_dim': self._lora_dims[i],
                'out_dim': self._lora_dims[i+1],
                'description': f'Deep Layer {i+1}'
            })
        return cfgs

    def forward_with_lora_experts(self, x, lora_experts, mode='none', zeta=None, alpha=None, num_experts=None,
                                   return_features=False, **kwargs):
        x0 = x
        x_cross = x
        for i in range(self.num_cross_layers):
            xw = torch.matmul(x_cross, self.cross_weights[i])
            x_cross = x0 * xw + self.cross_biases[i] + x_cross

        h = x
        for i in range(self.num_hidden_layers):
            h_in = h
            h = self.deep_layers[i](h)
            h = _apply_lora(h, h_in, lora_experts, i, mode, zeta, alpha, num_experts)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout_layers[i](h)

        combined = torch.cat([x_cross, h], dim=1)

        if return_features:
            return combined, torch.zeros(combined.shape[0], device=combined.device)

        logits = self.combine_layer(combined)
        return logits.squeeze(-1)


class CIN(nn.Module):
    """Compressed Interaction Network for xDeepFM"""
    
    def __init__(self, num_fields, cin_layer_sizes, split_half=True):
        super().__init__()
        self.num_fields = num_fields
        self.cin_layer_sizes = cin_layer_sizes
        self.split_half = split_half

        self.conv_layers = nn.ModuleList()
        prev_layer_size = num_fields

        for i, layer_size in enumerate(cin_layer_sizes):
            self.conv_layers.append(
                nn.Conv1d(num_fields * prev_layer_size, layer_size, kernel_size=1)
            )
            if split_half and i < len(cin_layer_sizes) - 1:
                prev_layer_size = layer_size // 2
            else:
                prev_layer_size = layer_size

    def forward(self, x):
        batch_size = x.shape[0]
        embed_dim = x.shape[2]

        hidden_layers = [x]
        final_results = []

        for i, conv in enumerate(self.conv_layers):
            x0 = hidden_layers[0]
            xi = hidden_layers[-1]

            outer = torch.einsum('bmd,bnd->bmnd', x0, xi)
            outer = outer.view(batch_size, -1, embed_dim)

            out = conv(outer)
            out = F.relu(out)

            if self.split_half and i < len(self.cin_layer_sizes) - 1:
                out, direct = torch.split(out, out.shape[1] // 2, dim=1)
                final_results.append(direct)
            else:
                final_results.append(out)

            hidden_layers.append(out)

        result = torch.cat(final_results, dim=1)
        result = torch.sum(result, dim=2)
        return result


class xDeepFMBackbone(ADLSBackbone):
    """xDeepFM Backbone: Linear + CIN + Deep Network."""
    
    def __init__(self, input_dim, cin_layer_sizes=None, deep_dims=None, dropout=0.2,
                 num_fields=None, embed_dim=None):
        super().__init__()
        if cin_layer_sizes is None:
            cin_layer_sizes = [128, 128]
        if deep_dims is None:
            deep_dims = [256, 128, 64]

        self.input_dim = input_dim
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.num_hidden_layers = len(deep_dims)
        self._feature_dim = deep_dims[-1]

        self.linear = nn.Linear(input_dim, 1)

        if num_fields is not None:
            self.cin = CIN(num_fields, cin_layer_sizes, split_half=True)
            cin_output_dim = sum([size // 2 for size in cin_layer_sizes[:-1]]) + cin_layer_sizes[-1]
            self.cin_output = nn.Linear(cin_output_dim, 1)
        else:
            self.cin = None
            self.cin_output = None

        dims = [input_dim] + deep_dims
        self.deep_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.deep_layers.append(nn.Linear(dims[i], dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(dims[i+1]))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.deep_output = nn.Linear(deep_dims[-1], 1)
        self._lora_dims = dims
        self.backbone_layers = list(self.deep_layers)
        self._init_weights()

    @property
    def feature_dim(self):
        return self._feature_dim

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        for layer in self.deep_layers:
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.deep_output.weight)
        nn.init.zeros_(self.deep_output.bias)

    def get_lora_configs(self):
        cfgs = []
        for i in range(self.num_hidden_layers):
            cfgs.append({
                'layer_idx': i,
                'in_dim': self._lora_dims[i],
                'out_dim': self._lora_dims[i+1],
                'description': f'Deep Layer {i+1}'
            })
        return cfgs

    def forward_with_lora_experts(self, x, lora_experts, mode='none', zeta=None, alpha=None, num_experts=None,
                                   return_features=False, emb_3d=None, **kwargs):
        linear_out = self.linear(x)

        if self.cin is not None and emb_3d is not None:
            cin_out = self.cin(emb_3d)
            cin_out = self.cin_output(cin_out)
        else:
            cin_out = 0.0

        h = x
        for i in range(self.num_hidden_layers):
            h_in = h
            h = self.deep_layers[i](h)
            h = _apply_lora(h, h_in, lora_experts, i, mode, zeta, alpha, num_experts)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout_layers[i](h)

        if return_features:
            aux = linear_out.squeeze(-1)
            if isinstance(cin_out, torch.Tensor):
                aux = aux + cin_out.squeeze(-1)
            return h, aux

        deep_out = self.deep_output(h)
        logits = linear_out + cin_out + deep_out
        return logits.squeeze(-1)


class AutoIntBackbone(ADLSBackbone):
    """AutoInt Backbone: Multi-head Self-Attention + Deep Network."""
    
    def __init__(self, input_dim, deep_dims=None, dropout=0.2,
                 num_heads=4, num_attention_layers=3, num_fields=None, embed_dim=None):
        super().__init__()
        if deep_dims is None:
            deep_dims = [256, 128, 64]

        self.input_dim = input_dim
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.num_attention_layers = num_attention_layers
        self.num_hidden_layers = len(deep_dims)
        self._feature_dim = deep_dims[-1]

        if num_fields is not None and embed_dim is not None:
            self.attention_dim = (embed_dim // num_heads) * num_heads
            if self.attention_dim != embed_dim:
                self.proj = nn.Linear(embed_dim, self.attention_dim)
            else:
                self.proj = None

            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=self.attention_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ) for _ in range(num_attention_layers)
            ])
            self.attention_norms = nn.ModuleList([
                nn.LayerNorm(self.attention_dim) for _ in range(num_attention_layers)
            ])

            attn_output_dim = num_fields * self.attention_dim
        else:
            self.attention_layers = None
            self.attention_norms = None
            self.proj = None
            self.attention_dim = None
            attn_output_dim = input_dim

        dims = [attn_output_dim] + deep_dims
        self.deep_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.deep_layers.append(nn.Linear(dims[i], dims[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(dims[i+1]))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.output_layer = nn.Linear(deep_dims[-1], 1)
        self._lora_dims = dims
        self.backbone_layers = list(self.deep_layers)
        self._init_weights()

    @property
    def feature_dim(self):
        return self._feature_dim

    def _init_weights(self):
        for layer in self.deep_layers:
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def get_lora_configs(self):
        cfgs = []
        for i in range(self.num_hidden_layers):
            cfgs.append({
                'layer_idx': i,
                'in_dim': self._lora_dims[i],
                'out_dim': self._lora_dims[i+1],
                'description': f'Deep Layer {i+1}'
            })
        return cfgs

    def forward_with_lora_experts(self, x, lora_experts, mode='none', zeta=None, alpha=None, num_experts=None,
                                   return_features=False, emb_3d=None, **kwargs):
        if self.attention_layers is not None and emb_3d is not None:
            attn_input = emb_3d
            if self.proj is not None:
                attn_input = self.proj(attn_input)

            attn_output = attn_input
            for i in range(self.num_attention_layers):
                residual = attn_output
                attn_output, _ = self.attention_layers[i](attn_output, attn_output, attn_output)
                attn_output = self.attention_norms[i](attn_output + residual)

            h = attn_output.flatten(start_dim=1)
        else:
            h = x

        for i in range(self.num_hidden_layers):
            h_in = h
            h = self.deep_layers[i](h)
            h = _apply_lora(h, h_in, lora_experts, i, mode, zeta, alpha, num_experts)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout_layers[i](h)

        if return_features:
            return h, torch.zeros(h.shape[0], device=h.device)

        logits = self.output_layer(h)
        return logits.squeeze(-1)


class _GateNU(nn.Module):
    """Scenario-aware gating unit (from EPNet)."""

    def __init__(self, input_dim, output_dim, hidden_dim=None, gamma=2.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        self.gamma = gamma
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x) * self.gamma


class EPNetBackbone(ADLSBackbone):
    """EPNet Backbone: Scenario-aware Gating + Deep Network.

    The gate modulates agnostic features using scenario (domain) context,
    then feeds through an MLP. Requires `domain_id` in forward kwargs.
    """

    def __init__(self, input_dim, domain_num, fcn_dims=None, dropout=0.2,
                 domain_embed_dim=16, gate_gamma=2.0, **kwargs):
        super().__init__()
        if fcn_dims is None:
            fcn_dims = [256, 128, 64]

        self.input_dim = input_dim
        self.domain_num = domain_num
        self.num_hidden_layers = len(fcn_dims)
        self._feature_dim = fcn_dims[-1]

        self.sce_embedding = nn.Embedding(domain_num, domain_embed_dim)
        self.gatenu = _GateNU(domain_embed_dim + input_dim, input_dim, gamma=gate_gamma)

        dims = [input_dim] + fcn_dims
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.hidden_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.bn_layers.append(nn.BatchNorm1d(dims[i + 1]))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.output_layer = nn.Linear(fcn_dims[-1], 1)
        self._lora_dims = dims
        self.backbone_layers = list(self.hidden_layers)
        self._init_weights()

    @property
    def feature_dim(self):
        return self._feature_dim

    def _init_weights(self):
        nn.init.normal_(self.sce_embedding.weight, mean=0.0, std=0.0001)
        for layer in self.hidden_layers:
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def get_lora_configs(self):
        cfgs = []
        for i in range(self.num_hidden_layers):
            cfgs.append({
                'layer_idx': i,
                'in_dim': self._lora_dims[i],
                'out_dim': self._lora_dims[i + 1],
                'description': f'Hidden Layer {i + 1}'
            })
        return cfgs

    def forward_with_lora_experts(self, x, lora_experts, mode='none', zeta=None, alpha=None, num_experts=None,
                                   return_features=False, domain_id=None, **kwargs):
        if domain_id is not None:
            sce_emb = self.sce_embedding(domain_id)
            gate_input = torch.cat([sce_emb, x.detach()], dim=1)
            gate_output = self.gatenu(gate_input)
            h = x * gate_output
        else:
            h = x

        for i in range(self.num_hidden_layers):
            h_in = h
            h = self.hidden_layers[i](h)
            h = _apply_lora(h, h_in, lora_experts, i, mode, zeta, alpha, num_experts)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout_layers[i](h)

        if return_features:
            return h, torch.zeros(h.shape[0], device=h.device)

        logits = self.output_layer(h)
        return logits.squeeze(-1)


# Aliases
MLPBackbone = FCNBackbone


def get_adls_backbone(backbone_type, input_dim, **kwargs):
    """Factory function to create ADLS backbone."""
    backbone_type = backbone_type.lower()

    if backbone_type in ['fcn', 'mlp']:
        return FCNBackbone(input_dim, **kwargs)
    elif backbone_type == 'deepfm':
        return DeepFMBackbone(input_dim, **kwargs)
    elif backbone_type == 'dcn':
        return DCNBackbone(input_dim, **kwargs)
    elif backbone_type == 'xdeepfm':
        return xDeepFMBackbone(input_dim, **kwargs)
    elif backbone_type == 'autoint':
        return AutoIntBackbone(input_dim, **kwargs)
    elif backbone_type == 'epnet':
        return EPNetBackbone(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")