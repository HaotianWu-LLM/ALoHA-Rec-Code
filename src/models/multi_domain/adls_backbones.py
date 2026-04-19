import torch
import torch.nn as nn

from ...basic.layers import EmbeddingLayer, MLP, GateNU, build_extractor
from .sharebottom import SharedBottom
from .epnet import EPNet


def _find_mlp_linears(module, exclude_final_binary=False):
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if exclude_final_binary and len(linears) > 1 and linears[-1].out_features == 1:
        return linears[:-1]
    return linears


class SharedBottomFramework(nn.Module):
    def __init__(self, features, domain_num, extractor='fcn',
                 bottom_params=None, tower_params=None, **extractor_kwargs):
        super().__init__()
        self.extractor_name = extractor.lower()
        self.features = features
        self.domain_num = domain_num

        if bottom_params is None:
            bottom_params = {"dims": [128], "dropout": 0.0, "activation": "relu"}
        if tower_params is None:
            tower_params = {"dims": [8], "dropout": 0.0, "activation": "relu"}

        if self.extractor_name == 'fcn':
            self.base = SharedBottom(features, domain_num, bottom_params, tower_params)
            self._hook_linears = _find_mlp_linears(self.base.bottom_mlp, exclude_final_binary=False)
            self.feature_dim = bottom_params['dims'][-1]
        else:
            self.embedding = EmbeddingLayer(features)
            from ...basic.features import SparseFeature, DenseFeature
            sparse_features = [f for f in features if isinstance(f, SparseFeature)]
            dense_features = [f for f in features if isinstance(f, DenseFeature)]
            input_dim = sum(f.embed_dim for f in sparse_features) + len(dense_features)
            num_fields = len(sparse_features)
            embed_dim = sparse_features[0].embed_dim if sparse_features else None
            self.extractor = build_extractor(
                self.extractor_name, input_dim,
                num_fields=num_fields, embed_dim=embed_dim,
                deep_dims=bottom_params['dims'],
                dropout=bottom_params.get('dropout', 0.0),
                **extractor_kwargs
            )
            tower_dropout = tower_params.get('dropout', 0.0)
            tower_activation = tower_params.get('activation', 'relu')
            self.towers = nn.ModuleList([
                MLP(self.extractor.feature_dim, output_layer=True,
                    dims=tower_params['dims'], dropout=tower_dropout,
                    activation=tower_activation)
                for _ in range(domain_num)
            ])
            self._hook_linears = list(self.extractor.backbone_layers)
            self.feature_dim = self.extractor.feature_dim
            self._sparse_features = sparse_features
            self._dense_features = dense_features

    @property
    def backbone_layers(self):
        return self._hook_linears

    def forward(self, x):
        if self.extractor_name == 'fcn':
            return self.base(x)

        domain_id = x["domain_indicator"].long()
        input_flat = self.embedding(x, self.features, squeeze_dim=True)
        emb_3d = None
        if self._sparse_features:
            emb_3d = self.embedding(x, self._sparse_features, squeeze_dim=False)

        features_out = self.extractor(input_flat, emb_3d=emb_3d)
        if isinstance(features_out, tuple):
            features_repr, aux_logit = features_out
        else:
            features_repr = features_out
            aux_logit = torch.zeros(features_repr.shape[0], device=features_repr.device)

        B = features_repr.shape[0]
        result = torch.zeros(B, device=features_repr.device)
        for d in range(self.domain_num):
            mask = (domain_id == d)
            if mask.any():
                tower_logit = self.towers[d](features_repr[mask]).squeeze(-1)
                logit = tower_logit + aux_logit[mask]
                result[mask] = torch.sigmoid(logit)
        return result


class EPNetFramework(nn.Module):
    def __init__(self, features, domain_num, extractor='fcn',
                 bottom_params=None, tower_params=None, **extractor_kwargs):
        super().__init__()
        self.extractor_name = extractor.lower()
        self.features = features
        self.domain_num = domain_num

        if bottom_params is None:
            bottom_params = {"dims": [128, 64, 32], "dropout": 0.2}

        self.sce_features = [f for f in features if f.name == 'domain_indicator']
        self.agn_features = [f for f in features if f.name != 'domain_indicator']

        if not self.sce_features:
            raise ValueError(
                "EPNetFramework requires a SparseFeature with name 'domain_indicator' in features."
            )

        if self.extractor_name == 'fcn':
            self.base = EPNet(
                sce_features=self.sce_features,
                agn_features=self.agn_features,
                fcn_dims=bottom_params['dims']
            )
            self._hook_linears = _find_mlp_linears(self.base.mlp, exclude_final_binary=True)
            if len(self._hook_linears) == 0:
                raise RuntimeError(
                    "EPNetFramework(extractor='fcn'): no hidden Linear layers found in "
                    "EPNet.mlp. Check that EPNet builds self.mlp as MLP(agn_dims, dims=fcn_dims) "
                    "so that fcn_dims produces hidden layers (currently found 0 or only the "
                    "final binary head)."
                )
            self.feature_dim = bottom_params['dims'][-1]
        else:
            self.sce_embedding = EmbeddingLayer(self.sce_features)
            self.agn_embedding = EmbeddingLayer(self.agn_features)

            from ...basic.features import SparseFeature, DenseFeature
            agn_sparse = [f for f in self.agn_features if isinstance(f, SparseFeature)]
            agn_dense = [f for f in self.agn_features if isinstance(f, DenseFeature)]
            self.sce_dim = sum(f.embed_dim for f in self.sce_features)
            self.agn_dim = sum(f.embed_dim for f in agn_sparse) + len(agn_dense)

            self.gatenu = GateNU(self.sce_dim + self.agn_dim, self.agn_dim)

            num_fields = len(agn_sparse)
            embed_dim = agn_sparse[0].embed_dim if agn_sparse else None
            self.extractor = build_extractor(
                self.extractor_name, self.agn_dim,
                num_fields=num_fields, embed_dim=embed_dim,
                deep_dims=bottom_params['dims'],
                dropout=bottom_params.get('dropout', 0.0),
                **extractor_kwargs
            )

            self.output_layer = nn.Linear(self.extractor.feature_dim, 1)
            self._hook_linears = list(self.extractor.backbone_layers)
            self.feature_dim = self.extractor.feature_dim
            self._agn_sparse = agn_sparse
            self._agn_dense = agn_dense

    @property
    def backbone_layers(self):
        return self._hook_linears

    def forward(self, x):
        if self.extractor_name == 'fcn':
            return self.base(x)

        sce_x = self.sce_embedding(x, self.sce_features, squeeze_dim=True)
        agn_x = self.agn_embedding(x, self.agn_features, squeeze_dim=True)
        gate_input = torch.cat((sce_x, agn_x.detach()), dim=1)
        gate_output = self.gatenu(gate_input)

        emb_3d = None
        if self._agn_sparse:
            emb_3d = self.agn_embedding(x, self._agn_sparse, squeeze_dim=False)

        if emb_3d is not None and not self._agn_dense:
            B = gate_output.shape[0]
            gate_3d = gate_output.view(B, len(self._agn_sparse), -1)
            emb_3d = emb_3d * gate_3d
            agn_x = emb_3d.flatten(start_dim=1)
        else:
            agn_x = agn_x * gate_output

        features_out = self.extractor(agn_x, emb_3d=emb_3d)
        if isinstance(features_out, tuple):
            features_repr, aux_logit = features_out
        else:
            features_repr = features_out
            aux_logit = torch.zeros(features_repr.shape[0], device=features_repr.device)

        logit = self.output_layer(features_repr).squeeze(-1) + aux_logit
        return torch.sigmoid(logit)