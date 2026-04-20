import torch
from torch import nn

from ...basic.layers import MLP, EmbeddingLayer, GateNU, build_extractor
from ...basic.features import SparseFeature, DenseFeature


class EPNet(nn.Module):
    def __init__(self, sce_features, agn_features, fcn_dims,
                 extractor='fcn', dropout=0.0, **extractor_kwargs):
        super().__init__()
        self.sce_features = sce_features
        self.agn_features = agn_features
        self.extractor_name = extractor.lower()

        self.sce_embedding = EmbeddingLayer(sce_features)
        self.agn_embedding = EmbeddingLayer(agn_features)
        self.sce_dims = sum(fea.embed_dim for fea in sce_features)
        self.agn_dims = sum(fea.embed_dim for fea in agn_features)
        self.gatenu = GateNU(self.sce_dims + self.agn_dims, self.agn_dims)

        self._agn_sparse = [f for f in agn_features if isinstance(f, SparseFeature)]
        self._agn_dense = [f for f in agn_features if isinstance(f, DenseFeature)]
        self._sparse_numel = sum(f.embed_dim for f in self._agn_sparse)
        self._sparse_embed_dim = (
            self._agn_sparse[0].embed_dim if self._agn_sparse else None
        )

        if self.extractor_name == 'fcn':
            # Now that layers.MLP uses LayerNorm internally (no BN drift under
            # per-sample PEP gating), we can use it directly.
            # output_layer=True appends the final Linear(dims[-1], 1).
            self.extractor = MLP(self.agn_dims, output_layer=True,
                                 dims=fcn_dims, dropout=dropout)
            self.output_layer = None
            # Expose only the hidden Linear layers as LoRA targets — matches
            # the other extractors, which expose their `deep_layers` and not
            # their final output head.
            all_linears = [m for m in self.extractor.modules() if isinstance(m, nn.Linear)]
            if len(all_linears) > 1 and all_linears[-1].out_features == 1:
                self._backbone_layers = all_linears[:-1]
            else:
                self._backbone_layers = all_linears
        else:
            num_fields = len(self._agn_sparse)
            embed_dim = self._sparse_embed_dim
            self.extractor = build_extractor(
                self.extractor_name, self.agn_dims,
                num_fields=num_fields, embed_dim=embed_dim,
                deep_dims=fcn_dims, dropout=dropout, **extractor_kwargs,
            )
            self.output_layer = nn.Linear(self.extractor.feature_dim, 1)
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)
            self._backbone_layers = list(self.extractor.backbone_layers)

    @property
    def backbone_layers(self):
        return self._backbone_layers

    def forward(self, x):
        sce_x = self.sce_embedding(x, self.sce_features, squeeze_dim=True)
        agn_x = self.agn_embedding(x, self.agn_features, squeeze_dim=True)
        gate = self.gatenu(torch.cat((sce_x, agn_x.detach()), dim=1))

        agn_x_gated = agn_x * gate

        if self.extractor_name == 'fcn':
            output = self.extractor(agn_x_gated)
            return torch.sigmoid(output).squeeze()

        emb_3d = None
        if self._agn_sparse:
            emb_3d = self.agn_embedding(x, self._agn_sparse, squeeze_dim=False)
            gate_sparse_3d = gate[:, :self._sparse_numel].view(
                -1, len(self._agn_sparse), self._sparse_embed_dim
            )
            emb_3d = emb_3d * gate_sparse_3d

        features, aux = self.extractor(agn_x_gated, emb_3d=emb_3d)
        logit = self.output_layer(features).squeeze(-1) + aux
        return torch.sigmoid(logit)