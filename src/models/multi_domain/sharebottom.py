import torch
import torch.nn as nn

from ...basic.layers import MLP, EmbeddingLayer, build_extractor
from ...basic.features import SparseFeature, DenseFeature


class SharedBottom(nn.Module):
    def __init__(self, features, domain_num, bottom_params, tower_params,
                 extractor='fcn', **extractor_kwargs):
        super().__init__()
        self.features = features
        self.embedding = EmbeddingLayer(features)
        self.bottom_dims = sum(fea.embed_dim for fea in features)
        self.domain_num = domain_num
        self.extractor_name = extractor.lower()

        if self.extractor_name == 'fcn':
            # layers.MLP is now LN-based; output_layer=False so it returns a
            # representation tensor that the per-domain towers consume.
            self.bottom_mlp = MLP(
                self.bottom_dims,
                **{**bottom_params, **{"output_layer": False}},
            )
            self._feature_dim = bottom_params["dims"][-1]
            self._backbone_layers = [
                m for m in self.bottom_mlp.modules() if isinstance(m, nn.Linear)
            ]
            self._sparse_features = None
            self._dense_features = None
        else:
            self._sparse_features = [f for f in features if isinstance(f, SparseFeature)]
            self._dense_features = [f for f in features if isinstance(f, DenseFeature)]
            num_fields = len(self._sparse_features)
            embed_dim = self._sparse_features[0].embed_dim if self._sparse_features else None
            self.bottom_mlp = build_extractor(
                self.extractor_name, self.bottom_dims,
                num_fields=num_fields, embed_dim=embed_dim,
                deep_dims=bottom_params["dims"],
                dropout=bottom_params.get("dropout", 0.0),
                **extractor_kwargs,
            )
            self._feature_dim = self.bottom_mlp.feature_dim
            self._backbone_layers = list(self.bottom_mlp.backbone_layers)

        self.towers = nn.ModuleList(
            MLP(self._feature_dim, **tower_params) for _ in range(self.domain_num)
        )

    @property
    def backbone_layers(self):
        return self._backbone_layers

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()
        input_bottom = self.embedding(x, self.features, squeeze_dim=True)

        if self.extractor_name == 'fcn':
            features_repr = self.bottom_mlp(input_bottom)
            aux = None
        else:
            emb_3d = None
            if self._sparse_features:
                emb_3d = self.embedding(x, self._sparse_features, squeeze_dim=False)
            features_repr, aux = self.bottom_mlp(input_bottom, emb_3d=emb_3d)

        final = torch.zeros(features_repr.shape[0], device=features_repr.device)
        for d in range(self.domain_num):
            mask = (domain_id == d)
            tower_logit = self.towers[d](features_repr).squeeze(-1)
            if aux is not None:
                tower_logit = tower_logit + aux
            y_d = torch.sigmoid(tower_logit)
            final = torch.where(mask, y_d, final)
        return final