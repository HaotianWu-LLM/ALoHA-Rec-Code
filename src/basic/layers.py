import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation import activation_layer
from .features import DenseFeature, SparseFeature, SequenceFeature


class PredictionLayer(nn.Module):
    """Prediction Layer.

    Args:
        task_type (str): if `task_type='classification'`, then return sigmoid(x),
                    change the input logits to probability. if`task_type='regression'`, then return x.
    """

    def __init__(self, task_type='classification'):
        super(PredictionLayer, self).__init__()
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be classification or regression")
        self.task_type = task_type

    def forward(self, x):
        if self.task_type == "classification":
            x = torch.sigmoid(x)
        return x


class EmbeddingLayer(nn.Module):
    """General Embedding Layer.
    We save all the feature embeddings in embed_dict: `{feature_name : embedding table}`.


    Args:
        features (list): the list of `Feature Class`. It means all the features which we want to create an embedding table.

    Shape:
        - Input:
            x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:`(batch_size, seq_len)`,\
                      sparse/dense feature value is a 1D tensor with shape `(batch_size)`.
            features (list): the list of `Feature Class`. It is means the current features which we want to do embedding lookup.
            squeeze_dim (bool): whether to squeeze dim of output (default = `False`).
        - Output:
            - if input Dense: `(batch_size, num_features_dense)`.
            - if input Sparse: `(batch_size, num_features, embed_dim)` or `(batch_size, num_features * embed_dim)`.
            - if input Sequence: same with input sparse or `(batch_size, num_features_seq, seq_length, embed_dim)` when `pooling=="concat"`.
            - if input Dense and Sparse/Sequence: `(batch_size, num_features_sparse * embed_dim)`. Note we must squeeze_dim for concat dense value with sparse embedding.
    """

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.n_dense = 0

        for fea in features:
            if fea.name in self.embed_dict:
                continue
            if isinstance(fea, SparseFeature) and fea.shared_with == None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, SequenceFeature) and fea.shared_with == None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, DenseFeature):
                self.n_dense += 1

    def forward(self, x, features, squeeze_dim=False):
        sparse_emb, dense_values = [], []
        sparse_exists, dense_exists = False, False
        for fea in features:
            if isinstance(fea, SparseFeature):
                if fea.shared_with == None:
                    sparse_emb.append(self.embed_dict[fea.name](x[fea.name].long()).unsqueeze(1))
                else:
                    sparse_emb.append(self.embed_dict[fea.shared_with](x[fea.name].long()).unsqueeze(1))
            elif isinstance(fea, SequenceFeature):
                if fea.pooling == "sum":
                    pooling_layer = SumPooling()
                elif fea.pooling == "mean":
                    pooling_layer = AveragePooling()
                elif fea.pooling == "concat":
                    pooling_layer = ConcatPooling()
                else:
                    raise ValueError("Sequence pooling method supports only pooling in %s, got %s." %
                                     (["sum", "mean"], fea.pooling))
                fea_mask = InputMask()(x, fea)
                if fea.shared_with == None:
                    sparse_emb.append(
                        pooling_layer(self.embed_dict[fea.name](x[fea.name].long()), fea_mask).unsqueeze(1))
                else:
                    sparse_emb.append(
                        pooling_layer(self.embed_dict[fea.shared_with](x[fea.name].long()), fea_mask).unsqueeze(
                            1))  # shared specific sparse feature embedding
            else:
                dense_values.append(x[fea.name].float().unsqueeze(1))  # .unsqueeze(1).unsqueeze(1)

        if len(dense_values) > 0:
            dense_exists = True
            dense_values = torch.cat(dense_values, dim=1)
        if len(sparse_emb) > 0:
            sparse_exists = True
            sparse_emb = torch.cat(sparse_emb, dim=1)  # [batch_size, num_features, embed_dim]

        if squeeze_dim:  # Note: if the emb_dim of sparse features is different, we must squeeze_dim
            if dense_exists and not sparse_exists:  # only input dense features
                return dense_values
            elif not dense_exists and sparse_exists:
                return sparse_emb.flatten(start_dim=1)  # squeeze dim to : [batch_size, num_features*embed_dim]
            elif dense_exists and sparse_exists:
                return torch.cat((sparse_emb.flatten(start_dim=1), dense_values),
                                 dim=1)  # concat dense value with sparse embedding
            else:
                raise ValueError("The input features can note be empty")
        else:
            if sparse_exists:
                return sparse_emb  # [batch_size, num_features, embed_dim]
            else:
                raise ValueError(
                    "If keep the original shape:[batch_size, num_features, embed_dim], expected %s in feature list, got %s" %
                    ("SparseFeatures", features))


class InputMask(nn.Module):
    """Return inputs mask from given features

    Shape:
        - Input:
            x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:`(batch_size, seq_len)`,\
                      sparse/dense feature value is a 1D tensor with shape `(batch_size)`.
            features (list or SparseFeature or SequenceFeature): Note that the elements in features are either all instances of SparseFeature or all instances of SequenceFeature.
        - Output:
            - if input Sparse: `(batch_size, num_features)`
            - if input Sequence: `(batch_size, num_features_seq, seq_length)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, features):
        mask = []
        if not isinstance(features, list):
            features = [features]
        for fea in features:
            if isinstance(fea, SparseFeature) or isinstance(fea, SequenceFeature):
                if fea.padding_idx != None:
                    fea_mask = x[fea.name].long() != fea.padding_idx
                else:
                    fea_mask = x[fea.name].long() != -1
                mask.append(fea_mask.unsqueeze(1).float())
            else:
                raise ValueError("Only SparseFeature or SequenceFeature support to get mask.")
        return torch.cat(mask, dim=1)


class LR(nn.Module):
    """Logistic Regression Module. It is the one Non-linear
    transformation for input feature.

    Args:
        input_dim (int): input size of Linear module.
        sigmoid (bool): whether to add sigmoid function before output.

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)`
    """

    def __init__(self, input_dim, sigmoid=False):
        super().__init__()
        self.sigmoid = sigmoid
        self.fc = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        if self.sigmoid:
            return torch.sigmoid(self.fc(x))
        else:
            return self.fc(x)


class ConcatPooling(nn.Module):
    """Keep the origin sequence embedding shape

    Shape:
    - Input: `(batch_size, seq_length, embed_dim)`
    - Output: `(batch_size, seq_length, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x


class AveragePooling(nn.Module):
    """Pooling the sequence embedding matrix by `mean`.

    Shape:
        - Input
            x: `(batch_size, seq_length, embed_dim)`
            mask: `(batch_size, 1, seq_length)`
        - Output: `(batch_size, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask == None:
            return torch.mean(x, dim=1)
        else:
            sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
            non_padding_length = mask.sum(dim=-1)
            return sum_pooling_matrix / (non_padding_length.float() + 1e-16)


class SumPooling(nn.Module):
    """Pooling the sequence embedding matrix by `sum`.

    Shape:
        - Input
            x: `(batch_size, seq_length, embed_dim)`
            mask: `(batch_size, 1, seq_length)`
        - Output: `(batch_size, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask == None:
            return torch.sum(x, dim=1)
        else:
            return torch.bmm(mask, x).squeeze(1)


class MLP(nn.Module):
    """Multi-Layer Perceptron Module, it is the most widely used module for
    learning feature. Note we default add `BatchNorm1d` and `Activation`
    `Dropout` for each `Linear` Module.

    Args:
        input dim (int): input size of the first Linear Layer.
        output_layer (bool): whether this MLP module is the output layer. If `True`, then append one Linear(*,1) module.
        dims (list): output size of Linear Layer (default=[]).
        dropout (float): probability of an element to be zeroed (default = 0.5).
        activation (str): the activation function, support `[sigmoid, relu, prelu, dice, softmax]` (default='relu').

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)` or `(batch_size, dims[-1])`
    """

    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0, activation="relu"):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Pruner(nn.Module):
    """The Domain-adaptive Pruner， mentioned in the `AdaSparse paper
    <https://dl.acm.org/doi/pdf/10.1145/3511808.3557541>`. It is used
    to generate pruners for mlp layers

    Args:

    Shape:
    """

    def __init__(self, sce_dims, agn_dims, form='Binarization', epsilon=1e-2, beta=2.0):
        super().__init__()
        if form not in ['Binarization', 'Scaling', 'Fusion']:
            raise ValueError("The input 'form' must be one of ['Binarization', 'Scaling', 'Fusion']")
        self.form = form
        self.epsilon = torch.tensor(epsilon)
        self.beta = torch.tensor(beta)

        self.linear = nn.Linear(sce_dims + agn_dims, agn_dims, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sce_features, agn_features, alpha=torch.tensor(1.0)):
        x = torch.cat((sce_features, agn_features), 1)
        vin = self.linear(x)
        if self.form == 'Binarization':
            vout = self.sigmoid(vin * alpha)
            s = torch.sign(vout - self.epsilon)
            return s
        if self.form == 'Scaling':
            vout = self.beta * self.sigmoid(vin)
            s = vout * torch.sign(vout - self.epsilon)
            return s
        if self.form == 'Fusion':
            vout = self.beta * self.sigmoid(vin * alpha)
            s = vout * torch.sign(vout - self.epsilon)
            return s


class GateNU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, gemma=2.0):
        super(GateNU, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        self.gemma = gemma
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, output_dim),
                                     nn.Sigmoid())

    def forward(self, inputs):
        out_layer = self.network(inputs)
        return out_layer * self.gemma


# =============================================================================
# Non-FCN feature extractors used by the ALoHA-Rec Framework wrappers when
# extractor != 'fcn'. These are NEW classes and do NOT modify any pre-existing
# class above.
#
# Each extractor exposes:
#   * feature_dim      : output dim fed to the downstream head
#   * backbone_layers  : list of nn.Linear eligible for LoRA injection & GABA
#   * forward(x_flat, emb_3d=None) -> (features, aux_logit)
# =============================================================================


class DeepFMExtractor(nn.Module):
    """DeepFM extractor: FM first-order + second-order (aux logit) +
    Deep Network (feature trunk). Deep Linear layers are exposed for LoRA.
    """

    def __init__(self, input_dim, deep_dims=None, dropout=0.2,
                 num_fields=None, embed_dim=None, **kwargs):
        super().__init__()
        if deep_dims is None:
            deep_dims = [256, 128, 64]
        self.input_dim = input_dim
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self._feature_dim = deep_dims[-1]

        self.fm_first_order = nn.Linear(input_dim, 1)

        dims = [input_dim] + list(deep_dims)
        self.deep_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.deep_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.bn_layers.append(nn.BatchNorm1d(dims[i + 1]))
            self.dropout_layers.append(nn.Dropout(dropout))

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def backbone_layers(self):
        return list(self.deep_layers)

    def forward(self, x, emb_3d=None):
        fm_first = self.fm_first_order(x).squeeze(-1)
        if emb_3d is not None:
            sum_emb = emb_3d.sum(dim=1)
            square_sum = sum_emb ** 2
            sum_square = (emb_3d ** 2).sum(dim=1)
            fm_second = 0.5 * (square_sum - sum_square).sum(dim=1)
            aux = fm_first + fm_second
        else:
            aux = fm_first

        h = x
        for i in range(len(self.deep_layers)):
            h = self.deep_layers[i](h)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout_layers[i](h)
        return h, aux


class DCNExtractor(nn.Module):
    """DCN extractor: Cross Network (side branch) + Deep Network.
    Output is concat(cross_last, deep_last). Deep Linear layers are exposed
    for LoRA; cross layers are not routed (they are a side branch).
    """

    def __init__(self, input_dim, num_cross_layers=3, deep_dims=None, dropout=0.2, **kwargs):
        super().__init__()
        if deep_dims is None:
            deep_dims = [256, 128, 64]
        self.input_dim = input_dim
        self.num_cross_layers = int(num_cross_layers)
        self._feature_dim = input_dim + deep_dims[-1]

        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, 1)) for _ in range(self.num_cross_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(self.num_cross_layers)
        ])
        for w in self.cross_weights:
            nn.init.xavier_uniform_(w)

        dims = [input_dim] + list(deep_dims)
        self.deep_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.deep_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.bn_layers.append(nn.BatchNorm1d(dims[i + 1]))
            self.dropout_layers.append(nn.Dropout(dropout))

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def backbone_layers(self):
        return list(self.deep_layers)

    def forward(self, x, emb_3d=None):
        x0 = x
        x_cross = x
        for i in range(self.num_cross_layers):
            xw = torch.matmul(x_cross, self.cross_weights[i])
            x_cross = x0 * xw + self.cross_biases[i] + x_cross

        h = x
        for i in range(len(self.deep_layers)):
            h = self.deep_layers[i](h)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout_layers[i](h)

        features = torch.cat([x_cross, h], dim=1)
        aux = torch.zeros(features.shape[0], device=features.device)
        return features, aux


class _CIN(nn.Module):
    """Compressed Interaction Network used by xDeepFMExtractor."""

    def __init__(self, num_fields, cin_layer_sizes, split_half=True):
        super().__init__()
        self.num_fields = num_fields
        self.cin_layer_sizes = cin_layer_sizes
        self.split_half = split_half

        self.conv_layers = nn.ModuleList()
        prev = num_fields
        for i, layer_size in enumerate(cin_layer_sizes):
            self.conv_layers.append(nn.Conv1d(num_fields * prev, layer_size, kernel_size=1))
            if split_half and i < len(cin_layer_sizes) - 1:
                prev = layer_size // 2
            else:
                prev = layer_size

    def forward(self, x):
        B, _, D = x.shape
        hidden = [x]
        outs = []
        for i, conv in enumerate(self.conv_layers):
            x0 = hidden[0]
            xi = hidden[-1]
            outer = torch.einsum('bmd,bnd->bmnd', x0, xi).view(B, -1, D)
            out = F.relu(conv(outer))
            if self.split_half and i < len(self.cin_layer_sizes) - 1:
                out, direct = torch.split(out, out.shape[1] // 2, dim=1)
                outs.append(direct)
            else:
                outs.append(out)
            hidden.append(out)
        return torch.sum(torch.cat(outs, dim=1), dim=2)


class xDeepFMExtractor(nn.Module):
    """xDeepFM extractor: Linear term + CIN (aux logit) + Deep Network.
    Deep Linear layers are exposed for LoRA.
    """

    def __init__(self, input_dim, cin_layer_sizes=None, deep_dims=None,
                 dropout=0.2, num_fields=None, embed_dim=None, **kwargs):
        super().__init__()
        if cin_layer_sizes is None:
            cin_layer_sizes = [128, 128]
        if deep_dims is None:
            deep_dims = [256, 128, 64]
        self.input_dim = input_dim
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self._feature_dim = deep_dims[-1]

        self.linear = nn.Linear(input_dim, 1)

        if num_fields is not None:
            self.cin = _CIN(num_fields, cin_layer_sizes, split_half=True)
            cin_out_dim = sum(s // 2 for s in cin_layer_sizes[:-1]) + cin_layer_sizes[-1]
            self.cin_output = nn.Linear(cin_out_dim, 1)
        else:
            self.cin = None
            self.cin_output = None

        dims = [input_dim] + list(deep_dims)
        self.deep_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.deep_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.bn_layers.append(nn.BatchNorm1d(dims[i + 1]))
            self.dropout_layers.append(nn.Dropout(dropout))

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def backbone_layers(self):
        return list(self.deep_layers)

    def forward(self, x, emb_3d=None):
        aux = self.linear(x).squeeze(-1)
        if self.cin is not None and emb_3d is not None:
            aux = aux + self.cin_output(self.cin(emb_3d)).squeeze(-1)

        h = x
        for i in range(len(self.deep_layers)):
            h = self.deep_layers[i](h)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout_layers[i](h)
        return h, aux


class AutoIntExtractor(nn.Module):
    """AutoInt extractor: stacked multi-head self-attention over per-field
    embeddings + Deep Network. Deep Linear layers are exposed for LoRA.
    Attention runs only when emb_3d (per-field embedding) is provided.
    """

    def __init__(self, input_dim, deep_dims=None, dropout=0.2,
                 num_heads=2, num_attention_layers=2,
                 num_fields=None, embed_dim=None, **kwargs):
        super().__init__()
        if deep_dims is None:
            deep_dims = [256, 128, 64]
        self.input_dim = input_dim
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.num_attention_layers = int(num_attention_layers)

        if num_fields is not None and embed_dim is not None:
            self.attention_dim = (embed_dim // num_heads) * num_heads
            if self.attention_dim != embed_dim:
                self.proj = nn.Linear(embed_dim, self.attention_dim)
            else:
                self.proj = None
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=self.attention_dim, num_heads=num_heads,
                                      dropout=dropout, batch_first=True)
                for _ in range(self.num_attention_layers)
            ])
            self.attention_norms = nn.ModuleList([
                nn.LayerNorm(self.attention_dim) for _ in range(self.num_attention_layers)
            ])
            attn_out_dim = num_fields * self.attention_dim
        else:
            self.proj = None
            self.attention_layers = None
            self.attention_norms = None
            attn_out_dim = input_dim

        dims = [attn_out_dim] + list(deep_dims)
        self.deep_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.deep_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.bn_layers.append(nn.BatchNorm1d(dims[i + 1]))
            self.dropout_layers.append(nn.Dropout(dropout))

        self._feature_dim = deep_dims[-1]

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def backbone_layers(self):
        return list(self.deep_layers)

    def forward(self, x, emb_3d=None):
        if self.attention_layers is not None and emb_3d is not None:
            a = emb_3d if self.proj is None else self.proj(emb_3d)
            for i in range(self.num_attention_layers):
                residual = a
                a, _ = self.attention_layers[i](a, a, a)
                a = self.attention_norms[i](a + residual)
            h = a.flatten(start_dim=1)
        else:
            h = x

        for i in range(len(self.deep_layers)):
            h = self.deep_layers[i](h)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout_layers[i](h)
        aux = torch.zeros(h.shape[0], device=h.device)
        return h, aux


def build_extractor(name, input_dim, num_fields=None, embed_dim=None,
                    deep_dims=None, dropout=0.2, **kwargs):
    """Factory for non-FCN extractors.

    Returns an extractor whose `forward(x_flat, emb_3d)` yields
    `(features, aux_logit)`.
    """
    name = name.lower()
    if name == 'deepfm':
        return DeepFMExtractor(input_dim, deep_dims=deep_dims, dropout=dropout,
                               num_fields=num_fields, embed_dim=embed_dim, **kwargs)
    if name == 'dcn':
        return DCNExtractor(input_dim, deep_dims=deep_dims, dropout=dropout, **kwargs)
    if name == 'xdeepfm':
        return xDeepFMExtractor(input_dim, deep_dims=deep_dims, dropout=dropout,
                                num_fields=num_fields, embed_dim=embed_dim, **kwargs)
    if name == 'autoint':
        return AutoIntExtractor(input_dim, deep_dims=deep_dims, dropout=dropout,
                                num_fields=num_fields, embed_dim=embed_dim, **kwargs)
    raise ValueError(f"Unknown extractor name: {name}")