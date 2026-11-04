"""
NBFNetFeat: BioPathNet's NBFNet + frozen node features (added for the KEGG KO strict benchmark; not part of the
original BioPathNet release). Inference version: Bellman-Ford message passing is the unchanged official code; the
released checkpoints were trained with BioPathNet's official training loop.

forward() keeps the upstream TorchDrug model interface (graph, h_index, t_index, r_index, all_loss, metric). The
`all_loss` / `metric` arguments are passed by the upstream task only in its training step; this release does not
include training, so a call with all_loss set raises instead of applying the training-time edge removal.

Changes relative to model.NBFNet
  x_v = LayerNorm+Linear(frozen feature_v) (features aligned to the dataset's entity order by the runner; zero if missing)
  boundary condition at the head:  query(r) + W_b x_h        (official: query(r))
  score(h, r, t) = MLP([path representation of t, query(r)]) + MLP_x([x_t, query(r)]) + <W_r(query) * x_h, x_t>
The two extra terms let the model score tails that are unreachable from h in the train graph (e.g. KOs that never
occur in train), which the path term alone cannot rank.
"""

import torch
from torch import nn
from torchdrug import layers
from torchdrug.core import Registry as R

from .model import NeuralBellmanFordNetwork


@R.register("model.NBFNetFeat")
class NBFNetFeat(NeuralBellmanFordNetwork):

    def __init__(self, input_dim, hidden_dims, feature_file=None, num_relation=None, feat_layers=False,
                 feat_dropout=0.0, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation=num_relation, **kwargs)
        # feat_layers: also add a (layer-specific) projection of x_v to every node state before every layer, so the
        # path representation can use the features of the intermediate KOs, not only of head and tail
        self.feat_layers = feat_layers
        self.feat_drop = nn.Dropout(feat_dropout)
        X = torch.load(feature_file)
        self.register_buffer("feat", X["X"].float())
        self.register_buffer("has", X["has"].float())
        D = self.feat.shape[1]
        self.proj = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, input_dim))
        self.boundary_proj = nn.Linear(input_dim, input_dim)
        self.rel_bl = nn.Linear(input_dim, input_dim)
        self.mlp_x = layers.MLP(2 * input_dim, [2 * input_dim, 1])
        if feat_layers:
            self.layer_feat = nn.ModuleList([nn.Linear(input_dim, self.dims[i]) for i in range(len(self.dims) - 1)])

    def node_x(self):
        return self.feat_drop(self.proj(self.feat)) * self.has[:, None]

    def bellmanford(self, graph, h_index, r_index, separate_grad=False):
        # identical to the parent, except for the feature term in the boundary condition
        query = self.query(r_index)
        x = self.node_x()
        init = query + self.boundary_proj(x[h_index])
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(graph.num_node, *query.shape, device=self.device)
        boundary.scatter_add_(0, index.unsqueeze(0), init.unsqueeze(0))
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary
        hiddens, layer_input = [], boundary
        for li, layer in enumerate(self.layers):
            if self.feat_layers:
                layer_input = layer_input + self.layer_feat[li](x)[:, None, :]
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden
        node_query = query.expand(graph.num_node, -1, -1)
        output = torch.cat(hiddens + [node_query], dim=-1) if self.concat_hidden else \
            torch.cat([hiddens[-1], node_query], dim=-1)
        return {"node_feature": output, "step_graphs": []}

    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None):
        if all_loss is not None:
            raise NotImplementedError("NBFNetFeat in this release is inference-only (called with all_loss set)")
        shape = h_index.shape
        graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        output = self.bellmanford(graph, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"].transpose(0, 1)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)
        score = self.mlp(feature).squeeze(-1)
        x = self.node_x()
        query = self.query(r_index[:, 0])                                   # [B, d]
        xt = x[t_index]                                                      # [B, N, d]
        q = query.unsqueeze(1).expand_as(xt)
        score = score + self.mlp_x(torch.cat([xt, q], -1)).squeeze(-1)
        xh = x[h_index[:, 0]] * self.rel_bl(query)                           # [B, d]
        score = score + (xt * xh.unsqueeze(1)).sum(-1)
        return score.view(shape)
