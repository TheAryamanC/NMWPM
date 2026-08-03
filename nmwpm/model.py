import numpy as np
import torch
import torch.nn as nn


def mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.Linear(output_dim, output_dim)
    )


def positional_encoding(x, dim):
    freq = 10000.0 ** (-torch.arange(dim // 2, dtype=torch.float32) * 2 / dim)
    scaled_coords = x[:, None] * freq[None, :]
    return torch.cat([torch.sin(scaled_coords), torch.cos(scaled_coords)], 1)


class TransformerConvLayer(nn.Module):
    """Single graph-transformer message-passing layer."""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim, self.num_heads = hidden_dim, num_heads
        d, K = hidden_dim, num_heads
        self.norm1 = nn.LayerNorm(d)
        self.query_weight = nn.Parameter(torch.empty(K, d, d))
        self.query_bias = nn.Parameter(torch.zeros(K, d))
        self.key_weight = nn.Parameter(torch.empty(K, d, d))
        self.key_bias = nn.Parameter(torch.zeros(K, d))
        self.value_weight = nn.Parameter(torch.empty(K, d, d))
        self.value_bias = nn.Parameter(torch.zeros(K, d))
        for w in (self.query_weight, self.key_weight, self.value_weight):
            nn.init.xavier_uniform_(w)
        self.lin_skip = nn.Linear(d, d)
        self.gate_projection = nn.Linear(3 * d, 1, bias=False)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d)
        )

    def convolutional_layer(self, hn, adj):
        proj = lambda w, b: torch.einsum("bnd,kde->bkne", hn, w) + b[None, :, None, :]
        q = proj(self.query_weight, self.query_bias)
        k = proj(self.key_weight, self.key_bias)
        v = proj(self.value_weight, self.value_bias)
        att = (q @ k.transpose(-1, -2)) / self.hidden_dim ** 0.5
        att = att.masked_fill(~adj[None, None], float("-inf")).softmax(-1)
        neighbor_message = (att @ v).mean(1)
        self_projection = self.lin_skip(hn)
        beta = torch.sigmoid(self.gate_projection(torch.cat(
            [neighbor_message, self_projection, neighbor_message - self_projection], -1)))
        return beta * self_projection + (1 - beta) * neighbor_message

    def forward(self, h, adj):
        h = self.convolutional_layer(self.norm1(h), adj) + h
        return self.ffn(self.norm2(h)) + h


class QWP(nn.Module):
    """Quantum Weight Predictor — the neural network at the core of NMWPM.

    Given a syndrome and a candidate set of defect-pair edges, this model
    outputs a scalar logit for each edge indicating how likely that edge is
    part of the true minimum-weight perfect matching.

    Parameters
    ----------
    code:
        A :class:`~nmwpm.codes.ToricCode` or
        :class:`~nmwpm.codes.RotatedSurfaceCode` instance.
    hidden_dim:
        Width of all hidden representations (default 128).
    gnn_layers:
        Number of graph-transformer message-passing layers (default 4).
    num_heads:
        Attention heads per GNN layer (default 4).
    enc_layers:
        Number of Transformer encoder layers applied to the edge tokens
        (default 2; set to 0 to disable).
    d_pe:
        Dimension of positional encodings (default 16).
    """

    def __init__(self, code, hidden_dim=128, gnn_layers=4, num_heads=4, enc_layers=2, d_pe=16):
        super().__init__()
        self.hidden_dim, sub_dim = hidden_dim, hidden_dim // 4
        num_graph_nodes = code.num_graph_nodes
        coords = torch.tensor(code.coords, dtype=torch.float32)
        stab_type = nn.functional.one_hot(torch.tensor(code.stabilizer_type), 2).float()
        radius = torch.tensor(code.rho, dtype=torch.float32)
        posenc = torch.cat([positional_encoding(coords[:, 0], d_pe // 2),
                            positional_encoding(coords[:, 1], d_pe // 2)], 1)
        if code.has_boundary:
            coords = torch.cat([coords, torch.zeros(1, 2)])
            stab_type = torch.cat([stab_type, torch.zeros(1, 2)])
            radius = torch.cat([radius, torch.zeros(1, radius.shape[1])])
            vpe = torch.cat([positional_encoding(-torch.ones(1), d_pe // 2)] * 2, 1)
            posenc = torch.cat([posenc, vpe])
        for name, t in [("node_coords", coords), ("node_type", stab_type),
                        ("node_radius", radius), ("node_posenc", posenc)]:
            self.register_buffer(name, t)
        self.register_buffer("adj", torch.tensor(code.gnn_adj))
        self.mlp_coords = mlp(2, sub_dim)
        self.mlp_radius = mlp(radius.shape[1], sub_dim)
        self.mlp_posenc = mlp(d_pe, sub_dim)
        self.lin_type = nn.Linear(2, sub_dim)
        self.stab_embedding = nn.Embedding(num_graph_nodes, hidden_dim)
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.proj_norm = nn.LayerNorm(hidden_dim)
        self.gnn = nn.ModuleList(
            TransformerConvLayer(hidden_dim, num_heads) for _ in range(gnn_layers))
        max_dist = int(max(
            code.pair_distance.max(),
            code.boundary_distance.max() if code.has_boundary else 0))
        self.dist_emb = nn.Embedding(max_dist + 1, hidden_dim)
        self.edge_geo = nn.Sequential(
            nn.Linear(3, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2))
        dm = 2 * hidden_dim + 3 * hidden_dim // 2
        self.tok_norm = nn.LayerNorm(dm)
        self.encoder = (None if enc_layers == 0 else
                        nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(dm, num_heads,
                                                       dim_feedforward=dm,
                                                       batch_first=True),
                            enc_layers, enable_nested_tensor=False))
        self.out = nn.Linear(dm, 1)

    def forward(self, syndrome, edge_index, edge_distance_id, edge_geometry, edge_mask):
        B, act = syndrome.shape[0], syndrome.float().unsqueeze(-1)
        a = torch.cat([
            self.mlp_coords(self.node_coords * act),
            self.lin_type(self.node_type * act),
            self.mlp_radius(self.node_radius * act),
            self.mlp_posenc(self.node_posenc.expand(B, -1, -1)),
            self.stab_embedding.weight.expand(B, -1, -1)], -1)
        h = self.proj_norm(self.proj((2.0 * syndrome.float() - 1.0).unsqueeze(-1) * a))
        for layer in self.gnn:
            h = layer(h, self.adj)
        gather = lambda ix: h.gather(1, ix.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        e = torch.cat([self.dist_emb(edge_distance_id), self.edge_geo(edge_geometry)], -1)
        u = self.tok_norm(torch.cat([gather(edge_index[:, :, 0]),
                                      gather(edge_index[:, :, 1]), e], -1))
        o = u if self.encoder is None else self.encoder(u, src_key_padding_mask=~edge_mask)
        return self.out(o).squeeze(-1)


def build_batch(code, syndromes, edge_lists, device):
    """Pack a list of syndromes and their edge lists into padded tensors.

    Parameters
    ----------
    code:
        The code object (provides geometry look-up tables).
    syndromes:
        List of ``(num_stabilizers,)`` uint8 syndrome arrays.
    edge_lists:
        List of ``(E_i, 2)`` int64 directed edge arrays.
    device:
        Torch device string or object.

    Returns
    -------
    Tuple of five tensors ready to be passed to :class:`QWP`.
    """
    B, E = len(syndromes), max(len(e) for e in edge_lists)
    syndrome = np.ones((B, code.num_graph_nodes), np.int64)
    syndrome[:, :code.num_stabilizers] = np.asarray(syndromes)
    edge_index = np.zeros((B, E, 2), np.int64)
    edge_distance_id = np.zeros((B, E), np.int64)
    edge_geometry = np.zeros((B, E, 3), np.float32)
    edge_mask = np.zeros((B, E), bool)
    for b, el in enumerate(edge_lists):
        if len(el):
            edge_index[b, :len(el)] = el
            edge_distance_id[b, :len(el)], edge_geometry[b, :len(el)] = \
                code.compute_edge_features(el)
            edge_mask[b, :len(el)] = True
    t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=device)
    return (t(syndrome, torch.long), t(edge_index, torch.long),
            t(edge_distance_id, torch.long), t(edge_geometry, torch.float32),
            t(edge_mask, torch.bool))
