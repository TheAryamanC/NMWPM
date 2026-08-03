"""High-level decoder API for Neural Minimum-Weight Perfect Matching."""

from __future__ import annotations

import numpy as np
import pymatching
import torch

from .codes import ToricCode, RotatedSurfaceCode
from .model import QWP, build_batch


# ---------------------------------------------------------------------------
# Low-level decoding primitives (also used by the evaluate module)
# ---------------------------------------------------------------------------

def _decode_homology_parity(code, defects, weighted_edges):
    """Run pymatching on a weighted defect graph and return logical parity."""
    loc = {int(s): k for k, s in enumerate(defects)}
    m = pymatching.Matching()
    nl = code.num_logicals
    for i, j, w in weighted_edges:
        g = code.stabilizer_type[i]
        if j == code.boundary_index:
            f = {g * nl + l for l in range(nl) if code.boundary_crossings[i, l]}
            m.add_boundary_edge(loc[i], weight=w, fault_ids=f)
        else:
            a, b = min(i, j), max(i, j)
            f = {g * nl + l for l in range(nl) if code.pair_crossings[a, b, l]}
            m.add_edge(loc[i], loc[j], weight=w, fault_ids=f)
    out = m.decode(np.ones(len(defects), dtype=np.uint8))
    par = np.zeros(2 * nl, np.uint8)
    par[:len(out)] = out
    return par


def _canonical_weighted_pairs(code, edge_index, uniform=False):
    """Convert a directed edge list to canonical undirected weighted triples."""
    return [
        (i, j, 1.0 if uniform else (
            code.boundary_distance[i] if j == code.boundary_index
            else code.pair_distance[i, j]))
        for i, j in edge_index
        if (j == code.boundary_index and i < code.boundary_index) or i < j
    ]


def _aggregate_directed_edge_probs(code, edge_index, probs, agg="max"):
    """Aggregate directed edge probabilities into undirected log-weights."""
    base = code.boundary_index + 1
    key = (np.minimum(edge_index[:, 0], edge_index[:, 1]) * base
           + np.maximum(edge_index[:, 0], edge_index[:, 1]))
    uk, inv = np.unique(key, return_inverse=True)
    pu = np.full(len(uk), 0.0 if agg == "max" else 1.0)
    (np.maximum if agg == "max" else np.minimum).at(pu, inv, probs)
    w = -np.log(np.clip(pu, 1e-7, 1 - 1e-7))
    return [(int(k // base), int(k % base), w[t]) for t, k in enumerate(uk)]


# ---------------------------------------------------------------------------
# High-level Decoder class
# ---------------------------------------------------------------------------

class Decoder:
    """Neural Minimum-Weight Perfect Matching decoder.

    Wraps a :class:`~nmwpm.model.QWP` model and a CSS code object to provide a
    simple :meth:`decode` / :meth:`decode_batch` interface.

    Parameters
    ----------
    code:
        A :class:`~nmwpm.codes.ToricCode` or
        :class:`~nmwpm.codes.RotatedSurfaceCode` instance.
    model:
        A trained :class:`~nmwpm.model.QWP` network.  If ``None`` the decoder
        falls back to standard distance-weighted MWPM.
    device:
        Torch device (e.g. ``"cpu"``, ``"cuda"``).  Defaults to CUDA when
        available.
    agg:
        Edge-probability aggregation strategy for directed → undirected
        conversion: ``"max"`` (default) or ``"min"``.
    fast_path:
        Syndromes with at most this many active stabilizers skip the neural
        network and use plain MWPM (useful to cut latency for quiet syndromes).
    """

    def __init__(self, code, model=None, device=None, agg="max", fast_path=0):
        self.code = code
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.agg = agg
        self.fast_path = fast_path

        if self.model is not None:
            self.model = self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: str, device: str | None = None,
                        agg: str = "max", fast_path: int = 0) -> "Decoder":
        """Load a pretrained NMWPM decoder from a checkpoint file.

        Parameters
        ----------
        path:
            Path to a ``.pt`` checkpoint saved by the training script.
        device:
            Override the target device.  Defaults to CUDA when available.
        agg:
            Edge aggregation strategy (``"max"`` or ``"min"``).
        fast_path:
            Syndrome defect-count threshold below which the neural network is
            bypassed (default 0 = always use the network).

        Returns
        -------
        Decoder
            A ready-to-use :class:`Decoder` instance.

        Examples
        --------
        >>> decoder = Decoder.from_checkpoint("checkpoints/nmwpm_toric_L8_depolarizing.pt")
        >>> parity = decoder.decode(syndrome)
        """
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=resolved_device, weights_only=True)
        cfg = ck["config"]
        code = (ToricCode(cfg["L"]) if cfg["code"] == "toric"
                else RotatedSurfaceCode(cfg["L"]))
        model = QWP(
            code,
            hidden_dim=cfg.get("hidden_dim", 128),
            gnn_layers=cfg.get("gnn_layers", 4),
            num_heads=cfg.get("num_heads", 4),
            enc_layers=cfg.get("enc_layers", 2),
        ).to(resolved_device).eval()
        model.load_state_dict(ck["model"])
        return cls(code, model, resolved_device, agg=agg, fast_path=fast_path)

    @classmethod
    def mwpm(cls, code, device: str | None = None) -> "Decoder":
        """Create a plain distance-weighted MWPM decoder (no neural network).

        Parameters
        ----------
        code:
            A :class:`~nmwpm.codes.ToricCode` or
            :class:`~nmwpm.codes.RotatedSurfaceCode` instance.

        Returns
        -------
        Decoder
            A :class:`Decoder` with ``model=None``.
        """
        return cls(code, model=None, device=device)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a single syndrome measurement.

        Parameters
        ----------
        syndrome:
            1-D binary array of length ``code.num_stabilizers`` where ``1``
            marks a triggered (flipped) stabilizer.

        Returns
        -------
        np.ndarray
            Binary array of length ``2 * code.num_logicals`` giving the
            predicted logical error parity.  Index ``l`` is the X-type logical
            error on logical qubit ``l``; index ``num_logicals + l`` is the
            Z-type logical error on logical qubit ``l``.
        """
        return self.decode_batch(np.asarray(syndrome, dtype=np.uint8)[np.newaxis])[0]

    @torch.no_grad()
    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a batch of syndrome measurements.

        Parameters
        ----------
        syndromes:
            2-D binary array of shape ``(batch, code.num_stabilizers)``.

        Returns
        -------
        np.ndarray
            Binary array of shape ``(batch, 2 * code.num_logicals)``.
        """
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        B = syndromes.shape[0]
        nl = self.code.num_logicals
        results = np.zeros((B, 2 * nl), np.uint8)

        edge_lists = [self.code.build_syndrome_graph(syndromes[k]) for k in range(B)]

        # Partition samples: fast-path (no NN) vs. full NN pass
        fast = [syndromes[k].sum() <= self.fast_path for k in range(B)]
        nz = [k for k in range(B) if len(edge_lists[k]) > 0 and not fast[k]]

        probs: dict[int, np.ndarray] = {}
        if self.model is not None and nz:
            tensors = build_batch(
                self.code,
                [syndromes[k] for k in nz],
                [edge_lists[k] for k in nz],
                self.device,
            )
            logits = self.model(*tensors)
            for r, k in enumerate(nz):
                probs[k] = torch.sigmoid(logits[r, :len(edge_lists[k])]).cpu().numpy()

        for k in range(B):
            el = edge_lists[k]
            if len(el) == 0:
                continue  # no defects → no logical error predicted
            defects = np.flatnonzero(syndromes[k])
            if self.model is not None and not fast[k]:
                weighted = _aggregate_directed_edge_probs(
                    self.code, el, probs[k], self.agg)
            else:
                weighted = _canonical_weighted_pairs(self.code, el)
            results[k] = _decode_homology_parity(self.code, defects, weighted)

        return results

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def code_name(self) -> str:
        """Human-readable code identifier."""
        if isinstance(self.code, ToricCode):
            return f"toric L={self.code.L}"
        if isinstance(self.code, RotatedSurfaceCode):
            return f"rotated L={self.code.L}"
        return type(self.code).__name__

    def __repr__(self) -> str:
        nn_info = (f"QWP hidden_dim={self.model.hidden_dim}"
                   if self.model is not None else "MWPM-only")
        return f"Decoder({self.code_name}, {nn_info}, device={self.device!r})"
