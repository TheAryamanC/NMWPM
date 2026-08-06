import time
from collections import deque

import numpy as np
import stim


class CSSCode:
    """Base class for Calderbank-Shor-Steane (CSS) quantum error-correcting codes.

    CSS codes separate X-type and Z-type error correction.  Subclasses must
    populate ``supports``, ``stabilizer_type``, ``coords``, ``rho``, and
    ``logical_supports`` before calling :meth:`build_structures`.
    """

    has_boundary = False

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def build_check_matrix(self):
        N, n = len(self.supports), self.n
        check_matrix = np.zeros((N, n), dtype=np.uint8)
        for s, qs in enumerate(self.supports):
            check_matrix[s, qs] = 1
        self.check_matrix = check_matrix

    def build_adjacency_matrix(self):
        N = len(self.supports)
        self.num_stabilizers = self.boundary_index = N
        self.num_logicals = self.logical_supports.shape[1]
        self.stabilizer_type = np.asarray(self.stabilizer_type, dtype=np.int64)

        self.num_graph_nodes = N + (1 if self.has_boundary else 0)
        adj = np.zeros((self.num_graph_nodes, self.num_graph_nodes), dtype=bool)

        share = (self.check_matrix.astype(np.int64) @ self.check_matrix.T.astype(np.int64)) > 0
        np.fill_diagonal(share, False)
        adj[:N, :N] = share

        if self.has_boundary:
            adj[N, :N] = adj[:N, N] = self.boundary_distance == 1
        self.gnn_adj = adj

    def build_stim_circuit(self):
        n = self.n
        self.qubit_to_stabilizers_by_type = [[[] for _ in range(n)], [[] for _ in range(n)]]
        for s, qs in enumerate(self.supports):
            for q in qs:
                self.qubit_to_stabilizers_by_type[self.stabilizer_type[s]][q].append(s)

        lines = []
        for s, qs in enumerate(self.supports):
            pauli = "X" if self.stabilizer_type[s] == 0 else "Z"
            lines.append("MPP " + "*".join(f"{pauli}{q}" for q in qs))
        self.circuit = stim.Circuit("\n".join(lines))

    def build_structures(self):
        self.build_check_matrix()
        self.build_adjacency_matrix()
        self.build_stim_circuit()

    # ------------------------------------------------------------------
    # Error sampling
    # ------------------------------------------------------------------

    #: All noise model identifiers accepted by :meth:`sample`.
    NOISE_MODELS = (
        "depolarizing", "independent",
        "biased", "biased_x",
        "x_only", "z_only", "y_only",
        "pauli",
    )

    def sample(self, shots, p, noise="depolarizing", seed=None, eta=10.0,
               px_frac=1 / 3, py_frac=1 / 3, pz_frac=1 / 3):
        """Sample physical errors and syndromes.

        Parameters
        ----------
        shots:
            Number of Monte-Carlo samples.
        p:
            Physical error rate (total probability of any Pauli error per qubit).
        noise:
            Noise model — one of ``CSSCode.NOISE_MODELS``:

            ``"depolarizing"``
                Equal X/Y/Z via Stim ``DEPOLARIZE1``.
            ``"independent"``
                Independent X and Z errors each at rate *p*.
            ``"biased"``
                Z-biased Pauli channel.  ``eta`` sets the Z/X ratio:
                pz = p·η/(1+η), px = py = p/(2(1+η)).
            ``"biased_x"``
                X-biased Pauli channel (mirrors ``"biased"``).
            ``"x_only"``
                Pure X (bit-flip) errors at rate *p*.
            ``"z_only"``
                Pure Z (phase-flip) errors at rate *p*.
            ``"y_only"``
                Pure Y errors at rate *p* (triggers both X and Z syndromes).
            ``"pauli"``
                General Pauli channel via ``PAULI_CHANNEL_1``.  Set
                ``px_frac``, ``py_frac``, ``pz_frac`` (they are normalised
                automatically); default is equal thirds (= depolarizing).
        seed:
            Optional integer seed for the Stim simulator.
        eta:
            Bias ratio for ``"biased"`` / ``"biased_x"`` noise (default 10).
        px_frac, py_frac, pz_frac:
            Relative X/Y/Z weights for ``"pauli"`` noise (default 1/3 each).

        Returns
        -------
        error_x, error_z, syndrome:
            Arrays of shape ``(shots, n)``, ``(shots, n)``, and
            ``(shots, num_stabilizers)``, all ``uint8``.
        """
        circuit = stim.Circuit()
        if noise == "depolarizing":
            circuit.append("DEPOLARIZE1", range(self.n), p)
        elif noise == "biased":
            px = py = p / (2.0 * (1.0 + eta))
            pz = p * eta / (1.0 + eta)
            circuit.append("PAULI_CHANNEL_1", range(self.n), [px, py, pz])
        elif noise == "biased_x":
            pz = py = p / (2.0 * (1.0 + eta))
            px = p * eta / (1.0 + eta)
            circuit.append("PAULI_CHANNEL_1", range(self.n), [px, py, pz])
        elif noise == "x_only":
            circuit.append("X_ERROR", range(self.n), p)
        elif noise == "z_only":
            circuit.append("Z_ERROR", range(self.n), p)
        elif noise == "y_only":
            circuit.append("Y_ERROR", range(self.n), p)
        elif noise == "pauli":
            total = px_frac + py_frac + pz_frac
            circuit.append("PAULI_CHANNEL_1", range(self.n),
                           [p * px_frac / total, p * py_frac / total, p * pz_frac / total])
        else:  # independent
            circuit.append("X_ERROR", range(self.n), p)
            circuit.append("Z_ERROR", range(self.n), p)

        sim = stim.FlipSimulator(batch_size=shots, num_qubits=self.n,
                                 disable_stabilizer_randomization=True, seed=seed)
        sim.do(circuit)
        flips = sim.peek_pauli_flips()
        sim.do(self.circuit)

        syndrome = sim.get_measurement_flips().T.astype(np.uint8)
        error_x = np.zeros((shots, self.n), np.uint8)
        error_z = np.zeros((shots, self.n), np.uint8)
        for k, ps in enumerate(flips):
            xs, zs = ps.to_numpy()
            error_x[k], error_z[k] = xs, zs
        return error_x, error_z, syndrome

    def error_parities(self, ex, ez):
        """Return logical error parities (shape ``(shots, 2*num_logicals)``)."""
        logical_x_flips = ez.astype(np.int64) @ self.logical_supports[0].T.astype(np.int64) % 2
        logical_z_flips = ex.astype(np.int64) @ self.logical_supports[1].T.astype(np.int64) % 2
        return np.concatenate([logical_x_flips, logical_z_flips], axis=1).astype(np.uint8)

    # ------------------------------------------------------------------
    # Syndrome graph construction
    # ------------------------------------------------------------------

    def build_syndrome_graph(self, syndrome_row):
        """Return directed edge array ``(E, 2)`` for all defect pairs."""
        out = []
        for stabilizer_type in (0, 1):
            active = np.flatnonzero(
                (syndrome_row == 1) & (self.stabilizer_type == stabilizer_type))
            if active.size >= 2:
                src, tgt = np.meshgrid(active, active, indexing="ij")
                mask = src != tgt
                out.append(np.stack([src[mask], tgt[mask]], 1))
            if self.has_boundary and active.size:
                bp = np.full(active.size, self.boundary_index, dtype=np.int64)
                out.append(np.stack([active, bp], 1))
                out.append(np.stack([bp, active], 1))
        return np.concatenate(out, 0) if out else np.zeros((0, 2), np.int64)

    def compute_edge_features(self, edge_index):
        """Return ``(distance_ids, geometry_features)`` for a set of directed edges."""
        if len(edge_index) == 0:
            return np.zeros(0, np.int64), np.zeros((0, 3), np.float32)
        i, j = edge_index[:, 0], edge_index[:, 1]
        physical = np.where(i < self.num_stabilizers, i, j)
        is_bdy = (i == self.boundary_index) | (j == self.boundary_index)
        ii = np.where(is_bdy, physical, i)
        jj = np.where(is_bdy, physical, j)
        dist = np.where(is_bdy,
                        self.boundary_distance[ii] if self.has_boundary else 0,
                        self.pair_distance[ii, jj])
        dx = np.where(is_bdy, 0.0, self.pair_dx[ii, jj])
        dy = np.where(is_bdy, 0.0, self.pair_dy[ii, jj])
        geo = np.stack([dx, dy, self.stabilizer_type[physical]], 1).astype(np.float32)
        return dist.astype(np.int64), geo

    def generate_training_labels(self, edge_index, pairs):
        """Return binary label array for supervised training."""
        y = np.zeros(len(edge_index), np.float32)
        if pairs:
            base = self.boundary_index + 1
            key = edge_index[:, 0] * base + edge_index[:, 1]
            hashes = [i * base + j for i, j in pairs] + [j * base + i for i, j in pairs]
            y[np.isin(key, hashes)] = 1.0
        return y

    # ------------------------------------------------------------------
    # Ground-truth label generation
    # ------------------------------------------------------------------

    def ground_truth(self, x_errors, z_errors, timeout=10.0):
        """Compute canonical ground-truth matchings for supervised training.

        Returns a list of ``(stabilizer_i, stabilizer_j)`` pairs, or ``None``
        when the timeout is exceeded or no valid homologically-correct matching
        exists.
        """
        deadline = time.monotonic() + timeout
        pairs = []
        for stabilizer_type, error in ((0, z_errors), (1, x_errors)):
            qubit_errors = np.flatnonzero(error)
            if qubit_errors.size == 0:
                continue

            parent = {int(q): int(q) for q in qubit_errors}

            def find(node):
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = parent[node]
                return node

            seen = {}
            for q in map(int, qubit_errors):
                for s in self.qubit_to_stabilizers_by_type[stabilizer_type][q]:
                    if s in seen:
                        ra, rb = find(q), find(seen[s])
                        if ra != rb:
                            parent[ra] = rb
                    else:
                        seen[s] = q
            clusters = {}
            for q in map(int, qubit_errors):
                clusters.setdefault(find(q), []).append(q)

            for cluster_qubits in clusters.values():
                count = {}
                for q in cluster_qubits:
                    for s in self.qubit_to_stabilizers_by_type[stabilizer_type][q]:
                        count[s] = count.get(s, 0) + 1
                defects = sorted(s for s, c in count.items() if c % 2)
                target = tuple(
                    int(self.logical_supports[stabilizer_type][l, cluster_qubits].sum() % 2)
                    for l in range(self.num_logicals))
                if not defects:
                    if any(target):
                        return None
                    continue
                if len(defects) == 2:
                    matching = [tuple(defects)]
                else:
                    matching = self.local_mwpm(defects, deadline)
                if matching is None or self.matching_parity(matching) != target:
                    matching = self.brute_force_matching(defects, target, deadline)
                if matching is None:
                    return None
                pairs += matching
        return pairs

    def matching_parity(self, matching):
        parity = [0] * self.num_logicals
        for i, j in matching:
            bits = self.boundary_crossings[i] if j == self.boundary_index else self.pair_crossings[i, j]
            parity = [a ^ int(b) for a, b in zip(parity, bits)]
        return tuple(parity)

    def local_mwpm(self, defects, deadline):
        total = len(defects)
        if (total % 2 and not self.has_boundary) or total > 20:
            return None
        INF = float("inf")
        dp, choice = [INF] * (1 << total), [None] * (1 << total)
        dp[0] = 0.0
        for mask in range(1, 1 << total):
            if mask % 4096 == 0 and time.monotonic() > deadline:
                return None
            a = (mask & -mask).bit_length() - 1
            rest = mask ^ (1 << a)
            if self.has_boundary:
                w = dp[rest] + self.boundary_distance[defects[a]]
                if w < dp[mask]:
                    dp[mask], choice[mask] = w, (a, None)
            bb = rest
            while bb:
                b = (bb & -bb).bit_length() - 1
                bb &= bb - 1
                w = dp[rest ^ (1 << b)] + self.pair_distance[defects[a], defects[b]]
                if w < dp[mask]:
                    dp[mask], choice[mask] = w, (a, b)
        full = (1 << total) - 1
        if dp[full] == INF:
            return None
        matching, mask = [], full
        while mask:
            a, b = choice[mask]
            if b is None:
                matching.append((defects[a], self.boundary_index))
                mask ^= 1 << a
            else:
                matching.append((defects[a], defects[b]))
                mask ^= (1 << a) | (1 << b)
        return matching

    def brute_force_matching(self, defects, target, deadline):
        steps = [0]

        def rec(remaining, accumulated, parity):
            steps[0] += 1
            if steps[0] % 1024 == 0 and time.monotonic() > deadline:
                raise TimeoutError
            if not remaining:
                return list(accumulated) if parity == target else None
            i, rest = remaining[0], remaining[1:]
            for k in range(len(rest)):
                j = rest[k]
                updated = tuple(a ^ int(b) for a, b in zip(parity, self.pair_crossings[i, j]))
                out = rec(rest[:k] + rest[k + 1:], accumulated + [(i, j)], updated)
                if out is not None:
                    return out
            if self.has_boundary:
                updated = tuple(a ^ int(b) for a, b in zip(parity, self.boundary_crossings[i]))
                out = rec(rest, accumulated + [(i, self.boundary_index)], updated)
                if out is not None:
                    return out
            return None

        try:
            return rec(tuple(defects), [], tuple(0 for _ in target))
        except TimeoutError:
            return None


class ToricCode(CSSCode):
    """Toric code on an ``L x L`` periodic lattice.

    Parameters
    ----------
    L:
        Even lattice size.  The code has ``2L²`` physical qubits and ``2L²``
        stabilizers (``L²`` X-type vertex stabilizers and ``L²`` Z-type
        plaquette stabilizers), encoding 2 logical qubits with distance ``L``.
    """

    def __init__(self, L):
        self.L, self.n = L, 2 * L * L
        h = lambda x, y: (y % L) * L + (x % L)
        v = lambda x, y: L * L + (y % L) * L + (x % L)
        self.supports, self.stabilizer_type, coords = [], [], []
        for g in (0, 1):
            for y in range(L):
                for x in range(L):
                    if g == 0:
                        self.supports.append([h(x, y), h(x - 1, y), v(x, y), v(x, y - 1)])
                    else:
                        self.supports.append([h(x, y), h(x, y + 1), v(x, y), v(x + 1, y)])
                    self.stabilizer_type.append(g)
                    coords.append((x, y))
        self.coords = np.array(coords, float)
        self.rho = np.sqrt(((self.coords - L / 2) ** 2).sum(1, keepdims=True))

        Ls = np.zeros((2, 2, self.n), np.uint8)
        for x in range(L):
            Ls[0, 0, v(x, 0)] = 1
            Ls[1, 0, h(x, 0)] = 1
        for y in range(L):
            Ls[0, 1, h(0, y)] = 1
            Ls[1, 1, v(0, y)] = 1
        self.logical_supports = Ls

        N = 2 * L * L
        self.pair_distance = np.zeros((N, N))
        self.pair_dx = np.zeros((N, N))
        self.pair_dy = np.zeros((N, N))
        self.pair_crossings = np.zeros((N, N, 2), np.uint8)
        wrap = lambda a, b: (b - a) % L if (b - a) % L <= (a - b) % L else -((a - b) % L)
        for g in (0, 1):
            idx = [s for s in range(N) if self.stabilizer_type[s] == g]
            for i in idx:
                xi, yi = int(coords[i][0]), int(coords[i][1])
                for j in idx:
                    if i == j:
                        continue
                    xj, yj = int(coords[j][0]), int(coords[j][1])
                    qs = self.path(g, xi, yi, xj, yj, h, v)
                    self.pair_distance[i, j] = len(qs)
                    self.pair_dx[i, j] = wrap(xi, xj)
                    self.pair_dy[i, j] = wrap(yi, yj)
                    for l in range(2):
                        self.pair_crossings[i, j, l] = self.logical_supports[g][l, qs].sum() % 2
        self.build_structures()

    def path(self, stabilizer_type, source_col, source_row, target_col, target_row, h, v):
        L, qs = self.L, []
        dx = (target_col - source_col) % L
        sx, nx = (1, dx) if dx <= L - dx else (-1, L - dx)
        x = source_col
        for _ in range(nx):
            lo = x if sx == 1 else (x - 1) % L
            qs.append(h(lo, source_row) if stabilizer_type == 0 else v(lo + 1, source_row))
            x = (x + sx) % L
        dy = (target_row - source_row) % L
        sy, ny = (1, dy) if dy <= L - dy else (-1, L - dy)
        y = source_row
        for _ in range(ny):
            lo = y if sy == 1 else (y - 1) % L
            qs.append(v(target_col, lo) if stabilizer_type == 0 else h(target_col, lo + 1))
            y = (y + sy) % L
        return qs


class RotatedSurfaceCode(CSSCode):
    """Rotated surface code on an ``L x L`` square lattice with open boundaries.

    Parameters
    ----------
    L:
        Odd code distance.  The code has ``L²`` physical qubits, encodes 1
        logical qubit, and has distance ``L``.
    """

    has_boundary = True

    def __init__(self, L):
        assert L % 2 == 1, "rotated surface code distance must be odd"
        self.L, self.n = L, L * L
        q = lambda r, c: r * L + c
        self.supports, self.stabilizer_type, coords = [], [], []
        for i in range(-1, L):
            for j in range(-1, L):
                g = 0 if (i + j) % 2 == 0 else 1
                qs = [q(r, c) for r in (i, i + 1) for c in (j, j + 1)
                      if 0 <= r < L and 0 <= c < L]
                if len(qs) == 4 or (len(qs) == 2 and (
                        (g == 0 and i in (-1, L - 1)) or (g == 1 and j in (-1, L - 1)))):
                    self.supports.append(qs)
                    self.stabilizer_type.append(g)
                    coords.append((i + 0.5, j + 0.5))
        self.coords = np.array(coords, float)
        self.rho = np.array([[np.mean([qq // L for qq in qs]),
                               np.mean([qq % L for qq in qs])]
                              for qs in self.supports])

        Ls = np.zeros((2, 1, self.n), np.uint8)
        Ls[0, 0, [q(r, 0) for r in range(L)]] = 1
        Ls[1, 0, [q(0, c) for c in range(L)]] = 1
        self.logical_supports = Ls

        N = len(self.supports)
        qubit_stabs = [[[] for _ in range(self.n)], [[] for _ in range(self.n)]]
        for s, qs in enumerate(self.supports):
            for qq in qs:
                qubit_stabs[self.stabilizer_type[s]][qq].append(s)
        adjacency = [[] for _ in range(N)]
        boundary_qubits = [[] for _ in range(N)]
        for g in (0, 1):
            for qq in range(self.n):
                ss = qubit_stabs[g][qq]
                if len(ss) == 2:
                    adjacency[ss[0]].append((ss[1], qq))
                    adjacency[ss[1]].append((ss[0], qq))
                elif len(ss) == 1:
                    boundary_qubits[ss[0]].append(qq)

        self.pair_distance = np.zeros((N, N))
        self.pair_crossings = np.zeros((N, N, 1), np.uint8)
        self.pair_dx = self.coords[None, :, 0] - self.coords[:, None, 0]
        self.pair_dy = self.coords[None, :, 1] - self.coords[:, None, 1]
        for s0 in range(N):
            g = self.stabilizer_type[s0]
            dist, par, dq = {s0: 0}, {s0: 0}, deque([s0])
            while dq:
                u = dq.popleft()
                for vtx, qq in adjacency[u]:
                    if vtx not in dist:
                        dist[vtx] = dist[u] + 1
                        par[vtx] = par[u] ^ int(self.logical_supports[g][0, qq])
                        dq.append(vtx)
            for vtx, dd in dist.items():
                self.pair_distance[s0, vtx] = dd
                self.pair_crossings[s0, vtx, 0] = par[vtx]

        self.boundary_distance = np.full(N, np.inf)
        self.boundary_crossings = np.zeros((N, 1), np.uint8)
        dq = deque()
        for s in range(N):
            if boundary_qubits[s]:
                self.boundary_distance[s] = 1
                self.boundary_crossings[s, 0] = int(
                    self.logical_supports[self.stabilizer_type[s]][0, sorted(boundary_qubits[s])[0]])
                dq.append(s)
        while dq:
            u = dq.popleft()
            for vtx, qq in adjacency[u]:
                if self.boundary_distance[vtx] > self.boundary_distance[u] + 1:
                    self.boundary_distance[vtx] = self.boundary_distance[u] + 1
                    self.boundary_crossings[vtx, 0] = self.boundary_crossings[u, 0] ^ int(
                        self.logical_supports[self.stabilizer_type[vtx]][0, qq])
                    dq.append(vtx)
        self.build_structures()
