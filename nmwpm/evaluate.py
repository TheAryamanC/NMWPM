"""Evaluation utilities for NMWPM decoders."""

import argparse
import os
import time

import numpy as np
import torch

from .codes import ToricCode, RotatedSurfaceCode
from .decoder import (
    Decoder,
    _decode_homology_parity,
    _canonical_weighted_pairs,
    _aggregate_directed_edge_probs,
)
from .model import QWP, build_batch


@torch.no_grad()
def evaluate_point(code, model, p, shots, noise, device, chunk, seed,
                   agg="max", eta=10.0, uniform=False, fast_path=0,
                   px_frac=1/3, py_frac=1/3, pz_frac=1/3):
    """Evaluate logical error rates for MWPM, NMWPM, and optionally uniform MWPM.

    Returns
    -------
    tuple
        ``(ler_mwpm, ler_nmwpm, ler_uniform, fast_fraction, nn_ms_per_shot)``
        where ``nn_ms_per_shot`` is mean NN batch-inference time in milliseconds
        per syndrome (0.0 when no model is used).
    """
    logical_error_counts = np.zeros(3, np.int64)
    fast_count = 0
    t_nn_total = 0.0
    nn_shots = 0
    rng = np.random.default_rng(seed)
    done = 0
    while done < shots:
        B = min(chunk, shots - done)
        ex, ez, synd = code.sample(B, p, noise, seed=int(rng.integers(2 ** 48)),
                                   eta=eta, px_frac=px_frac, py_frac=py_frac,
                                   pz_frac=pz_frac)
        true_parity = code.error_parities(ex, ez)
        edge_lists = [code.build_syndrome_graph(synd[k]) for k in range(B)]
        fast = [synd[k].sum() <= fast_path for k in range(B)]
        nz = [k for k in range(B) if len(edge_lists[k]) and not fast[k]]
        probs = {}
        if model is not None and nz:
            t0 = time.perf_counter()
            tensors = build_batch(code, [synd[k] for k in nz],
                                  [edge_lists[k] for k in nz], device)
            logits = model(*tensors)
            t_nn_total += time.perf_counter() - t0
            nn_shots += len(nz)
            for r, k in enumerate(nz):
                probs[k] = torch.sigmoid(logits[r, :len(edge_lists[k])]).cpu().numpy()
        for k in range(B):
            if len(edge_lists[k]) == 0:
                logical_error_counts += true_parity[k].any()
                continue
            defects = np.flatnonzero(synd[k])
            par = _decode_homology_parity(
                code, defects, _canonical_weighted_pairs(code, edge_lists[k]))
            logical_error_counts[0] += (par != true_parity[k]).any()
            if model is not None:
                if fast[k]:
                    fast_count += 1
                    logical_error_counts[1] += (par != true_parity[k]).any()
                else:
                    par2 = _decode_homology_parity(
                        code, defects,
                        _aggregate_directed_edge_probs(code, edge_lists[k], probs[k], agg))
                    logical_error_counts[1] += (par2 != true_parity[k]).any()
            if uniform:
                par3 = _decode_homology_parity(
                    code, defects,
                    _canonical_weighted_pairs(code, edge_lists[k], uniform=True))
                logical_error_counts[2] += (par3 != true_parity[k]).any()
        done += B
    nn_ms = 1e3 * t_nn_total / max(nn_shots, 1)
    return (logical_error_counts[0] / shots,
            logical_error_counts[1] / shots,
            logical_error_counts[2] / shots if uniform else float("nan"),
            fast_count / shots,
            nn_ms)


@torch.no_grad()
def benchmark(code, model, p, noise, device, reps=200, eta=10.0, fast_path=0,
             px_frac=1/3, py_frac=1/3, pz_frac=1/3):
    """Print mean single-shot latency for QWP inference and MWPM decoding."""
    ex, ez, synd = code.sample(reps, p, noise, seed=0, eta=eta,
                                px_frac=px_frac, py_frac=py_frac, pz_frac=pz_frac)
    t_nn = t_mwpm = n = fast_n = 0.0
    for k in range(reps):
        edge_index = code.build_syndrome_graph(synd[k])
        if len(edge_index) == 0:
            continue
        if synd[k].sum() <= fast_path:
            t0 = time.perf_counter()
            _decode_homology_parity(
                code, np.flatnonzero(synd[k]),
                _canonical_weighted_pairs(code, edge_index))
            t_mwpm += time.perf_counter() - t0
            n, fast_n = n + 1, fast_n + 1
            continue
        t0 = time.perf_counter()
        tensors = build_batch(code, [synd[k]], [edge_index], device)
        pr = torch.sigmoid(model(*tensors))[0].cpu().numpy()
        t1 = time.perf_counter()
        _decode_homology_parity(
            code, np.flatnonzero(synd[k]),
            _aggregate_directed_edge_probs(code, edge_index, pr))
        t2 = time.perf_counter()
        t_nn, t_mwpm, n = t_nn + t1 - t0, t_mwpm + t2 - t1, n + 1
    pct = 100 * fast_n / max(n, 1)
    print(f"NMWPM single-shot latency at p={p}, fast-path<={fast_path}, "
          f"({int(n)} shots, {pct:.0f}% fast-pathed): "
          f"QWP {1e3 * t_nn / n:.3f} ms  MWPM {1e3 * t_mwpm / n:.3f} ms  "
          f"total {1e3 * (t_nn + t_mwpm) / n:.3f} ms/shot")


def threshold_from_csv(path):
    """Estimate the threshold error rate from a CSV results file."""
    rows = [l.strip().split(",") for l in open(path)]
    header, rows = rows[0], rows[1:]
    cols = [("MWPM", header.index("ler_mwpm")), ("NMWPM", header.index("ler_nmwpm"))]
    if "ler_uniform" in header:
        cols.insert(1, ("UNIFORM", header.index("ler_uniform")))
    iL, ip = header.index("L"), header.index("p")
    for name, col in cols:
        data = {}
        for r in rows:
            if r[col] not in ("nan", "None", ""):
                data.setdefault(int(r[iL]), {})[float(r[ip])] = float(r[col])
        Ls = sorted(data)
        for La, Lb in zip(Ls, Ls[1:]):
            ps = sorted(set(data[La]) & set(data[Lb]))
            d = [np.log(max(data[Lb][q], 1e-9)) - np.log(max(data[La][q], 1e-9)) for q in ps]
            for t in range(len(ps) - 1):
                if d[t] <= 0 <= d[t + 1]:
                    pth = ps[t] + (ps[t + 1] - ps[t]) * (-d[t]) / (d[t + 1] - d[t])
                    print(f"{name}: L={La}/{Lb} curves cross at p_th ~ {pth:.4f}")


def main():
    parser = argparse.ArgumentParser(
        prog="nmwpm-evaluate",
        description="Evaluate NMWPM (or MWPM baseline) logical error rates.")
    parser.add_argument("--code", choices=["toric", "rotated"], default="toric")
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--noise",
                        choices=["depolarizing", "independent",
                                 "biased", "biased_x",
                                 "x_only", "z_only", "y_only", "pauli"],
                        default="depolarizing")
    parser.add_argument("--eta", type=float, default=10.0,
                        help="Z- (or X-) bias ratio for biased/biased_x noise")
    parser.add_argument("--px-frac", type=float, default=1/3,
                        help="Relative X weight for pauli noise")
    parser.add_argument("--py-frac", type=float, default=1/3,
                        help="Relative Y weight for pauli noise")
    parser.add_argument("--pz-frac", type=float, default=1/3,
                        help="Relative Z weight for pauli noise")
    parser.add_argument("--ckpt", default=None,
                        help="Path to a trained .pt checkpoint")
    parser.add_argument("--p", type=float, nargs="+",
                        default=[0.10, 0.13, 0.16, 0.19],
                        help="Physical error rate(s) to evaluate")
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", default=None, help="Append results to this CSV file")
    parser.add_argument("--agg", choices=["max", "min"], default="max")
    parser.add_argument("--uniform", action="store_true",
                        help="Also evaluate uniform-weight MWPM baseline")
    parser.add_argument("--fast-path", type=int, default=0, metavar="K",
                        help="Bypass NN for syndromes with <=K active stabilizers")
    parser.add_argument("--benchmark", action="store_true",
                        help="Measure single-shot latency instead of LER")
    parser.add_argument("--reps", type=int, default=200,
                        help="Number of shots for latency benchmark")
    parser.add_argument("--threshold", metavar="CSV", default=None,
                        help="Estimate threshold from an existing CSV file")
    args = parser.parse_args()

    if args.threshold:
        threshold_from_csv(args.threshold)
        return

    code = ToricCode(args.L) if args.code == "toric" else RotatedSurfaceCode(args.L)
    model = None
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=args.device, weights_only=True)
        cfg = ck["config"]
        assert cfg["code"] == args.code and cfg["L"] == args.L, \
            f"checkpoint was trained for {cfg}"
        model = QWP(
            code,
            hidden_dim=cfg.get("hidden_dim", 128),
            gnn_layers=cfg.get("gnn_layers", 4),
            num_heads=cfg.get("num_heads", 4),
            enc_layers=cfg.get("enc_layers", 2),
        ).to(args.device).eval()
        model.load_state_dict(ck["model"])
        n_params = sum(param.numel() for param in model.parameters())
        trained_epoch = ck.get("epoch", "?")
        print(f"Loaded  {args.ckpt}")
        print(f"        hidden={cfg.get('hidden_dim',128)}  gnn={cfg.get('gnn_layers',4)}  "
              f"enc={cfg.get('enc_layers',2)}  heads={cfg.get('num_heads',4)}  "
              f"params={n_params / 1e6:.3f}M  epoch={trained_epoch}")

    if args.benchmark:
        assert model is not None, "--benchmark requires --ckpt"
        benchmark(code, model, args.p[0], args.noise, args.device,
                  reps=args.reps, eta=args.eta, fast_path=args.fast_path,
                  px_frac=args.px_frac, py_frac=args.py_frac, pz_frac=args.pz_frac)
        return

    noise_tag = args.noise
    if args.noise in ("biased", "biased_x"):
        noise_tag = f"{args.noise}(eta={args.eta:g})"
    elif args.noise == "pauli":
        noise_tag = (f"pauli(px={args.px_frac:.2f} py={args.py_frac:.2f} "
                     f"pz={args.pz_frac:.2f})")
    print(f"{args.code} L={args.L}  noise={noise_tag}  "
          f"shots={args.shots}  fast-path<={args.fast_path}")
    hdr = (f"{'p':>8} {'LER-MWPM':>10} {'LER-UNIF':>10} {'LER-NMWPM':>10} "
           f"{'fast%':>6} {'NN-ms/shot':>11}")
    print(hdr)
    print("-" * len(hdr))
    for p in args.p:
        l0, l1, lu, ff, t_nn = evaluate_point(
            code, model, p, args.shots, args.noise, args.device,
            args.chunk, args.seed, args.agg, args.eta, args.uniform, args.fast_path,
            args.px_frac, args.py_frac, args.pz_frac)
        if model is None:
            l1 = t_nn = float("nan")
        print(f"{p:>8.4f} {l0:>10.5f} {lu:>10.5f} {l1:>10.5f} "
              f"{100 * ff:>5.1f}% {t_nn:>10.3f}", flush=True)
        if args.csv:
            new = not os.path.exists(args.csv)
            with open(args.csv, "a") as f:
                if new:
                    f.write("code,L,noise,eta,p,shots,ler_mwpm,ler_uniform,"
                            "ler_nmwpm,fastpath_frac,nn_ms_per_shot\n")
                eta_val = args.eta if args.noise in ("biased", "biased_x") else ""
                f.write(f"{args.code},{args.L},{args.noise},{eta_val},{p},{args.shots},"
                        f"{l0},{lu},{l1},{ff},{t_nn}\n")


if __name__ == "__main__":
    main()
