"""Training script for the QWP neural network."""

import argparse
import time

import numpy as np
import torch

from .codes import ToricCode, RotatedSurfaceCode
from .model import QWP, build_batch


def sample_labeled_training_batch(code, batch_size, p, noise, rng, gt_timeout,
                                  eta, px_frac=1/3, py_frac=1/3, pz_frac=1/3):
    """Draw a batch of (syndrome, edge list, label) triples for supervised training.

    Samples with no defects or for which ground-truth labelling times out are
    silently skipped.  Returns ``None`` when every sample in the batch was
    rejected.
    """
    ex, ez, synd = code.sample(batch_size, p, noise,
                                seed=int(rng.integers(2 ** 48)), eta=eta,
                                px_frac=px_frac, py_frac=py_frac, pz_frac=pz_frac)
    syndromes, edge_lists, labels = [], [], []
    for k in range(batch_size):
        edge_index = code.build_syndrome_graph(synd[k])
        if len(edge_index) == 0:
            continue
        pairs = code.ground_truth(ex[k], ez[k], timeout=gt_timeout)
        if pairs is None:
            continue
        syndromes.append(synd[k])
        edge_lists.append(edge_index)
        labels.append(code.generate_training_labels(edge_index, pairs))
    return (syndromes, edge_lists, labels) if syndromes else None


def main():
    parser = argparse.ArgumentParser(
        prog="nmwpm-train",
        description="Train the QWP neural network for NMWPM decoding.")
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
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batches-per-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=9e-5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--p-min", type=float, default=0.05)
    parser.add_argument("--p-max", type=float, default=0.20)
    parser.add_argument("--p-count", type=int, default=9)
    parser.add_argument("--gt-timeout", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--enc-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="train.pt",
                        help="Output path for the checkpoint")
    parser.add_argument("--transfer-from", default=None,
                        help="Initialise Transformer layers from an existing checkpoint")
    parser.add_argument("--resume", default=None, metavar="CKPT",
                        help="Resume training from an existing checkpoint")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    code = ToricCode(args.L) if args.code == "toric" else RotatedSurfaceCode(args.L)
    model = QWP(code, hidden_dim=args.hidden_dim, gnn_layers=args.gnn_layers,
                num_heads=args.heads, enc_layers=args.enc_layers).to(args.device)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=True)
        model.load_state_dict(ck["model"])
        start_epoch = ck.get("epoch", 0)
        print(f"Resumed {args.resume} from epoch {start_epoch}/{args.epochs}")
    elif args.transfer_from:
        src = torch.load(args.transfer_from, map_location=args.device, weights_only=True)
        keep = {k: v for k, v in src["model"].items()
                if k.startswith(("encoder.", "out.", "tok_norm."))}
        model.load_state_dict(keep, strict=False)
        print(f"initialised {len(keep)} Transformer tensors from {args.transfer_from}")

    n_params = sum(param.numel() for param in model.parameters())
    print(f"QWP  code={args.code} L={args.L}  noise={args.noise}  "
          f"hidden={args.hidden_dim} gnn={args.gnn_layers} enc={args.enc_layers} "
          f"heads={args.heads}  params={n_params / 1e6:.3f}M")

    if start_epoch >= args.epochs:
        print(f"Already complete ({start_epoch}/{args.epochs} epochs). Nothing to do.")
        return

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.min_lr)
    # Advance the scheduler to the correct position when resuming
    for _ in range(start_epoch):
        sched.step()
    physical_error_rates = np.linspace(args.p_min, args.p_max, args.p_count)

    for epoch in range(start_epoch, args.epochs):
        start_time, tot, nb = time.time(), 0.0, 0
        for _ in range(args.batches_per_epoch):
            batch = sample_labeled_training_batch(
                code, args.batch_size,
                float(rng.choice(physical_error_rates)),
                args.noise, rng, args.gt_timeout, args.eta,
                px_frac=args.px_frac, py_frac=args.py_frac, pz_frac=args.pz_frac)
            if batch is None:
                continue
            syndromes, edge_lists, labels = batch
            tensors = build_batch(code, syndromes, edge_lists, args.device)
            edge_mask = tensors[-1]
            y = torch.zeros_like(edge_mask, dtype=torch.float32)
            for b, lab in enumerate(labels):
                y[b, :len(lab)] = torch.as_tensor(lab, device=args.device)
            logits = model(*tensors)
            m = edge_mask.float()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, reduction="none")
            p_hat = torch.sigmoid(logits)
            ent = -(p_hat * torch.nn.functional.logsigmoid(logits)
                    + (1 - p_hat) * torch.nn.functional.logsigmoid(-logits))
            loss = ((bce + args.lam * ent) * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        sched.step()
        elapsed = time.time() - start_time
        syndromes_per_sec = (nb * args.batch_size) / max(elapsed, 1e-9)
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "config": {
                "code": args.code, "L": args.L, "noise": args.noise, "eta": args.eta,
                "hidden_dim": args.hidden_dim, "gnn_layers": args.gnn_layers,
                "num_heads": args.heads, "enc_layers": args.enc_layers,
                "px_frac": args.px_frac, "py_frac": args.py_frac, "pz_frac": args.pz_frac,
            }
        }, args.out)
        print(f"epoch {epoch + 1}/{args.epochs}  "
              f"loss {tot / max(nb, 1):.6f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  "
              f"{elapsed:.1f}s  {syndromes_per_sec:.0f} syn/s  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
