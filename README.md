# Neural Minimum-Weight Perfect Matching (NMWPM)

An open-source, [Stim](https://github.com/quantumlib/Stim)-based implementation of **Neural MWPM** for quantum error correction on surface codes. The decoder improves on standard distance-weighted MWPM by using a learned **Quantum Weight Predictor (QWP)** network to assign correlation-aware edge weights to the matching graph.

> A single 3.9M-parameter model decodes a syndrome in ~7 ms on a laptop GPU and reduces logical error rates by 51–60% under depolarizing noise compared to MWPM.

---

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Python API](#python-api)
  - [Decoder](#decoder)
  - [Code classes](#code-classes)
  - [QWP model](#qwp-model)
- [CLI tools](#cli-tools)
  - [nmwpm-evaluate](#nmwpm-evaluate)
  - [nmwpm-train](#nmwpm-train)
- [Noise models](#noise-models)
- [Comprehensive run script](#comprehensive-run-script)
- [Pretrained checkpoints](#pretrained-checkpoints)
- [Training from scratch](#training-from-scratch)
- [Reproducing paper results](#reproducing-paper-results)
- [Background](#background)

---

## Installation

```bash
pip install nmwpm
```

**Requirements**: Python ≥ 3.9, PyTorch ≥ 2.0, [Stim](https://github.com/quantumlib/Stim) ≥ 1.14, [PyMatching](https://github.com/oscarhiggott/PyMatching) ≥ 2.2.

Install from source:

```bash
git clone https://github.com/aryuc/NMWPM
cd NMWPM
pip install -e .
```

---

## Quickstart

```python
import numpy as np
import nmwpm

# Load a pretrained decoder (automatically reconstructs the code)
decoder = nmwpm.Decoder.from_checkpoint("checkpoints/nmwpm_toric_L8_depolarizing.pt")

# Decode a single syndrome measurement.
# syndrome is a 1-D binary array of length code.num_stabilizers:
#   1 = stabilizer fired (defect), 0 = no defect
syndrome = np.zeros(128, dtype=np.uint8)  # toric L=8 has 128 stabilizers
syndrome[[3, 17]] = 1                     # two active defects

parity = decoder.decode(syndrome)
# parity shape: (2 * num_logicals,) = (4,) for the toric code
# parity[l]               = 1 if X-type logical error on logical qubit l
# parity[num_logicals + l] = 1 if Z-type logical error on logical qubit l

# Batch decode many syndromes at once
syndromes = np.stack([syndrome, syndrome])  # shape (B, num_stabilizers)
parities  = decoder.decode_batch(syndromes) # shape (B, 2*num_logicals)
```

---

## Python API

### Decoder

The main entry point. Wraps a code geometry and an optional neural network.

```python
nmwpm.Decoder(code, model=None, device=None, agg="max", fast_path=0)
```

| Parameter | Description |
|-----------|-------------|
| `code` | A `ToricCode` or `RotatedSurfaceCode` instance. |
| `model` | A trained `QWP` network. If `None`, falls back to plain distance-weighted MWPM. |
| `device` | Torch device string (`"cpu"`, `"cuda"`, `"cuda:0"`, …). Defaults to CUDA when available. |
| `agg` | Aggregation strategy for converting directed neural edge scores to undirected weights: `"max"` (default) or `"min"`. |
| `fast_path` | Skip the neural network for syndromes with ≤ `fast_path` active stabilizers (default `0` = always use the network). Useful for reducing latency on quiet syndromes. |

#### `Decoder.from_checkpoint(path, device=None, agg="max", fast_path=0)`

Load a pretrained decoder. The checkpoint stores the model weights **and** the code configuration, so the matching code object is reconstructed automatically.

```python
decoder = nmwpm.Decoder.from_checkpoint(
    "checkpoints/nmwpm_toric_L8_depolarizing.pt",
    device="cpu",   # override device
    fast_path=3,    # bypass NN for trivial syndromes
)
print(decoder)
# Decoder(toric L=8, QWP hidden_dim=128, device='cpu')
```

#### `Decoder.mwpm(code, device=None)`

Create a plain MWPM decoder without a neural network.

```python
code    = nmwpm.ToricCode(L=8)
decoder = nmwpm.Decoder.mwpm(code)
```

#### `decoder.decode(syndrome) → np.ndarray`

Decode a **single** syndrome measurement.

```python
# syndrome: 1-D uint8 array, shape (num_stabilizers,)
parity = decoder.decode(syndrome)
# returns uint8 array of shape (2 * num_logicals,)
```

`parity[l]` is `1` if an X-type logical error was predicted on logical qubit `l`.
`parity[num_logicals + l]` is `1` if a Z-type logical error was predicted.

#### `decoder.decode_batch(syndromes) → np.ndarray`

Decode a **batch** of syndromes. Preferred for throughput; the neural network processes all non-fast-path syndromes in a single GPU call.

```python
# syndromes: 2-D uint8 array, shape (batch, num_stabilizers)
parities = decoder.decode_batch(syndromes)
# returns uint8 array of shape (batch, 2 * num_logicals)
```

---

### Code classes

Both codes are CSS codes and share the same interface.

#### `nmwpm.ToricCode(L)`

The toric code on an `L × L` periodic lattice.

- **Physical qubits**: `2L²`
- **Stabilizers**: `2L²` (`L²` X-type vertex + `L²` Z-type plaquette)
- **Logical qubits**: 2
- **Code distance**: `L`
- `L` must be a positive even integer.

```python
code = nmwpm.ToricCode(L=8)
print(code.n)               # 128 physical qubits
print(code.num_stabilizers) # 128 stabilizers
print(code.num_logicals)    # 2 logical qubits
```

#### `nmwpm.RotatedSurfaceCode(L)`

The rotated surface code on an `L × L` square lattice with open boundaries.

- **Physical qubits**: `L²`
- **Stabilizers**: `L² - 1`
- **Logical qubits**: 1
- **Code distance**: `L`
- `L` must be a positive odd integer.

```python
code = nmwpm.RotatedSurfaceCode(L=7)
print(code.n)               # 49 physical qubits
print(code.num_stabilizers) # 48 stabilizers
print(code.num_logicals)    # 1 logical qubit
```

#### Sampling errors and syndromes

```python
# Draw Monte-Carlo error samples
error_x, error_z, syndrome = code.sample(
    shots=1000,
    p=0.10,                # physical error rate
    noise="depolarizing",  # see Noise models section for all options
    seed=42,
    eta=10.0,              # Z- or X-bias ratio (biased / biased_x)
    px_frac=1/3,           # relative X weight (pauli noise only)
    py_frac=1/3,           # relative Y weight (pauli noise only)
    pz_frac=1/3,           # relative Z weight (pauli noise only)
)
# error_x, error_z: shape (shots, n),              uint8 — Pauli X and Z errors per qubit
# syndrome:         shape (shots, num_stabilizers), uint8 — measurement outcomes

# Check which samples caused a logical error
true_parity = code.error_parities(error_x, error_z)
# shape (shots, 2 * num_logicals), uint8
```

#### Building syndrome graphs manually

```python
# Get all candidate defect-pair edges for a single syndrome row
edge_index = code.build_syndrome_graph(syndrome[0])
# shape (E, 2) — directed edges between active stabilizer indices
# code.boundary_index is used for half-boundary edges (rotated code only)

# Compute per-edge geometric features
distance_ids, geometry = code.compute_edge_features(edge_index)
# distance_ids: shape (E,),    int64  — integer Manhattan/graph distance
# geometry:     shape (E, 3),  float32 — (dx, dy, stabilizer_type)
```

---

### QWP model

The `QWP` (Quantum Weight Predictor) is a graph-transformer network that scores each candidate defect-pair edge.

```python
nmwpm.QWP(code, hidden_dim=128, gnn_layers=4, num_heads=4, enc_layers=2, d_pe=16)
```

| Parameter | Description |
|-----------|-------------|
| `code` | Code object (provides geometry buffers). |
| `hidden_dim` | Width of all hidden representations. Default `128`. |
| `gnn_layers` | Number of graph-transformer message-passing layers. Default `4`. |
| `num_heads` | Attention heads per GNN layer. Default `4`. |
| `enc_layers` | Transformer encoder layers applied to edge tokens. Default `2`. Set to `0` to disable. |
| `d_pe` | Dimension of sinusoidal positional encodings. Default `16`. |

```python
import torch
from nmwpm.model import build_batch

code  = nmwpm.ToricCode(L=8)
model = nmwpm.QWP(code).to("cuda").eval()

# Build padded batch tensors from raw syndromes and their edge lists
syndromes  = [syndrome_array_1, syndrome_array_2]  # list of (num_stab,) uint8 arrays
edge_lists = [edge_index_1, edge_index_2]          # list of (E, 2) int64 arrays
tensors = build_batch(code, syndromes, edge_lists, device="cuda")

# Forward pass — returns edge logits of shape (B, max_edges)
with torch.no_grad():
    logits = model(*tensors)
    probs  = torch.sigmoid(logits)  # predicted edge probabilities, (B, max_edges)
```

---

## CLI tools

Two command-line tools are installed with the package.

### nmwpm-evaluate

Evaluate logical error rates at one or more physical error rates.

```
nmwpm-evaluate [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--code` | `toric` | Code type: `toric` or `rotated`. |
| `--L` | `8` | Lattice size. |
| `--noise` | `depolarizing` | Noise model (see [Noise models](#noise-models)). |
| `--eta` | `10.0` | Z- or X-bias ratio for `biased`/`biased_x` noise. |
| `--px-frac` | `0.333` | Relative X weight for `pauli` noise. |
| `--py-frac` | `0.333` | Relative Y weight for `pauli` noise. |
| `--pz-frac` | `0.333` | Relative Z weight for `pauli` noise. |
| `--ckpt` | — | Path to a `.pt` checkpoint. Omit to run plain MWPM only. |
| `--p` | `0.10 0.13 0.16 0.19` | Physical error rate(s) to evaluate. |
| `--shots` | `20000` | Monte-Carlo samples per error rate. |
| `--chunk` | `256` | Batch size for GPU inference. |
| `--seed` | `12345` | Random seed. |
| `--device` | auto | Torch device. |
| `--csv` | — | Append results to a CSV file. |
| `--uniform` | off | Also evaluate uniform-weight MWPM as a control. |
| `--fast-path K` | `0` | Bypass NN for syndromes with ≤ K active stabilizers. |
| `--benchmark` | off | Measure single-shot latency instead of LER. |
| `--reps` | `200` | Shots for latency benchmark. |
| `--threshold CSV` | — | Estimate threshold from an existing CSV (no simulation). |

The output table includes a `NN-ms/shot` column showing mean batch-inference
latency of the neural network per syndrome.

**Examples**

```bash
# MWPM baseline — no checkpoint needed
nmwpm-evaluate --code toric --L 8 --p 0.10 0.13 0.16 0.19

# NMWPM with a pretrained checkpoint (prints model size + NN latency)
nmwpm-evaluate --code toric --L 8 --noise depolarizing \
    --ckpt checkpoints/nmwpm_toric_L8_depolarizing.pt \
    --shots 50000 --csv results/my_run.csv

# New noise types
nmwpm-evaluate --code toric --L 8 --noise x_only \
    --ckpt checkpoints/nmwpm_toric_L8_x_only.pt \
    --p 0.05 0.07 0.09 0.11 0.13 0.15

nmwpm-evaluate --code toric --L 8 --noise y_only \
    --ckpt checkpoints/nmwpm_toric_L8_y_only.pt \
    --p 0.04 0.06 0.08 0.10 0.12

nmwpm-evaluate --code toric --L 8 --noise biased_x --eta 10 \
    --ckpt checkpoints/nmwpm_toric_L8_biased_x_eta10.pt

nmwpm-evaluate --code toric --L 8 --noise pauli \
    --px-frac 0.5 --py-frac 0.5 --pz-frac 0.0 \
    --ckpt checkpoints/nmwpm_toric_L8_pauli_xy.pt \
    --p 0.04 0.06 0.08 0.10

# Latency benchmark (prints QWP ms + MWPM ms + total ms/shot)
nmwpm-evaluate --code toric --L 8 \
    --ckpt checkpoints/nmwpm_toric_L8_depolarizing.pt \
    --benchmark --reps 500

# Estimate threshold from a saved CSV
nmwpm-evaluate --threshold results/toric_depolarizing_L8.csv
```

### nmwpm-train

Train a new QWP model from scratch.

```
nmwpm-train [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--code` | `toric` | Code type: `toric` or `rotated`. |
| `--L` | `8` | Lattice size. |
| `--noise` | `depolarizing` | Noise model (see [Noise models](#noise-models)). |
| `--eta` | `10.0` | Z- or X-bias ratio for `biased`/`biased_x` noise. |
| `--px-frac` | `0.333` | Relative X weight for `pauli` noise. |
| `--py-frac` | `0.333` | Relative Y weight for `pauli` noise. |
| `--pz-frac` | `0.333` | Relative Z weight for `pauli` noise. |
| `--epochs` | `400` | Training epochs. |
| `--batches-per-epoch` | `500` | Gradient steps per epoch. |
| `--batch-size` | `32` | Syndromes per gradient step. |
| `--lr` | `9e-5` | Peak learning rate (Adam). |
| `--min-lr` | `1e-5` | Minimum learning rate (cosine schedule). |
| `--lam` | `0.01` | Entropy regularisation coefficient. |
| `--p-min` / `--p-max` | `0.05` / `0.20` | Physical error rate range for training. |
| `--p-count` | `9` | Number of evenly spaced error rates to sample from. |
| `--hidden-dim` | `128` | Model width. |
| `--gnn-layers` | `4` | GNN depth (0 = no GNN, encoder only). |
| `--heads` | `4` | Attention heads. |
| `--enc-layers` | `2` | Transformer encoder layers (0 = no encoder). |
| `--seed` | `0` | Random seed. |
| `--device` | auto | Torch device. |
| `--out` | `train.pt` | Output checkpoint path. |
| `--resume CKPT` | — | Resume training from an existing checkpoint. Skips automatically when the stored epoch count already equals `--epochs`. |
| `--transfer-from` | — | Warm-start Transformer layers from an existing checkpoint. |

Each epoch prints: `loss`, `lr`, elapsed time in seconds, and **syndromes/second throughput**.

**Examples**

```bash
# Train a model for toric L=8 with depolarizing noise
nmwpm-train --code toric --L 8 --noise depolarizing --out my_model.pt

# Transfer Transformer weights from an existing model; retrain GNN for a new lattice size
nmwpm-train --code toric --L 10 \
    --transfer-from checkpoints/nmwpm_toric_L8_depolarizing.pt \
    --out nmwpm_toric_L10_depolarizing.pt

# Compact model (18× fewer parameters — halved width, fewer layers)
nmwpm-train --code toric --L 8 --hidden-dim 32 --gnn-layers 2 --enc-layers 1 \
    --out nmwpm_toric_L8_small.pt

# Train under X-biased noise (η = 100)
nmwpm-train --code toric --L 8 --noise biased_x --eta 100 \
    --p-min 0.05 --p-max 0.18 --out nmwpm_toric_L8_biased_x_eta100.pt

# Train under x_only (pure bit-flip) noise
nmwpm-train --code toric --L 8 --noise x_only \
    --p-min 0.03 --p-max 0.14 --out nmwpm_toric_L8_x_only.pt

# Resume interrupted training
nmwpm-train --code toric --L 8 --noise depolarizing \
    --resume my_model.pt --out my_model.pt --epochs 400
```

---

## Noise models

All noise models operate at the **code-capacity level**: errors are applied
independently to each physical qubit before any measurement, using Stim's
single-qubit error channels.  The full set of supported models is exposed as
`nmwpm.codes.CSSCode.NOISE_MODELS`.

| `--noise` | Stim channel | Description |
|-----------|-------------|-------------|
| `depolarizing` | `DEPOLARIZE1(p)` | Equal probability of X, Y, Z errors. Each occurs with probability p/3. Default model. |
| `independent` | `X_ERROR(p)` + `Z_ERROR(p)` | Independent X and Z bit/phase flips, each at rate *p*. |
| `biased` | `PAULI_CHANNEL_1(px,py,pz)` | Z-biased channel. `--eta` sets the Z/X ratio: pz = p·η/(1+η), px = py = p/(2(1+η)). |
| `biased_x` | `PAULI_CHANNEL_1(px,py,pz)` | X-biased channel (mirror of `biased`): px = p·η/(1+η), py = pz = p/(2(1+η)). |
| `x_only` | `X_ERROR(p)` | Pure bit-flip noise: only X errors at rate *p*. Only Z-type syndromes fire. |
| `z_only` | `Z_ERROR(p)` | Pure phase-flip noise: only Z errors at rate *p*. Only X-type syndromes fire. |
| `y_only` | `Y_ERROR(p)` | Pure Y errors at rate *p*. Both X- and Z-type syndromes fire (Y = XZ). |
| `pauli` | `PAULI_CHANNEL_1(px,py,pz)` | Fully configurable Pauli channel. Set `--px-frac`, `--py-frac`, `--pz-frac` (normalised automatically). Default is equal thirds = `depolarizing`. |

### Python usage

```python
code = nmwpm.ToricCode(L=8)

# Pure Z (phase-flip) noise
ex, ez, synd = code.sample(1000, p=0.10, noise="z_only", seed=0)

# X-biased noise, η = 100 (almost all errors are X-type)
ex, ez, synd = code.sample(1000, p=0.12, noise="biased_x", eta=100, seed=0)

# Custom Pauli channel: 50% X, 30% Y, 20% Z (normalised to total rate p)
ex, ez, synd = code.sample(1000, p=0.10, noise="pauli",
                            px_frac=0.5, py_frac=0.3, pz_frac=0.2, seed=0)
```

### Threshold p-ranges by noise type

These are the p ranges used in `run_all.sh` and recommended for training:

| Noise | Training `--p-min` | Training `--p-max` | Evaluation p range |
|-------|-------|-------|-------|
| `depolarizing` | 0.05 | 0.20 | 0.08 – 0.20 |
| `independent` | 0.03 | 0.11 | 0.04 – 0.11 |
| `x_only` / `z_only` | 0.03 | 0.14 | 0.05 – 0.17 |
| `y_only` | 0.03 | 0.11 | 0.04 – 0.12 |
| `biased` / `biased_x` (η=10) | 0.05 | 0.18 | 0.08 – 0.20 |
| `biased` / `biased_x` (η=100) | 0.05 | 0.18 | 0.08 – 0.20 |
| `pauli` (equal X+Y) | 0.03 | 0.12 | 0.04 – 0.11 |

---

## Comprehensive run script

`run_all.sh` is a fully resumable bash script that trains and evaluates NMWPM
models across all noise types, a large range of code distances, and multiple
neural-network sizes.

```
bash run_all.sh [all | train | eval]
```

**Key environment variables** (set before running):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | PyTorch device. |
| `EPOCHS` | `400` | Target training epochs per model. |
| `SHOTS` | `50000` | Evaluation shots per error-rate point. |
| `CHUNK` | `512` | NN batch size for evaluation. |
| `CKPT_DIR` | `checkpoints` | Directory for checkpoint files. |
| `RESULTS_DIR` | `results` | Directory for evaluation CSV files. |
| `PYTHON` | auto | Python interpreter path. |

**Resuming**: the script stores the completed epoch count inside every
checkpoint.  Re-running the script after an interruption automatically resumes
training from the last completed epoch and skips already-evaluated models.

**Experiments covered**:

| Block | What is swept |
|-------|--------------|
| Toric standard | L ∈ {4, 6, 8, 10, 12} × 10 noise types × default model |
| Rotated standard | L ∈ {3, 5, 7, 9, 11} × 10 noise types × default model |
| Width ablation | L=8 toric, depolarizing × hidden ∈ {32, 64, 96, 256, 512} |
| GNN depth ablation | L=8 toric, depolarizing × gnn_layers ∈ {0, 2, 4, 8} |
| Transformer depth ablation | L=8 toric, depolarizing × enc_layers ∈ {0, 2, 6} |
| Rotated size ablation | L=7 rotated, depolarizing × hidden ∈ {32, 64, 256} |
| Cross-distance size | L ∈ {4, 6, 10, 12} toric × hidden ∈ {64, 256} |

```bash
# Run everything (train then evaluate all combinations)
DEVICE=cuda EPOCHS=400 bash run_all.sh

# Training pass only (skip evaluation)
bash run_all.sh train

# Evaluation pass only (skip training, assumes checkpoints exist)
SHOTS=50000 bash run_all.sh eval

# Quick smoke test (1 epoch per model, 500 eval shots)
EPOCHS=1 SHOTS=500 DEVICE=cpu bash run_all.sh
```

---

## Pretrained checkpoints

Pre-trained checkpoints are in the `checkpoints/` directory. Each checkpoint embeds its own code configuration, so `Decoder.from_checkpoint()` requires no extra arguments.

| File | Code | L | Noise |
|------|------|---|-------|
| `nmwpm_toric_L6_depolarizing.pt` | Toric | 6 | Depolarizing |
| `nmwpm_toric_L8_depolarizing.pt` | Toric | 8 | Depolarizing |
| `nmwpm_toric_L10_depolarizing.pt` | Toric | 10 | Depolarizing |
| `nmwpm_toric_L12_depolarizing.pt` | Toric | 12 | Depolarizing |
| `nmwpm_toric_L6_independent.pt` | Toric | 6 | Independent |
| `nmwpm_toric_L8_independent.pt` | Toric | 8 | Independent |
| `nmwpm_toric_L10_independent.pt` | Toric | 10 | Independent |
| `nmwpm_toric_L12_independent.pt` | Toric | 12 | Independent |
| `nmwpm_toric_L8_biased_eta10.pt` | Toric | 8 | Biased η=10 |
| `nmwpm_toric_L8_biased_eta100.pt` | Toric | 8 | Biased η=100 |
| `nmwpm_toric_L10_biased_eta10.pt` | Toric | 10 | Biased η=10 |
| `nmwpm_toric_L10_biased_eta100.pt` | Toric | 10 | Biased η=100 |
| `nmwpm_toric_L8_d32.pt` | Toric | 8 | Depolarizing (hidden_dim=32) |
| `nmwpm_toric_L8_d64.pt` | Toric | 8 | Depolarizing (hidden_dim=64) |
| `nmwpm_toric_L8_enconly.pt` | Toric | 8 | Depolarizing (encoder only, no GNN) |
| `nmwpm_toric_L8_gnnonly.pt` | Toric | 8 | Depolarizing (GNN only, no encoder) |
| `nmwpm_toric_L8_h1.pt` | Toric | 8 | Depolarizing (1 attention head) |
| `nmwpm_rotated_L5_depolarizing.pt` | Rotated | 5 | Depolarizing |
| `nmwpm_rotated_L7_depolarizing.pt` | Rotated | 7 | Depolarizing |
| `nmwpm_rotated_L9_depolarizing.pt` | Rotated | 9 | Depolarizing |
| `nmwpm_rotated_L5_independent.pt` | Rotated | 5 | Independent |
| `nmwpm_rotated_L7_independent.pt` | Rotated | 7 | Independent |
| `nmwpm_rotated_L9_independent.pt` | Rotated | 9 | Independent |

---

## Training from scratch

The full training loop generates ground-truth MWPM labels on the fly and trains the QWP model with binary cross-entropy + entropy regularisation:

```python
from nmwpm.train import sample_labeled_training_batch
from nmwpm.model import build_batch
import nmwpm, torch, numpy as np

code  = nmwpm.ToricCode(L=8)
model = nmwpm.QWP(code, hidden_dim=128).to("cuda")
opt   = torch.optim.Adam(model.parameters(), lr=9e-5)
rng   = np.random.default_rng(0)

for step in range(1000):
    batch = sample_labeled_training_batch(
        code, batch_size=32, p=0.12,
        noise="depolarizing", rng=rng, gt_timeout=10.0, eta=10.0)
    if batch is None:
        continue
    syndromes, edge_lists, labels = batch
    tensors   = build_batch(code, syndromes, edge_lists, "cuda")
    edge_mask = tensors[-1]
    y = torch.zeros_like(edge_mask, dtype=torch.float32)
    for b, lab in enumerate(labels):
        y[b, :len(lab)] = torch.as_tensor(lab, device="cuda")
    logits = model(*tensors)
    loss   = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction="sum") / edge_mask.float().sum()
    opt.zero_grad(); loss.backward(); opt.step()
```

Or use the CLI:

```bash
nmwpm-train --code toric --L 8 --noise depolarizing --epochs 400 --out model.pt
```

---

## Reproducing paper results

All results in the paper were generated with the pretrained checkpoints and the `nmwpm-evaluate` CLI.

```bash
# Toric code depolarizing threshold (L = 6, 8, 10, 12)
for L in 6 8 10 12; do
    nmwpm-evaluate --code toric --L $L --noise depolarizing \
        --ckpt checkpoints/nmwpm_toric_L${L}_depolarizing.pt \
        --p 0.10 0.12 0.14 0.16 0.18 0.20 \
        --shots 50000 --csv results/toric_depolarizing_L${L}.csv
done

# Estimate threshold from saved results
nmwpm-evaluate --threshold results/toric_depolarizing_L8.csv

# Rotated surface code independent noise
for L in 5 7 9; do
    nmwpm-evaluate --code rotated --L $L --noise independent \
        --ckpt checkpoints/nmwpm_rotated_L${L}_independent.pt \
        --p 0.10 0.13 0.16 0.19 \
        --shots 50000 --csv results/rotated_independent_L${L}.csv
done

# Biased noise η = 100
nmwpm-evaluate --code toric --L 8 --noise biased --eta 100 \
    --ckpt checkpoints/nmwpm_toric_L8_biased_eta100.pt \
    --p 0.10 0.20 0.30 0.40 --shots 50000

# Latency benchmark
nmwpm-evaluate --code toric --L 8 \
    --ckpt checkpoints/nmwpm_toric_L8_depolarizing.pt \
    --benchmark --reps 500
```

Pre-computed CSV results are in the `results/` directory.

---

## Background

NMWPM extends the classical **Minimum-Weight Perfect Matching** (MWPM) decoder by replacing its fixed, distance-based edge weights with weights learned by a neural network. For a syndrome measurement on a CSS surface code:

1. **Syndrome graph**: Active stabilizers (defects) become graph nodes; all possible pairwise connections become candidate edges.
2. **QWP scoring**: The `QWP` network — a graph-transformer GNN followed by a Transformer encoder — reads the syndrome and assigns a probability to each edge indicating whether that edge belongs to the true matching.
3. **MWPM decoding**: `pymatching` solves the weighted perfect-matching problem on the scored graph. The resulting matching is decoded to a logical error parity via homology.

The key advantage over standard MWPM is that the neural network can capture **correlated errors** (e.g. Y errors triggering both X and Z stabilizers simultaneously), which fixed-weight MWPM treats as independent events.

For full details see the paper abstracts in `abstracts/`.