# SECL in ABC

The `&fraig -e` and `&cec -e` commands use DeepGate2 embeddings to prioritize
SAT queries for logic synthesis and combinational equivalence checking.
Equivalence decisions remain justified by SAT proofs or structural hashing.

## Build

From the repository root:

```bash
python abc/build.py gnn --jobs 8
```

The script downloads the pinned ABC source, applies `upstream.patch`, copies the
`cecGnn*` sources and builds `abc/build/gnn/abc`. Use `--archive PATH` to build
from a local copy of the archive.

## Usage

Use the released `pretrained/DeepGate2.ckpt`, or an encoder checkpoint exported
by `src/train.py`. The synthesis example uses a circuit from `data/example/`;
replace the equivalence-checking input paths with your own circuits:

```bash
# Logic synthesis
mkdir -p results
abc/build/gnn/abc -c '&r data/example/EPFL__bar__s073.aig; &fraig -e DeepGate2 pretrained/DeepGate2.ckpt -Y python -U cuda:0; &w results/reduced.aig'

# Equivalence checking
abc/build/gnn/abc -c '&cec -e DeepGate2 pretrained/DeepGate2.ckpt -Y python -U cuda:0 data/left.aig data/right.aig'

# Check an already constructed miter
abc/build/gnn/abc -c '&r data/miter.aig; &cec -e DeepGate2 pretrained/DeepGate2.ckpt -Y python -U cuda:0 -m'
```

Inputs must be combinational AIGs. Two-circuit verification requires matching
ordered PI/PO interfaces. Both exported encoder checkpoints and full Lightning
training checkpoints are accepted.

| Option | Description |
|---|---|
| `-e DeepGate2 CHECKPOINT` | Enable SECL with the specified weights |
| `-Y PYTHON` | Python interpreter containing the installed dependencies |
| `-U DEVICE` | GPU inference device, e.g. `cuda:0` |
| `-C N` | SAT conflict limit; default `100000` |
| `-G SCRIPT` | Override the inference script recorded at build time |
| `-v` | Verbose ABC output |
| `-m` | For `&cec -e`, use the currently loaded graph as a miter |

CEC reports `status=1` for equivalent, `status=0` for inequivalent, and
`status=-1` for unresolved. Missing inputs, incompatible weights or inference
failures return a command error. Commands without `-e` use the upstream ABC
interface.
