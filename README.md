# SECL: Signed Equivalence Contrastive Learning

Code repository for **Boolean Circuit Embeddings for Functional Reasoning via
Signed Equivalence Contrastive Learning**.

SECL learns node-level functional representations from SAT-proved signed
equivalence relations. It uses a PI-aware DeepGate2 encoder with the SECL
objective to align equivalent nodes and separate inequivalent ones.

## Installation

Requires Python 3.9 or newer. For GPU training and inference, install a CUDA
build of PyTorch 2.8 from [pytorch.org](https://pytorch.org/get-started/locally/)
first, then the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Building the patched ABC binaries also needs `gcc`, `g++`, `make` and `patch`.

## Model Training

### 1. Prepare the dataset

The paired AIG dataset contains 38,606 graphs from five benchmarks:
EPFL, ForgeEDA, ITC'99, IWLS 2005 and OpenRISC. The circuits are already
paired through shared PIs.

Download `GnnAbc_cut_pairs.tar.gz` from the
[Releases](https://github.com/FORMiND-Lab/SECL/releases) page and extract it:

```bash
mkdir -p data/dataset
curl -L -o data/GnnAbc_cut_pairs.tar.gz \
  https://github.com/FORMiND-Lab/SECL/releases/latest/download/GnnAbc_cut_pairs.tar.gz
tar -xzf data/GnnAbc_cut_pairs.tar.gz -C data/dataset
```

```text
data/dataset/
  raw_aig/<benchmark>/<source_circuit>/<pair>.aig
```

Build the label extractor and generate training graphs from the paired AIGs:

```bash
python abc/build.py labels --jobs 8
python src/dataset/class_parser.py \
  --aig_dir data/dataset/raw_aig --aig_out data/dataset/aig \
  --out data/dataset/proved.txt --workers 4
python src/dataset/preprocess.py --data_dir data/dataset --workers 4
```

Use fresh label and preprocessing output directories. The generated `aig/`
copies preserve the node mapping used by the labels.

### 2. Train the model

Preprocessing writes one `.pt` file per graph under `data/dataset/processed/`.
Split and pack these graphs yourself into `data/dataset/split/train.pt` and
`data/dataset/split/val.pt` before training. Each file is a `torch.save`
dictionary containing `feature_config` (from `processed/feature_config.json`),
`graphs` (the processed `AIGData` objects), and `graph_stats` (one
`estimate_graph_stats(graph)` result per graph, in the same order). The helper
is defined in [src/dataset/dataset.py](src/dataset/dataset.py).

Training defaults to one GPU. Adjust `--batch_size` and `--num_workers`
to suit your machine.

```bash
python src/train.py --data_dir data/dataset/split \
  --batch_size 128 --num_workers 4 --devices 1 --log_dir results/secl
```

Training uses `src/model/DeepGate2.py` and `src/model/loss.py`. Logs and
checkpoints are saved under `results/secl/`. The best encoder is exported to
`results/secl/checkpoints/DeepGate2.ckpt`.

Use `--checkpoint PATH` to initialize model weights, or
`--resume_checkpoint PATH` to continue an interrupted training run.

## Downstream Applications

A pretrained SECL model is provided in [pretrained/DeepGate2.ckpt](pretrained/DeepGate2.ckpt).

Build ABC with the SECL synthesis and equivalence-checking commands:

```bash
python abc/build.py gnn --jobs 8
```

Example circuits are provided in [data/example](data/example).
Run logic synthesis:

```bash
mkdir -p results
abc/build/gnn/abc -c '&r data/example/EPFL__bar__s073.aig; &fraig -e DeepGate2 pretrained/DeepGate2.ckpt -Y python -U cuda:0; &w results/reduced.aig'
```

Check equivalence between your own circuits with matching PI/PO interfaces:

```bash
abc/build/gnn/abc -c '&cec -e DeepGate2 pretrained/DeepGate2.ckpt -Y python -U cuda:0 data/left.aig data/right.aig'
```

See [abc/gnn/README.md](abc/gnn/README.md) for additional command options.

## Acknowledgements

The encoder is based on [DeepGate2](https://github.com/zshi0616/DeepGate2).
Label extraction and downstream applications build on ABC. Its copyright notice
is included in [abc/ABC_COPYRIGHT.txt](abc/ABC_COPYRIGHT.txt).
