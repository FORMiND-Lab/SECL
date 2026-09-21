import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '../..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')


while SRC_ROOT in sys.path:
    sys.path.remove(SRC_ROOT)
sys.path.insert(0, SRC_ROOT)

import re
import argparse
import glob
import json
import multiprocessing as mp
import shutil
import tempfile
import time
from pathlib import Path
import torch
import numpy as np

from utils.aig_schema import GateType


_PROBABILITY_HISTOGRAM_EDGES = (
    0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.9, 0.99, 0.999, 0.9999, 1.0
)
def _normalize_class(rep_abc_id, nodes):
    nodes = sorted(set(int(n) for n in nodes if int(n) >= 0))
    if rep_abc_id not in nodes:
        nodes = sorted(set(nodes + [rep_abc_id]))
    if len(nodes) < 2:
        return None
    return int(rep_abc_id), nodes

def _add_class_labels(labels, mode, rep_abc_id, nodes):
    item = _normalize_class(rep_abc_id, nodes)
    if item is None:
        return
    norm_rep, norm_nodes = item
    if mode not in ("pos", "inv"):
        raise ValueError(f"Class[{norm_rep}] appears outside positive/invpositive section")
    for n_abc_id in norm_nodes:
        sign = -1 if mode == "inv" and n_abc_id != norm_rep else 1
        label = (norm_rep, sign)
        previous = labels.get(n_abc_id)
        if previous is not None and previous != label:
            raise ValueError(
                f"Conflicting labels for ABC node {n_abc_id}: {previous} vs {label}"
            )
        labels[n_abc_id] = label

def parse_equivalence_file(file_path, *, return_unknown=False):
    data = {}
    unknown = {}
    current_circuit = None
    current_mode = None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Equivalence file not found: {file_path}")

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == "positive:":
                current_mode = "pos"
                continue
            if line == "invpositive:":
                current_mode = "inv"
                continue
            if line == "unknown:":
                current_mode = "unknown"
                continue
            if line == "none":
                continue
            if line == "No_Proved":
                continue


            if not line.startswith("Class[") and not line.startswith("SAT_Proved:") and not line.startswith("SAT_Unknown:") and " " not in line:
                current_circuit = line
                data[current_circuit] = {}
                unknown[current_circuit] = set()
                continue
            if line.startswith("SAT_Proved:") and current_circuit:
                raise ValueError(
                    f"Raw SAT_Proved pairs in {file_path} do not encode polarity. "
                    "Run dataset/class_parser.py first to produce positive/invpositive classes."
                )
            if line.startswith("SAT_Unknown:") and current_circuit:
                raise ValueError(
                    f"Raw SAT_Unknown pairs in {file_path} must be converted by "
                    "dataset/class_parser.py before preprocessing."
                )
            if line.startswith("Nodes ") and current_circuit:
                if current_mode != "unknown":
                    raise ValueError("Nodes line appears outside unknown section")
                for token in line.split()[1:]:
                    unknown[current_circuit].add(int(token))
                continue


            match = re.search(r'Class\[(\d+)\] (.*)', line)
            if match and current_circuit:
                rep_abc_id = int(match.group(1))
                nodes_str = match.group(2).split()
                nodes = [int(n) for n in nodes_str]

                _add_class_labels(data[current_circuit], current_mode, rep_abc_id, nodes)

    if return_unknown:
        return data, {name: sorted(nodes) for name, nodes in unknown.items()}
    return data

def build_equivalence_tensors(abc_id_to_label, x_data, num_base_nodes):
    x_data_np = np.asarray(x_data)
    num_nodes = len(x_data_np)
    y = torch.arange(num_nodes, dtype=torch.long)
    y_sign = torch.ones(num_nodes, dtype=torch.long)

    for abc_id, (repr_abc_id, sign) in abc_id_to_label.items():
        if not 0 <= abc_id < num_base_nodes:
            raise ValueError(
                f"Label node ABC ID {abc_id} is outside canonical base range "
                f"[0, {num_base_nodes - 1}]"
            )
        if not 0 <= repr_abc_id < num_base_nodes:
            raise ValueError(
                f"Representative ABC ID {repr_abc_id} is outside canonical "
                f"base range [0, {num_base_nodes - 1}]"
            )
        if int(x_data_np[abc_id, 1]) == GateType.NOT:
            raise ValueError(f"Label node {abc_id} points to a synthetic NOT")
        if int(x_data_np[repr_abc_id, 1]) == GateType.NOT:
            raise ValueError(f"Representative {repr_abc_id} points to a synthetic NOT")
        if sign not in (-1, 1):
            raise ValueError(f"Invalid equivalence sign {sign} for ABC node {abc_id}")
        y[abc_id] = repr_abc_id
        y_sign[abc_id] = sign
    return y, y_sign


def build_known_mask(unknown_abc_ids, x_data, num_base_nodes):
    x_data_np = np.asarray(x_data)
    known = torch.ones(len(x_data_np), dtype=torch.bool)
    for abc_id in sorted(set(int(node) for node in unknown_abc_ids)):
        if not 0 <= abc_id < num_base_nodes:
            raise ValueError(
                f"Unknown ABC node ID {abc_id} is outside canonical base range "
                f"[0, {num_base_nodes - 1}]"
            )
        if int(x_data_np[abc_id, 1]) == GateType.NOT:
            raise ValueError(f"Unknown node {abc_id} points to a synthetic NOT")
        known[abc_id] = False
    return known


def _process_circuit(task):
    (
        circuit_name,
        abc_id_to_label,
        unknown_abc_ids,
        aig_dir,
        staging_dir,
        feature_config,
    ) = task
    if Path(circuit_name).is_absolute() or '..' in Path(circuit_name).parts:
        return {'circuit_name': circuit_name, 'error': 'invalid relative circuit name'}
    aig_path = os.path.join(aig_dir, f"{circuit_name}.aig")
    if not os.path.isfile(aig_path):
        return {
            "circuit_name": circuit_name,
            "error": f"missing AIG {aig_path}",
        }

    try:
        torch.set_num_threads(1)
        from utils import aiger_utils, circuit_utils
        from utils.structural_pe import (
            build_functional_signatures,
            build_simulation_classes,
            build_single_channel_features,
            build_structural_tokens,
        )
        from dataset.dataset import AIGData, validate_static_schema

        started = time.perf_counter()
        x_data, edge_index, num_base_nodes = aiger_utils.myaig_to_xdata(
            aig_path
        )
        x_data_np = np.asarray(x_data)
        num_nodes = int(x_data_np.shape[0])
        y, y_sign = build_equivalence_tensors(
            abc_id_to_label, x_data_np, num_base_nodes
        )
        y_known = build_known_mask(
            unknown_abc_ids, x_data_np, num_base_nodes
        )
        gate = torch.tensor(
            x_data_np[:, 1], dtype=torch.long
        ).unsqueeze(1)
        edge_index_pyg = (
            torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            if edge_index
            else torch.empty((2, 0), dtype=torch.long)
        )
        forward_level, forward_index = (
            circuit_utils.return_forward_order_info(edge_index_pyg, num_nodes)
        )
        structural_token = build_structural_tokens(
            gate,
            edge_index_pyg,
            forward_level,
            forward_index,
        )
        functional_signature = build_functional_signatures(
            gate,
            edge_index_pyg,
            forward_level,
            forward_index,
            structural_token,
        )
        sim_class, sim_class_sign = build_simulation_classes(
            functional_signature
        )
        features = build_single_channel_features(
            gate,
            edge_index_pyg,
            forward_level,
            forward_index,
            functional_signature=functional_signature,
            sim_class=sim_class,
            sim_class_sign=sim_class_sign,
        )
        data = AIGData(
            gate=gate,
            edge_index=edge_index_pyg,
            forward_level=forward_level,
            forward_index=forward_index,
            num_nodes=num_nodes,
            num_base_nodes=torch.tensor(
                [num_base_nodes], dtype=torch.long
            ),
            circuit_name=circuit_name,
            y_index=y,
            y_sign=y_sign,
            y_known=y_known,
            **features,
        )
        validate_static_schema(
            [data],
            feature_config,
            source=f"preprocess:{circuit_name}",
        )
        graph_path = Path(staging_dir) / f"{circuit_name}.pt"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, graph_path)

        pi_count = int((gate.reshape(-1) == GateType.PI).sum())
        return {
            "circuit_name": circuit_name,
            "error": None,
            "num_nodes": num_nodes,
            "num_pi": pi_count,
            "seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "circuit_name": circuit_name,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _print_result(ordinal, total, result):
    if result["error"]:
        status = result["error"]
    else:
        status = (
            f"{result['num_nodes']} nodes, {result['seconds']:.2f}s"
        )
    print(
        f"[{ordinal}/{total}] {result['circuit_name']}: {status}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess AIGs and SAT-proved labels into packed graphs."
        )
    )
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--aig_dir', type=str, default=None)
    parser.add_argument('--proved', type=str, default=None)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--clear_out', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    from utils.structural_pe import build_feature_config

    if args.workers < 1:
        raise ValueError("workers must be positive")
    feature_config = build_feature_config()

    data_dir = args.data_dir or os.path.join(
        PROJECT_ROOT, 'data/dataset'
    )
    default_aig_dir = os.path.join(data_dir, 'aig')
    aig_dir = args.aig_dir or (
        default_aig_dir
        if os.path.isdir(default_aig_dir)
        else data_dir
    )
    root_equivalence_file = os.path.join(data_dir, 'proved.txt')
    equivalence_file = args.proved or (
        root_equivalence_file
        if os.path.isfile(root_equivalence_file)
        else os.path.join(data_dir, 'output', 'proved.txt')
    )
    output_dir = args.out or os.path.join(data_dir, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    existing_pt = glob.glob(os.path.join(output_dir, '**', '*.pt'), recursive=True)
    if existing_pt and not args.clear_out:
        raise RuntimeError(
            f"{output_dir} already contains {len(existing_pt)} processed "
            "graph(s). Pass --clear_out so old and new schemas cannot mix."
        )

    print(f"Reading equivalence from: {equivalence_file}")
    circuit_labels_map, circuit_unknown_map = parse_equivalence_file(
        equivalence_file, return_unknown=True
    )
    print(f"Found labels for {len(circuit_labels_map)} circuits.")
    if not circuit_labels_map:
        raise ValueError(
            "No circuit sections were found in proved.txt; refusing to "
            "build an empty dataset"
        )

    parent_dir = os.path.dirname(os.path.abspath(output_dir))
    staging_dir = tempfile.mkdtemp(
        prefix=".gatenexus-preprocess-", dir=parent_dir
    )
    tasks = [
        (
            circuit_name,
            circuit_labels_map[circuit_name],
            circuit_unknown_map.get(circuit_name, []),
            aig_dir,
            staging_dir,
            feature_config,
        )
        for circuit_name in sorted(circuit_labels_map)
    ]
    results = []
    started = time.perf_counter()
    try:
        if args.workers == 1:
            iterator = map(_process_circuit, tasks)
            for ordinal, result in enumerate(iterator, 1):
                results.append(result)
                _print_result(ordinal, len(tasks), result)
        else:
            with mp.Pool(
                args.workers, maxtasksperchild=1
            ) as pool:
                for ordinal, result in enumerate(
                    pool.imap_unordered(_process_circuit, tasks), 1
                ):
                    results.append(result)
                    _print_result(ordinal, len(tasks), result)

        failures = [
            f"{result['circuit_name']}: {result['error']}"
            for result in results
            if result["error"]
        ]
        if failures:
            raise RuntimeError(
                "SECL preprocessing failed for "
                f"{len(failures)} circuit(s):\n"
                + "\n".join(failures[:20])
            )
        if len(results) != len(tasks):
            raise RuntimeError(
                "preprocessing did not return one result per circuit"
            )

        with open(
            os.path.join(staging_dir, 'feature_config.json'),
            'w',
            encoding='utf-8',
        ) as handle:
            json.dump(feature_config, handle, indent=2, sort_keys=True)
        summary = {
            "feature_config": feature_config,
            "num_graphs": len(results),
            "num_nodes": sum(result["num_nodes"] for result in results),
            "workers": args.workers,
            "elapsed_seconds": time.perf_counter() - started,
        }
        with open(
            os.path.join(staging_dir, 'preprocess_stats.json'),
            'w',
            encoding='utf-8',
        ) as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

        if args.clear_out:
            for old_pt in existing_pt:
                os.remove(old_pt)
        for staged_path in Path(staging_dir).rglob('*.pt'):
            target = Path(output_dir) / staged_path.relative_to(staging_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, target)
        for name in ('feature_config.json', 'preprocess_stats.json'):
            os.replace(
                os.path.join(staging_dir, name),
                os.path.join(output_dir, name),
            )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(
        f"\nDone. Processed {len(results)} graphs in "
        f"{time.perf_counter() - started:.2f}s."
    )

if __name__ == "__main__":
    main()
