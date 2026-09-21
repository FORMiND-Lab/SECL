import os
import subprocess
import tempfile
import argparse
import shutil
import torch
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.aiger_utils import aig_to_xdata
from utils.aig_schema import GateType

def _quote_abc_path(path):
    path = str(Path(path).resolve())
    if any(c in path for c in '\r\n;"'):
        raise ValueError('ABC paths cannot contain quotes, semicolons or newlines')
    return '"' + path + '"'

def replace_aig_with_gia(gia_aig, aig_path):
    if os.path.exists(gia_aig):
        Path(aig_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(gia_aig, aig_path)

def run_abc_fraig(abc_path, aig_path, timeout):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        temp_file = tmp.name
    gia_aig = temp_file.replace('.txt', '_gia.aig')

    try:
        abc_cmd = (f"read {_quote_abc_path(aig_path)}; &get; &write {_quote_abc_path(gia_aig)}; "
                   f"&fraig -U {_quote_abc_path(temp_file)} -C 1000000")
        result = subprocess.run([str(abc_path), "-c", abc_cmd], capture_output=True,
                                text=True, check=True, timeout=timeout)
        raw = Path(temp_file).read_text()
        if not any(line.startswith(('SAT_Proved:', 'SAT_Unknown:', 'No_Proved',
                                    'positive:', 'invpositive:')) for line in raw.splitlines()):
            raise RuntimeError('ABC did not produce a -U label file; use bin/abc_labels.\n'
                               + (result.stdout + result.stderr)[-2000:])
        if not Path(gia_aig).is_file() or not Path(gia_aig).stat().st_size:
            raise RuntimeError('ABC did not write the GIA-ordered AIG')

        pos_classes, inv_classes, pairs, unknown_nodes = [], [], [], []
        if os.path.exists(temp_file):
            pos_classes, inv_classes, pairs, unknown_nodes = parse_fraig_output(temp_file)
        return pos_classes, inv_classes, pairs, unknown_nodes, gia_aig
    except BaseException:
        Path(gia_aig).unlink(missing_ok=True)
        raise
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def normalize_equiv_class(repr_id, nodes):
    unique_nodes = sorted(set(int(n) for n in nodes if int(n) >= 0))
    if repr_id not in unique_nodes:
        unique_nodes = sorted(set(unique_nodes + [int(repr_id)]))
    if len(unique_nodes) < 2:
        return None
    return int(repr_id), unique_nodes

def normalize_classes(classes):
    normalized = []
    seen = set()
    for repr_id, nodes in classes:
        item = normalize_equiv_class(repr_id, nodes)
        if item is None:
            continue
        key = (item[0], tuple(item[1]))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return sorted(normalized)

def parse_fraig_output(path):
    pos_classes = []
    inv_classes = []
    pairs = []
    unknown_nodes = []
    current_mode = None

    with open(path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line == 'No_Proved':
                continue
            if line == 'positive:':
                current_mode = 'pos'
                continue
            if line == 'invpositive:':
                current_mode = 'inv'
                continue
            if line == 'none':
                continue
            if line.startswith('SAT_Proved:'):
                parts = line.replace('SAT_Proved:', '').split('==')
                if len(parts) == 2:
                    try:
                        pairs.append((int(parts[0].strip()), int(parts[1].strip())))
                    except ValueError:
                        pass
                continue
            if line.startswith('SAT_Unknown:'):
                parts = line.replace('SAT_Unknown:', '').split('?=')
                if len(parts) == 2:
                    try:
                        unknown_nodes.append(int(parts[0].strip()))
                    except ValueError:
                        pass
                continue
            if line.startswith('Class['):
                head, *tail = line.split()
                try:
                    repr_id = int(head[len('Class['):-1])
                    nodes = [int(x) for x in tail]
                except ValueError:
                    continue
                if current_mode == 'inv':
                    inv_classes.append((repr_id, nodes))
                else:
                    pos_classes.append((repr_id, nodes))

    return (
        normalize_classes(pos_classes),
        normalize_classes(inv_classes),
        pairs,
        sorted(set(unknown_nodes)),
    )

def simulate_nodes(x_data, edge_index, nodes_to_sim, num_patterns=16384, device='cuda'):

    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'

    n_total = len(x_data)

    fanin = [[] for _ in range(n_total)]
    fanout = [[] for _ in range(n_total)]
    types = [-1] * n_total

    for i, info in enumerate(x_data):
        types[i] = int(info[1])

    for src, dst in edge_index:
        s = int(src)
        d = int(dst)
        if d < n_total:
            fanin[d].append(s)
            fanout[s].append(d)


    in_deg = [len(fanin[i]) for i in range(n_total)]
    q = [i for i in range(n_total) if in_deg[i] == 0]
    idx = 0
    while idx < len(q):
        u = q[idx]
        idx += 1
        for v in fanout[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)


    states = torch.zeros((n_total, num_patterns), dtype=torch.bool, device=device)

    pi_idx = [i for i in range(n_total) if types[i] == GateType.PI]
    if pi_idx:
        states[pi_idx] = (torch.rand((len(pi_idx), num_patterns), device=device) > 0.5)

    for u in q:
        t = types[u]
        if t == GateType.CONST:
            states[u].zero_()
        elif t == GateType.AND:
            fins = fanin[u]
            if len(fins) >= 2:
                res = states[fins[0]] & states[fins[1]]
                for i in range(2, len(fins)):
                    res &= states[fins[i]]
                states[u] = res
            elif len(fins) == 1:
                states[u] = states[fins[0]]
        elif t == GateType.NOT:
            if fanin[u]:
                states[u] = ~states[fanin[u][0]]


    results = {}
    for node in nodes_to_sim:
        if 0 <= node < n_total:
            results[node] = states[node]
    return results

def group_classes(pairs, signatures, circuit_name=""):

    adj = defaultdict(list)
    nodes_set = set()
    for u, v in pairs:
        adj[u].append(v)
        adj[v].append(u)
        nodes_set.add(u)
        nodes_set.add(v)

    visited = set()
    all_components = []


    for node in sorted(list(nodes_set)):
        if node not in visited:
            component = []
            stack = [node]
            visited.add(node)
            while stack:
                u = stack.pop()
                component.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            all_components.append(sorted(component))


    positive_results = []
    invpositive_results = []

    for comp in all_components:
        repr_node = comp[0]
        if repr_node not in signatures:
            raise RuntimeError(
                f"[{circuit_name}] Missing simulation signature for representative node {repr_node}."
            )
        sig_repr = signatures[repr_node]

        pos_group = [repr_node]
        inv_group = [repr_node]

        for n in comp[1:]:
            if n not in signatures:
                raise RuntimeError(
                    f"[{circuit_name}] Missing simulation signature for node {n} in class of repr {repr_node}."
                )
            sig_n = signatures[n]


            if torch.equal(sig_repr, sig_n):
                pos_group.append(n)
            elif torch.equal(sig_repr, ~sig_n):
                inv_group.append(n)
            else:
                hd = torch.mean((sig_repr ^ sig_n).float()).item()
                raise RuntimeError(
                    f"[{circuit_name}] Invalid equivalence class from proved pairs: "
                    f"repr={repr_node}, node={n}, hamming={hd:.6f}. "
                    "Signatures are neither identical nor fully inverted."
                )

        if len(pos_group) > 1:
            item = normalize_equiv_class(repr_node, sorted(pos_group))
            if item is not None:
                positive_results.append(item)
        if len(inv_group) > 1:
            item = normalize_equiv_class(repr_node, sorted(inv_group))
            if item is not None:
                invpositive_results.append(item)

    return normalize_classes(positive_results), normalize_classes(invpositive_results)

def split_classes_by_signatures(classes, signatures, circuit_name=""):

    positive_results = []
    invpositive_results = []

    for repr_node, nodes in classes:
        if repr_node not in signatures:
            raise RuntimeError(
                f"[{circuit_name}] Missing simulation signature for class representative {repr_node}."
            )
        sig_repr = signatures[repr_node]
        pos_group = [repr_node]
        inv_group = [repr_node]

        for n in nodes:
            if n == repr_node:
                continue
            if n not in signatures:
                raise RuntimeError(
                    f"[{circuit_name}] Missing simulation signature for node {n} in class of repr {repr_node}."
                )
            sig_n = signatures[n]
            if torch.equal(sig_repr, sig_n):
                pos_group.append(n)
            elif torch.equal(sig_repr, ~sig_n):
                inv_group.append(n)
            else:
                hd = torch.mean((sig_repr ^ sig_n).float()).item()
                raise RuntimeError(
                    f"[{circuit_name}] Invalid class from proved labels: "
                    f"repr={repr_node}, node={n}, hamming={hd:.6f}. "
                    "Signatures are neither identical nor fully inverted."
                )

        if len(pos_group) > 1:
            item = normalize_equiv_class(repr_node, sorted(pos_group))
            if item is not None:
                positive_results.append(item)
        if len(inv_group) > 1:
            item = normalize_equiv_class(repr_node, sorted(inv_group))
            if item is not None:
                invpositive_results.append(item)

    return normalize_classes(positive_results), normalize_classes(invpositive_results)

def _process_one(task):
    aig_name, aig_dir, abc_path, patterns, timeout, aig_out = task
    torch.set_num_threads(1)
    torch.manual_seed(42)
    circuit_name = Path(aig_name).with_suffix('').as_posix()
    aig_path = os.path.join(aig_dir, aig_name)
    output_aig = os.path.join(aig_out, aig_name)

    try:
        pos_classes, inv_classes, pairs, unknown_nodes, gia_aig = run_abc_fraig(
            abc_path, aig_path, timeout
        )
    except Exception as e:
        return circuit_name, None, None, None, f'error:{e}'

    if not (pos_classes or inv_classes or pairs):
        if not os.path.exists(gia_aig):
            return circuit_name, None, None, None, 'error:ABC did not write the GIA AIG'
        replace_aig_with_gia(gia_aig, output_aig)
        return circuit_name, [], [], unknown_nodes, 'ok_no_classes'


    try:
        x_data, edge_index = aig_to_xdata(gia_aig)
    except Exception as e:
        if os.path.exists(gia_aig):
            os.remove(gia_aig)
        return circuit_name, None, None, None, f'error:{e}'

    if pos_classes or inv_classes:
        nodes_to_sim = {n for _, nodes in (pos_classes + inv_classes) for n in nodes}
    else:
        nodes_to_sim = {u for u, v in pairs} | {v for u, v in pairs}
    try:
        signatures = simulate_nodes(x_data, edge_index, nodes_to_sim,
                                    num_patterns=patterns, device='cpu')
        if pos_classes or inv_classes:
            pos_from_pos, inv_from_pos = split_classes_by_signatures(
                pos_classes, signatures, circuit_name=circuit_name
            )
            pos_from_inv, inv_from_inv = split_classes_by_signatures(
                inv_classes, signatures, circuit_name=circuit_name
            )
            pos_classes = normalize_classes(pos_from_pos + pos_from_inv)
            inv_classes = normalize_classes(inv_from_pos + inv_from_inv)
        else:
            pos_classes, inv_classes = group_classes(pairs, signatures, circuit_name=circuit_name)
    except Exception as e:
        if os.path.exists(gia_aig):
            os.remove(gia_aig)
        return circuit_name, None, None, None, f'error:{e}'

    if not pos_classes and not inv_classes:
        replace_aig_with_gia(gia_aig, output_aig)
        return circuit_name, [], [], unknown_nodes, 'ok_no_classes'

    replace_aig_with_gia(gia_aig, output_aig)

    return circuit_name, pos_classes, inv_classes, unknown_nodes, 'ok'


def main():
    parser = argparse.ArgumentParser(description='Extract signed equivalence labels and GIA-aligned AIG copies.')
    parser.add_argument('--aig_dir', type=str, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--aig_out', type=Path, help='Aligned AIG output directory (default: next to --out, named aig)')
    parser.add_argument('--abc', type=str, default=str(SRC_ROOT.parent / 'bin/abc_labels'))
    parser.add_argument('--patterns', type=int, default=1600)
    parser.add_argument('--timeout', type=int, default=500,
                        help='Per-circuit ABC &fraig timeout in seconds')
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() // 2))
    args = parser.parse_args()

    if min(args.patterns, args.timeout, args.workers) < 1:
        parser.error('patterns, timeout and workers must be positive')
    args.aig_dir = Path(args.aig_dir).resolve()
    output_path = args.out.resolve()
    aig_out = (args.aig_out or output_path.parent / 'aig').resolve()
    if not args.aig_dir.is_dir():
        parser.error('The AIG input directory does not exist')
    if aig_out == args.aig_dir or args.aig_dir in aig_out.parents or aig_out in args.aig_dir.parents:
        parser.error('--aig_out must be separate from the source AIG directory')
    if output_path.exists() or (aig_out.exists() and any(aig_out.iterdir())):
        parser.error('Use a new label file and a new or empty aligned-AIG directory')
    binary = shutil.which(args.abc)
    if binary is None:
        parser.error('Label ABC was not found; run python scripts/build_abc.py')
    args.abc = str(Path(binary).resolve())
    aig_files = sorted(p.relative_to(args.aig_dir).as_posix() for p in args.aig_dir.rglob('*.aig') if p.is_file())
    if not aig_files:
        parser.error('No .aig files found')
    if any(any(c.isspace() for c in name) for name in aig_files):
        parser.error('Relative circuit names cannot contain whitespace in the label format')
    total = len(aig_files)
    print(f"Found {total} AIG files, using {args.workers} workers")

    n_ok = n_empty = n_err = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aig_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.secl-labels-', dir=aig_out.parent) as staging:
        stage = Path(staging)
        stage_aig = stage / 'aig'
        stage_aig.mkdir()
        temporary_output = stage / 'proved.txt'
        tasks = [(f, args.aig_dir, args.abc, args.patterns, args.timeout, stage_aig) for f in aig_files]
        with open(temporary_output, 'w') as f_out, mp.Pool(min(args.workers, total)) as pool:
            for done, (circuit_name, pos_classes, inv_classes, unknown_nodes, action) in enumerate(
                    pool.imap(_process_one, tasks), 1):
                if action in {'ok', 'ok_no_classes'}:
                    n_ok += 1
                    if action == 'ok_no_classes':
                        n_empty += 1
                    f_out.write(f"{circuit_name}\n")
                    f_out.write("positive:\n")
                    if pos_classes:
                        for repr_id, nodes in sorted(pos_classes):
                            f_out.write(f"Class[{repr_id}] " + " ".join(map(str, nodes)) + "\n")
                    else:
                        f_out.write("none\n")
                    f_out.write("invpositive:\n")
                    if inv_classes:
                        for repr_id, nodes in sorted(inv_classes):
                            f_out.write(f"Class[{repr_id}] " + " ".join(map(str, nodes)) + "\n")
                    else:
                        f_out.write("none\n")
                    f_out.write("unknown:\n")
                    if unknown_nodes:
                        f_out.write("Nodes " + " ".join(map(str, unknown_nodes)) + "\n")
                    else:
                        f_out.write("none\n")
                    f_out.write("\n")
                else:
                    n_err += 1
                    print(f"[{done}/{total}] Error {circuit_name}: {action}", flush=True)

                if done % 100 == 0:
                    print(f"Progress: {done}/{total} (ok={n_ok}, empty={n_empty}, err={n_err})",
                          flush=True)

        if n_err:
            raise RuntimeError(
                f"Label generation failed for {n_err} circuit(s); no label dataset was published"
            )
        if aig_out.exists():
            aig_out.rmdir()
        stage_aig.replace(aig_out)
        shutil.move(temporary_output, output_path)
    print(
        f"\nDone. {n_ok} written ({n_empty} without proved classes), "
        f"0 errors → {output_path}"
    )

if __name__ == "__main__":
    main()
