#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from model.DeepGate2 import DeepGate2


def load_model(name, checkpoint):
    if name != "DeepGate2":
        raise ValueError("SECL inference supports only DeepGate2")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Expected a DeepGate2 or Lightning checkpoint")
    if "model_config" in payload:
        return DeepGate2.from_checkpoint(checkpoint).eval()

    config = payload["hyper_parameters"]
    if config["encoder_type"] != name:
        raise ValueError("Requested model does not match checkpoint encoder_type")
    model = DeepGate2(num_rounds=config["deepgate2_num_rounds"],
                     dim_hidden=config["hidden"], pi_init_mode=config["deepgate2_pi_init_mode"])
    state = {key[len("encoder."):]: value for key, value in payload["state_dict"].items()
             if key.startswith("encoder.")}
    model.load_state_dict(state, strict=True)
    return model.eval()


def prepare(graph, device):
    ids = [0] + graph["pis"] + [row[0] for row in graph["ands"]]
    mapping = {node: k for k, node in enumerate(ids)}
    gates = [3] + [0] * len(graph["pis"]) + [1] * len(graph["ands"])
    edges, nots = [], {}

    def literal(lit):
        node = mapping[lit // 2]
        if not lit & 1:
            return node
        if node not in nots:
            nots[node] = len(gates)
            gates.append(2)
            edges.append((node, nots[node]))
        return nots[node]

    for node, a, b in graph["ands"]:
        edges.extend(((literal(a), mapping[node]), (literal(b), mapping[node])))
    successors = [[] for _ in gates]
    degree = [0] * len(gates)
    for a, b in edges:
        successors[a].append(b)
        degree[b] += 1
    queue = [i for i, d in enumerate(degree) if d == 0]
    levels = [0] * len(gates)
    for node in queue:
        for nxt in successors[node]:
            levels[nxt] = max(levels[nxt], levels[node] + 1)
            degree[nxt] -= 1
            if degree[nxt] == 0:
                queue.append(nxt)
    if len(queue) != len(gates):
        raise ValueError("Input graph is cyclic")
    edge = torch.tensor(sorted(edges, key=lambda e: (e[1], e[0])),
                        dtype=torch.long, device=device).reshape(-1, 2).T.contiguous()
    return ids, (torch.tensor(gates, device=device), edge,
                 torch.tensor(levels, device=device),
                 torch.arange(len(gates), device=device))


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = torch.device(args.device)
    model = load_model(args.model, args.checkpoint).to(device)
    graph = json.loads(Path(args.graph).read_text())
    ids, inputs = prepare(graph, device)
    kwargs = {"_compact_schedule": model._compact_fanin_schedule(*inputs[:3])}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward():
        return model(*inputs, **kwargs)

    with torch.no_grad():
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            warm = forward()
            torch.nn.functional.normalize(warm[:len(ids)], dim=1)
            sync()
        start = time.perf_counter()
        values = forward()
        sync()
        forward_seconds = time.perf_counter() - start
        start = time.perf_counter()
        values = torch.nn.functional.normalize(values[:len(ids)], dim=1)
        sync()
        normalization_seconds = time.perf_counter() - start
        array = values.to(device="cpu", dtype=torch.float32).numpy()
    if array.ndim != 2 or array.shape[0] != len(ids) or not np.isfinite(array).all():
        raise ValueError("Invalid model embeddings")

    dense = np.zeros((graph["nodes"], array.shape[1]), dtype="<f4")
    dense[ids] = array
    with open(args.output, "wb") as stream:
        stream.write(struct.pack("<8sIIQdd", b"ABCGNN1\0", graph["nodes"],
                                 array.shape[1], int(graph["fingerprint"], 16),
                                 forward_seconds, normalization_seconds))
        stream.write(dense.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Strict model inference for ABC. Timings exclude load, graph setup and warmup.')
    parser.add_argument("--model", required=True, choices=("DeepGate2",))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
