#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent

# Each target is pinned to the ABC commit its patch was written against.
#   labels: SAT-proved equivalence labels for training (&fraig -U)
#   gnn:    &fraig -e / &cec -e with SECL embeddings
TARGETS = {
    "labels": {
        "commit": "7fbcde9d2297d32a8f0ad761137958ac043d2cd1",
        "sha256": "aa8fee65f71607039ecbc27549a0ee86300265f244647eccebb2de67e2c535eb",
    },
    "gnn": {
        "commit": "03b6f0a4f94b55e9b2b502399d1a6c8761fa6ba5",
        "sha256": "32773aeda9b4470acd890283a5a8b14b55aa01d14d53555d41c77cf7453cf97d",
    },
}
GNN_SOURCES = ("cecGnn.h", "cecGnn.c", "cecGnnOrder.c")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(target, archive):
    spec = TARGETS[target]
    if archive is None:
        archive = ROOT / target / "upstream.tar.gz"
        if not archive.exists():
            url = f"https://codeload.github.com/berkeley-abc/abc/tar.gz/{spec['commit']}"
            print(f"Downloading ABC {spec['commit']}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response:
                archive.write_bytes(response.read())
    elif not archive.exists():
        raise FileNotFoundError(f"Archive does not exist: {archive}")
    if sha256(archive) != spec["sha256"]:
        raise RuntimeError(f"Checksum mismatch for {archive}")
    return archive


def extract(archive, destination):
    with tarfile.open(archive) as tar:
        for member in tar:
            parts = Path(member.name).parts[1:]  # drop the top-level abc-<commit>/
            if not parts or not member.isfile():
                continue
            if ".." in parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            path = destination.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tar.extractfile(member).read())
            path.chmod(member.mode)


def build(target, jobs, archive=None, build_dir=None):
    spec = TARGETS[target]
    patch = ROOT / target / "upstream.patch"
    tree = (build_dir or ROOT / "build") / target
    identity = {"upstream_commit": spec["commit"], "archive_sha256": spec["sha256"],
                "patch_sha256": sha256(patch)}

    # Reuse an existing tree only if it came from the same archive and patch.
    stamp = tree / "SECL_ORIGIN.json"
    if not (stamp.exists() and json.loads(stamp.read_text()) == identity):
        if tree.exists():
            shutil.rmtree(tree)
        tree.mkdir(parents=True)
        extract(fetch(target, archive), tree)
        subprocess.run(["patch", "-p1", "--batch", "--forward", "--fuzz=0", "-i", str(patch)],
                       cwd=tree, check=True)
        stamp.write_text(json.dumps(identity, indent=2) + "\n")

    if target == "gnn":
        for name in GNN_SOURCES:
            shutil.copy2(ROOT / "gnn" / name, tree / "src/proof/cec" / name)
        (tree / "src/proof/cec/cecGnnConfig.h").write_text(
            "#define ABC_GNN_SCRIPT " + json.dumps(str(ROOT / "gnn/infer.py")) + "\n")

    log_path = tree / "build.log"
    print(f"Building {target}; log: {log_path}", flush=True)
    with log_path.open("w") as log:
        result = subprocess.run(["make", f"-j{jobs}", "ABC_USE_NO_READLINE=1", "OPTFLAGS=-O3"],
                                cwd=tree, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"ABC build failed; see {log_path}")
    print(tree / "abc", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build the patched ABC binaries used by SECL.")
    parser.add_argument("target", choices=sorted(TARGETS))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--archive", type=Path, help="Local copy of the pinned ABC archive")
    parser.add_argument("--build-dir", type=Path, help="Default: abc/build")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    for command in ("make", "gcc", "g++", "patch"):
        if shutil.which(command) is None:
            parser.error(f"Required build tool is missing: {command}")
    build(args.target, args.jobs, args.archive and args.archive.resolve(),
          args.build_dir and args.build_dir.resolve())


if __name__ == "__main__":
    main()
