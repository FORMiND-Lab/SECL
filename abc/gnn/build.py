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
COMMIT = "03b6f0a4f94b55e9b2b502399d1a6c8761fa6ba5"
ARCHIVE_SHA256 = "32773aeda9b4470acd890283a5a8b14b55aa01d14d53555d41c77cf7453cf97d"


def build(destination, jobs, archive=None):
    local_archive = archive is not None
    archive = Path(archive) if local_archive else ROOT / "upstream.tar.gz"
    if not archive.exists():
        if local_archive:
            raise FileNotFoundError(f"Archive does not exist: {archive}")
        url = f"https://codeload.github.com/berkeley-abc/abc/tar.gz/{COMMIT}"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("Upstream archive checksum mismatch")
    patch_hash = hashlib.sha256((ROOT / "upstream.patch").read_bytes()).hexdigest()
    identity = {"upstream_commit": COMMIT, "archive_sha256": ARCHIVE_SHA256,
                "patch_sha256": patch_hash}
    if destination.exists():
        if json.loads((destination / "GNN_ORIGIN.json").read_text()) != identity:
            raise RuntimeError("Build source identity differs; use a new --source directory")
    else:
        destination.mkdir(parents=True)
        with tarfile.open(archive) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                path = destination / Path(*Path(member.name).parts[1:])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tar.extractfile(member).read())
                path.chmod(member.mode)
        subprocess.run(["patch", "-p1", "--input", str(ROOT / "upstream.patch")],
                       cwd=destination, check=True)
        (destination / "GNN_ORIGIN.json").write_text(json.dumps(identity, indent=2) + "\n")
    for name in ("cecGnn.h", "cecGnn.c", "cecGnnOrder.c"):
        shutil.copy2(ROOT / name, destination / "src/proof/cec" / name)
    (destination / "src/proof/cec/cecGnnConfig.h").write_text(
        "#define ABC_GNN_SCRIPT " + json.dumps(str(ROOT / "infer.py")) + "\n")
    with (ROOT / "build.log").open("w") as log:
        subprocess.run(["make", f"-j{jobs}", "ABC_USE_NO_READLINE=1", "OPTFLAGS=-O3"],
                       cwd=destination, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(destination / "abc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build GNN commands from the pinned official ABC archive and the local patch.')
    parser.add_argument("--source", type=Path, default=ROOT.parent / "abc-gnn")
    parser.add_argument("--archive", type=Path, help="Use a local pinned ABC archive")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build(args.source.resolve(), args.jobs, args.archive)
