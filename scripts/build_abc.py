#!/usr/bin/env python3

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "7fbcde9d2297d32a8f0ad761137958ac043d2cd1"
ARCHIVE_SHA256 = "aa8fee65f71607039ecbc27549a0ee86300265f244647eccebb2de67e2c535eb"
PATCH = ROOT / "third_party/abc_labels.patch"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(payload, destination):
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts:
                continue
            if ".." in parts or member.issym() or member.islnk() or Path(*parts).is_absolute():
                raise ValueError("Unsafe ABC archive member")
            if member.isfile():
                target = destination.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
                target.chmod(member.mode)


def load_archive(archive, *, local_only=False):
    if not archive.exists():
        if local_only:
            raise FileNotFoundError(f"The supplied archive does not exist: {archive}")
        url = f"https://codeload.github.com/berkeley-abc/abc/tar.gz/{COMMIT}"
        print(f"Downloading ABC {COMMIT}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
            raise RuntimeError("ABC archive checksum mismatch")
        archive.write_bytes(payload)
    payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("ABC archive checksum mismatch")
    return payload


def main():
    parser = argparse.ArgumentParser(description='Build the pinned ABC extractor used for SECL training labels.')
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--archive", type=Path, help="Local pinned archive; SHA256 is checked")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "bin")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    for command in ("make", "gcc", "g++", "patch"):
        if shutil.which(command) is None:
            parser.error(f"Required build tool is missing: {command}")
    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = load_archive(args.archive or cache / "abc-label.tar.gz",
                           local_only=args.archive is not None)
    tree = cache / "abc_labels"
    if tree.exists():
        shutil.rmtree(tree)
    extract(payload, tree)
    subprocess.run(["patch", "-p1", "--batch", "--forward", "--fuzz=0", "-i", str(PATCH)],
                   cwd=tree, check=True)
    log_path = cache / "build_labels.log"
    flags = [f"-j{args.jobs}", "ABC_USE_NO_READLINE=1", "OPTFLAGS=-O3"]
    print(f"Building label extractor; log: {log_path}", flush=True)
    with log_path.open("w") as log:
        result = subprocess.run(["make", *flags], cwd=tree, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"ABC build failed; see {log_path}")
    binary = output / "abc_labels"
    shutil.copy2(tree / "abc", binary)
    manifest = {
        "upstream_commit": COMMIT, "archive_sha256": ARCHIVE_SHA256,
        "patch_sha256": sha(PATCH), "make_flags": flags,
        "binary_sha256": sha(binary),
        "compiler": subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0],
    }
    binary.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {binary}", flush=True)


if __name__ == "__main__":
    main()
