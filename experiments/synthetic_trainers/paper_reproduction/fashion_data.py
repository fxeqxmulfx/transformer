"""Fetch checksum-verified Fashion-MNIST and strictly parse the public IDX files."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
from urllib.request import urlopen

import numpy as np


SOURCE = "https://github.com/zalandoresearch/fashion-mnist"
BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
CHECKSUMS = {"train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
             "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
             "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
             "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310"}


def images_from_idx(content):
    if len(content) < 16:
        raise ValueError("Truncated IDX image header")
    magic, count, height, width = struct.unpack(">IIII", content[:16])
    if magic != 2051 or count < 1 or (height, width) != (28, 28) or len(content) != 16 + count * height * width:
        raise ValueError("Invalid IDX image shape, magic, or payload length")
    return np.frombuffer(content, dtype=np.uint8, offset=16).reshape(count, height * width).copy()


def labels_from_idx(content):
    if len(content) < 8:
        raise ValueError("Truncated IDX label header")
    magic, count = struct.unpack(">II", content[:8])
    if magic != 2049 or count < 1 or len(content) != 8 + count:
        raise ValueError("Invalid IDX label magic or payload length")
    labels = np.frombuffer(content, dtype=np.uint8, offset=8).copy()
    if np.any(labels > 9):
        raise ValueError("Fashion-MNIST labels must be in 0..9")
    return labels


def verify(directory):
    result = {}
    for name, expected in CHECKSUMS.items():
        data = (Path(directory) / name).read_bytes()
        if hashlib.md5(data).hexdigest() != expected:
            raise ValueError(f"Published dataset checksum mismatch: {name}")
        result[name] = {"url": BASE + name, "md5": expected, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return result


def download(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in CHECKSUMS.items():
        path = directory / name
        if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == expected:
            continue
        data = urlopen(BASE + name, timeout=60).read()
        if hashlib.md5(data).hexdigest() != expected:
            raise ValueError(f"Published dataset checksum mismatch: {name}")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    manifest = {"source": SOURCE, "files": verify(directory)}
    (directory / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(directory):
    manifest = {"source": SOURCE, "files": verify(directory)}
    def read(name):
        return gzip.decompress((Path(directory) / name).read_bytes())
    train = images_from_idx(read("train-images-idx3-ubyte.gz"))
    train_labels = labels_from_idx(read("train-labels-idx1-ubyte.gz"))
    test = images_from_idx(read("t10k-images-idx3-ubyte.gz"))
    test_labels = labels_from_idx(read("t10k-labels-idx1-ubyte.gz"))
    if len(train) != len(train_labels) or len(train) != 60000 or len(test) != len(test_labels) or len(test) != 10000:
        raise ValueError("Dataset split sizes differ from the published Fashion-MNIST corpus")
    return train, train_labels, test, test_labels, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    manifest = download(args.directory)
    train, _, test, _, _ = load(args.directory)
    print(json.dumps({"source": manifest["source"], "train_examples": len(train), "test_examples": len(test), "directory": str(args.directory)}))


if __name__ == "__main__":
    main()
