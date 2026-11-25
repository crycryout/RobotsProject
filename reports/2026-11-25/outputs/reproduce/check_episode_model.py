"""Verify every local file against a pinned Hugging Face model revision before serving it."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REPO = "nvidia/Cosmos3-Edge-Policy-DROID"
REV = "68b17b3c959ccd0999de9a972c5f7c8b57112f86"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--metadata", type=Path, help="Previously downloaded pinned Hub API response")
    args = ap.parse_args()
    if args.metadata:
        meta = json.loads(args.metadata.read_text())
    else:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true", timeout=30) as response:
            meta = json.load(response)
    if (meta.get("repo_id", meta.get("id")), meta.get("revision", meta.get("sha"))) != (REPO, REV):
        raise ValueError("Model identity or revision mismatch")
    files = []
    for row in meta["siblings"]:
        path = args.model_dir / row["rfilename"]
        size = path.stat().st_size
        if size != row["size"]:
            raise ValueError(f"File size mismatch: {path}")
        sha = hashlib.sha256()
        git = None if "lfs" in row else hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(8 << 20), b""):
                sha.update(block)
                if git is not None:
                    git.update(block)
        actual = git.hexdigest() if git is not None else sha.hexdigest()
        expected = row["blobId"] if git is not None else row["lfs"]["sha256"]
        if actual != expected:
            raise ValueError(f"Digest mismatch: {path}")
        files.append(dict(file=row["rfilename"], bytes=size, sha256=sha.hexdigest(),
                          hub_digest=expected, hub_algorithm="git-blob-sha1" if git else "sha256"))
    args.output.write_text(json.dumps(dict(repo_id=REPO, revision=REV, files=files), indent=2)+"\n")
    print(f"{REPO}@{REV}: {len(files)} verified files, {sum(r['bytes'] for r in files)} bytes")


if __name__ == "__main__":
    main()
