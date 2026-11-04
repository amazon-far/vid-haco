#!/usr/bin/env python3
"""Download the Vid-HACO checkpoints from a Hugging Face Hub repository into this directory.

The checkpoints are not released yet, so there is no default repository; pass --repo or set VIDHACO_HF_REPO.

    python checkpoints/fetch_checkpoints.py --repo <hf-repo>            # both
    python checkpoints/fetch_checkpoints.py --repo <hf-repo> vitl       # just one
"""
import argparse
import os

from huggingface_hub import hf_hub_download

FILES = {
    "vitl": "vitl.pt",
    "vith": "vith.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", choices=sorted(FILES), help="default: all")
    ap.add_argument("--repo", default=os.environ.get("VIDHACO_HF_REPO"),
                    help="Hugging Face Hub repository id (default: $VIDHACO_HF_REPO)")
    a = ap.parse_args()
    if not a.repo:
        ap.error("no checkpoint repository given; pass --repo or set VIDHACO_HF_REPO")
    here = os.path.dirname(os.path.abspath(__file__))
    for m in a.models or sorted(FILES):
        path = hf_hub_download(a.repo, FILES[m], local_dir=here)
        print(f"[{m}] {path}")


if __name__ == "__main__":
    main()
