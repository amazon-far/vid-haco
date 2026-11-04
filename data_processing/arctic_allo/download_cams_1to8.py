#!/usr/bin/env python3
"""Download ARCTIC cropped_images zips and extract the ALLOCENTRIC cameras (1..8).

Background: the per-sequence zip at .../cropped_images_zips/<subj>/<seq>.zip contains ALL 9 cameras
as "<cam>/<img>.jpg". The original HACO staging (arctic_repo/haco_download_cropped.py) set CAM="0",
extracted only the egocentric view, and DELETED the zip. So the allocentric views were downloaded
and thrown away. This re-fetches the same 267 train-sequence zips and keeps cams 1..8.

SAFETY: this is ADDITIVE. It only creates NEW sibling dirs
    ARCTIC/data/cropped_images/<subj>/<seq>/{1..8}/
next to the existing /0/. A training job reading cam 0 does os.listdir() on the /0/ dir only, so its
view of the data is unchanged. Nothing under ARCTIC/splits/, ARCTIC/data/splits/ or
ARCTIC/preprocessed_data/ is touched.

THROTTLING: MPI's portal 403s on a high REQUEST RATE (not bandwidth). One big streaming request per
zip is fine; that is why this keeps concurrency low and paces between zips, with 403 backoff.
"""
import os
import sys
import time
import zipfile
import shutil
import warnings
import argparse
import threading
import requests

warnings.filterwarnings("ignore")

CAMS = [str(c) for c in range(1, 9)]          # allocentric views
_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


def seq_key(url):
    part = url.split("/cropped_images_zips/")[1]
    subject, seqzip = part.split("/")
    return subject, seqzip.replace(".zip", "")


def already_done(dest_root, subject, seq):
    """Done == the LAST cam dir exists and is non-empty (cams are extracted in order)."""
    d = os.path.join(dest_root, subject, seq, CAMS[-1])
    return os.path.isdir(d) and any(f.endswith(".jpg") for f in os.listdir(d))


def download(session, url, out_path, post):
    for attempt in range(8):
        try:
            r = session.post(url, data=post, stream=True, verify=False,
                             allow_redirects=True, timeout=600)
            if r.status_code == 403:
                wait = min(180, 20 * (attempt + 1))
                log(f"    403 throttled, sleep {wait}s")
                r.close(); time.sleep(wait); continue
            if r.status_code not in (200, 206):
                log(f"    status {r.status_code}, retry")
                r.close(); time.sleep(15); continue
            if "text/html" in r.headers.get("Content-Type", ""):
                log("    got HTML (auth/throttle), retry")
                r.close(); time.sleep(30); continue
            total = int(r.headers.get("Content-Length", "0"))
            got = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk); got += len(chunk)
            r.close()
            if total and got < total:
                log(f"    short read {got}/{total}, retry")
                time.sleep(15); continue
            with open(out_path, "rb") as f:
                if f.read(2) != b"PK":
                    log("    not a zip, retry")
                    time.sleep(15); continue
            return True
        except Exception as e:
            log(f"    download attempt {attempt} error: {e}")
            time.sleep(15)
    return False


def extract_allo(zip_path, dest_root, subject, seq):
    dest = os.path.join(dest_root, subject, seq)      # zip has "<cam>/<img>.jpg"
    want = set(CAMS)
    n = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            name = info.filename
            parts = name.split("/")
            if len(parts) >= 2 and parts[0] in want and name.endswith(".jpg"):
                out_p = os.path.join(dest, name)
                os.makedirs(os.path.dirname(out_p), exist_ok=True)
                with z.open(info) as src, open(out_p, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    return n


def worker(idx, url, dest_root, staging, post, counters, total_n):
    subject, seq = seq_key(url)
    if already_done(dest_root, subject, seq):
        with _print_lock:
            counters["skip"] += 1
        return
    zip_path = os.path.join(staging, f"{subject}__{seq}.zip")
    session = requests.Session()
    log(f"[{idx+1}/{total_n}] {subject}/{seq} downloading...")
    if not download(session, url, zip_path, post):
        log(f"    FAILED {subject}/{seq}")
        with _print_lock:
            counters["fail"].append(f"{subject}/{seq}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return
    try:
        n = extract_allo(zip_path, dest_root, subject, seq)
        with _print_lock:
            counters["ok"] += 1
            counters["imgs"] += n
        log(f"    {subject}/{seq}: extracted {n} allo jpgs (cams 1-8)  "
            f"[done {counters['ok']}/{total_n}]")
    except Exception as e:
        log(f"    EXTRACT FAILED {subject}/{seq}: {e}")
        with _print_lock:
            counters["fail"].append(f"{subject}/{seq} (extract)")
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        time.sleep(1)     # gentle pacing between zips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="data/arctic_allo_work/urls_train267.txt")
    ap.add_argument("--staging", default="data/arctic_allo_work/_zips")
    ap.add_argument("--dest", default="data/ARCTIC/data/cropped_images")
    ap.add_argument("--workers", type=int, default=2, help="keep LOW: MPI 403s on request rate")
    ap.add_argument("--start-delay", type=int, default=0, help="sleep before starting (throttle cooldown)")
    args = ap.parse_args()

    post = {"username": os.environ["ARCTIC_USERNAME"],
            "password": os.environ["ARCTIC_PASSWORD"]}
    os.makedirs(args.staging, exist_ok=True)
    os.makedirs(args.dest, exist_ok=True)

    if args.start_delay:
        log(f"[allo] cooling down {args.start_delay}s before first request (403 throttle)...")
        time.sleep(args.start_delay)

    urls = [l.strip() for l in open(args.urls) if l.strip()]
    log(f"[allo] {len(urls)} sequence zips; extracting cams {CAMS} -> {args.dest}")
    log(f"[allo] workers={args.workers} (low on purpose)")

    counters = {"ok": 0, "skip": 0, "imgs": 0, "fail": []}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, u in enumerate(urls):
            ex.submit(worker, i, u, args.dest, args.staging, post, counters, len(urls))

    log(f"\n[allo] DONE. seqs_ok={counters['ok']} skipped={counters['skip']} "
        f"failed={len(counters['fail'])} allo_images={counters['imgs']}")
    if counters["fail"]:
        with open(os.path.join(args.staging, "failed_seqs.txt"), "w") as f:
            f.write("\n".join(counters["fail"]))
        log("FAILED: " + ", ".join(counters["fail"][:10]))
    log("[allo] EXIT_OK")


if __name__ == "__main__":
    main()
