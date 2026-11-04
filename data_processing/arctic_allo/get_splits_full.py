#!/usr/bin/env python3
"""Download ARCTIC splits.zip (18.6 GB) as ONE resumable streaming request and extract
p1_train.npy (allocentric) + p2_train.npy (egocentric, ORIGINAL with its data_dict intact).

Why not partial-zip: MPI's portal 403-throttles on REQUEST RATE. The ZIP64 partial extractor issues
dozens of small Range requests and gets banned. One big streaming GET is exactly what the portal is
happy with (the cropped-image downloader does the same and ran 267/267 clean).

Why p2_train.npy too: HACO ships ARCTIC with an EMPTY data_dict (it kept only `imgnames`), and its
bbox_hand_r values come pre-computed in preprocessed_data/annot_data. Re-downloading the ORIGINAL
p2_train.npy lets us re-derive cam-0 bboxes from ARCTIC's own annotations and CHECK them against
HACO's shipped values. If they match, the same derivation is trustworthy for cams 1-8 (p1).
"""
import os
import sys
import time
import zipfile
import warnings
import requests

warnings.filterwarnings("ignore")

URL = ("https://download.is.tue.mpg.de/download.php?domain=arctic&resume=1&sfile=arctic_release/"
       "c7216c3b205186106a1f8326ed7b948f838e4907e69b21c8b3c87bb69d87206e/v1_0/data/splits.zip")
ZIP_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/arctic_allo_work/_splits.zip"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "data/arctic_allo_work"
WANT = ("p1_train.npy", "p2_train.npy")

U = os.environ["ARCTIC_USERNAME"]
P = os.environ["ARCTIC_PASSWORD"]
POST = {"username": U, "password": P}

os.makedirs(OUT_DIR, exist_ok=True)
session = requests.Session()


def total_size():
    for a in range(8):
        r = session.post(URL, data=POST, headers={"Range": "bytes=0-0"}, stream=True,
                         verify=False, allow_redirects=True, timeout=180)
        cr = r.headers.get("Content-Range")
        code = r.status_code
        r.close()
        if cr and "/" in cr:
            return int(cr.split("/")[-1])
        w = min(180, 30 * (a + 1))
        print(f"  size probe status={code} -> sleep {w}s", flush=True)
        time.sleep(w)
    raise RuntimeError("cannot get size")


def fetch():
    total = total_size()
    print(f"[splits] archive {total/1e9:.2f} GB", flush=True)
    for attempt in range(12):
        have = os.path.getsize(ZIP_PATH) if os.path.exists(ZIP_PATH) else 0
        if have >= total:
            print(f"[splits] complete ({have} bytes)", flush=True)
            return True
        headers = {"Range": f"bytes={have}-"} if have else {}
        print(f"[splits] attempt {attempt}: resume at {have/1e9:.2f} GB", flush=True)
        try:
            r = session.post(URL, data=POST, headers=headers, stream=True,
                             verify=False, allow_redirects=True, timeout=600)
            if r.status_code == 403:
                w = min(300, 30 * (attempt + 1))
                print(f"  403 throttled -> sleep {w}s", flush=True)
                r.close(); time.sleep(w); continue
            if r.status_code not in (200, 206):
                print(f"  status {r.status_code} -> retry", flush=True)
                r.close(); time.sleep(30); continue
            mode = "ab" if (have and r.status_code == 206) else "wb"
            if mode == "wb":
                have = 0
            t0, last = time.time(), have
            with open(ZIP_PATH, mode) as f:
                for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk); have += len(chunk)
                    if time.time() - t0 > 60:
                        mbps = (have - last) / (time.time() - t0) / 1e6
                        print(f"  {have/1e9:.2f}/{total/1e9:.2f} GB ({mbps:.1f} MB/s)", flush=True)
                        t0, last = time.time(), have
            r.close()
        except Exception as e:
            print(f"  error: {e} -> retry", flush=True)
            time.sleep(30)
    return os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) >= total


def extract():
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = z.namelist()
        print(f"[splits] {len(names)} members: {names}", flush=True)
        for w in WANT:
            cand = [n for n in names if n.endswith(w)]
            if not cand:
                print(f"  !! {w} NOT in archive", flush=True)
                continue
            src = cand[0]
            dst = os.path.join(OUT_DIR, w)
            with z.open(src) as fi, open(dst, "wb") as fo:
                while True:
                    b = fi.read(32 << 20)
                    if not b:
                        break
                    fo.write(b)
            print(f"  extracted {src} -> {dst} ({os.path.getsize(dst)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    if not fetch():
        sys.exit("[splits] download failed")
    extract()
    os.remove(ZIP_PATH)
    print("[splits] zip removed; EXIT_OK", flush=True)
