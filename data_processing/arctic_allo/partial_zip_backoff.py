#!/usr/bin/env python3
"""Extract specific member(s) from a remote ZIP served behind POST-credential auth,
using HTTP Range requests, without downloading the whole archive.
Handles ZIP64. Targets ARCTIC splits.zip to pull only p2_train.npy.
"""
import os
import sys
import struct
import zlib
import io
import warnings
import requests

warnings.filterwarnings("ignore")

URL = sys.argv[1]
TARGET_SUFFIX = sys.argv[2]  # e.g. "p2_train.npy"
OUT_PATH = sys.argv[3]

U = os.environ["ARCTIC_USERNAME"]
P = os.environ["ARCTIC_PASSWORD"]
POST = {"username": U, "password": P}

session = requests.Session()


def get_range(start, end):
    """Inclusive byte range [start, end]. Returns bytes.

    MPI's portal 403-throttles rapid successive range requests. The upstream version retried 6x with
    NO sleep, so all 6 retries hit the same throttle window and it died. Back off instead, and pace
    every request.
    """
    import time as _t
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(10):
        try:
            r = session.post(URL, data=POST, headers=headers, stream=True,
                             verify=False, allow_redirects=True, timeout=180)
            if r.status_code == 403:
                wait = min(120, 20 * (attempt + 1))
                print(f"  403 throttled, sleep {wait}s", flush=True)
                r.close(); _t.sleep(wait); continue
            if r.status_code not in (200, 206):
                r.close(); _t.sleep(10); continue
            data = r.content
            r.close()
            _t.sleep(2)                 # gentle pacing between range reads
            return data
        except Exception as e:
            print(f"  range {start}-{end} attempt {attempt} failed: {e}", flush=True)
            _t.sleep(10)
    raise RuntimeError(f"failed range {start}-{end}")


def get_total_size():
    import time as _t
    headers = {"Range": "bytes=0-0"}
    for attempt in range(10):
        r = session.post(URL, data=POST, headers=headers, stream=True,
                         verify=False, allow_redirects=True, timeout=180)
        cr = r.headers.get("Content-Range")  # bytes 0-0/TOTAL
        code = r.status_code
        r.close()
        if cr and "/" in cr:
            _t.sleep(2)
            return int(cr.split("/")[-1])
        wait = min(120, 20 * (attempt + 1))
        print(f"  size probe: status={code}, throttled -> sleep {wait}s", flush=True)
        _t.sleep(wait)
    raise RuntimeError("no content-range for size")


EOCD_SIG = b"PK\x05\x06"
EOCD64_LOC_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CEN_SIG = b"PK\x01\x02"
LOC_SIG = b"PK\x03\x04"


def main():
    total = get_total_size()
    print(f"Total archive size: {total} bytes ({total/1e9:.2f} GB)", flush=True)

    # Read tail to find EOCD
    tail_len = min(1 << 20, total)
    tail = get_range(total - tail_len, total - 1)
    tail_base = total - tail_len

    idx = tail.rfind(EOCD_SIG)
    if idx < 0:
        raise RuntimeError("EOCD not found")
    eocd = tail[idx:idx + 22]
    (_, disk, cd_disk, n_disk, n_total, cd_size, cd_offset, comment_len) = struct.unpack(
        "<IHHHHIIH", eocd[:22])

    # Check for ZIP64
    if cd_offset == 0xFFFFFFFF or n_total == 0xFFFF or cd_size == 0xFFFFFFFF:
        loc_idx = tail.rfind(EOCD64_LOC_SIG)
        if loc_idx < 0:
            raise RuntimeError("ZIP64 locator not found")
        loc = tail[loc_idx:loc_idx + 20]
        (_, _, eocd64_off, _) = struct.unpack("<IIQI", loc)
        # Fetch EOCD64 record (56 bytes minimum)
        z64 = get_range(eocd64_off, eocd64_off + 55)
        if z64[:4] != EOCD64_SIG:
            raise RuntimeError("EOCD64 sig mismatch")
        (_, eocd64_size, vmade, vneed, disk2, cddisk2, n_disk2, n_total2,
         cd_size, cd_offset) = struct.unpack("<IQHHIIQQQQ", z64[:56])
        n_total = n_total2
        print("ZIP64 detected", flush=True)

    print(f"Central dir: offset={cd_offset} size={cd_size} entries={n_total}", flush=True)

    # Fetch the whole central directory
    cd = get_range(cd_offset, cd_offset + cd_size - 1)

    # Parse central directory entries
    pos = 0
    found = None
    while pos < len(cd) - 4:
        if cd[pos:pos+4] != CEN_SIG:
            break
        (sig, vmade, vneed, flags, method, mtime, mdate, crc,
         comp_size, uncomp_size, name_len, extra_len, comment_len,
         disk_start, int_attr, ext_attr, lho) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos:pos+46])
        name = cd[pos+46:pos+46+name_len].decode("utf-8", "replace")
        extra = cd[pos+46+name_len:pos+46+name_len+extra_len]

        # ZIP64 extra field parse for sizes/offset if 0xFFFFFFFF
        local_off = lho
        cs = comp_size
        us = uncomp_size
        if cs == 0xFFFFFFFF or us == 0xFFFFFFFF or local_off == 0xFFFFFFFF:
            ep = 0
            while ep < len(extra) - 4:
                hid, hsz = struct.unpack("<HH", extra[ep:ep+4])
                body = extra[ep+4:ep+4+hsz]
                if hid == 0x0001:
                    bp = 0
                    if us == 0xFFFFFFFF:
                        us = struct.unpack("<Q", body[bp:bp+8])[0]; bp += 8
                    if cs == 0xFFFFFFFF:
                        cs = struct.unpack("<Q", body[bp:bp+8])[0]; bp += 8
                    if local_off == 0xFFFFFFFF:
                        local_off = struct.unpack("<Q", body[bp:bp+8])[0]; bp += 8
                    break
                ep += 4 + hsz

        if name.endswith(TARGET_SUFFIX):
            found = (name, method, cs, us, crc, local_off)
            print(f"FOUND member: {name} method={method} comp={cs} uncomp={us} local_off={local_off}", flush=True)
        pos += 46 + name_len + extra_len + comment_len

    if not found:
        raise RuntimeError(f"member ending with {TARGET_SUFFIX} not found")

    name, method, cs, us, crc, local_off = found

    # Read local file header to get the real data offset (name_len + extra_len can differ)
    lh = get_range(local_off, local_off + 29)
    if lh[:4] != LOC_SIG:
        raise RuntimeError("local header sig mismatch")
    (lsig, lver, lflag, lmethod, lmtime, lmdate, lcrc, lcs, lus, lname_len, lextra_len) = struct.unpack("<IHHHHHIIIHH", lh[:30])
    data_start = local_off + 30 + lname_len + lextra_len
    data_end = data_start + cs - 1

    print(f"Downloading member data: {cs/1e6:.1f} MB compressed -> {us/1e6:.1f} MB uncompressed", flush=True)

    # Download in chunks via range (member could be large)
    CHUNK = 64 * 1024 * 1024
    comp = io.BytesIO()
    s = data_start
    while s <= data_end:
        e = min(s + CHUNK - 1, data_end)
        comp.write(get_range(s, e))
        print(f"  got {comp.tell()/1e6:.1f}/{cs/1e6:.1f} MB", flush=True)
        s = e + 1
    comp_bytes = comp.getvalue()

    if method == 0:
        raw = comp_bytes
    elif method == 8:
        raw = zlib.decompress(comp_bytes, -15)
    else:
        raise RuntimeError(f"unsupported compression method {method}")

    if len(raw) != us:
        print(f"WARNING: uncompressed size mismatch {len(raw)} != {us}", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        f.write(raw)
    print(f"WROTE {OUT_PATH} ({len(raw)} bytes)", flush=True)


if __name__ == "__main__":
    main()
