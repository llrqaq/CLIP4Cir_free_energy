import argparse
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

BASE = Path(__file__).parent
URL_DIR = BASE / "fashion-iq-images-temp" / "image_url"
DATA_DIR = BASE / "fashionIQ_dataset"
IMAGE_DIR = DATA_DIR / "images"
BROKEN_DIR = URL_DIR / "broken_links"
FAIL_LOG = BASE / "fashioniq_download_failures.tsv"

_thread_local = threading.local()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--proxy", default=os.environ.get("HTTP_PROXY", "http://127.0.0.1:7890"))
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--all-url-images", action="store_true", help="download every URL instead of only images used by FashionIQ splits")
    return parser.parse_args()


def load_needed_images():
    if not DATA_DIR.exists():
        return None

    needed = set()
    split_dir = DATA_DIR / "image_splits"
    for split in ["train", "val", "test"]:
        for category in ["dress", "shirt", "toptee"]:
            path = split_dir / f"split.{category}.{split}.json"
            if path.exists():
                with open(path) as f:
                    needed.update(json.load(f))
    return needed


def load_url_map():
    url_map = {}
    for category in ["dress", "shirt", "toptee"]:
        path = URL_DIR / f"asin2url.{category}.txt"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                asin, url = parts[0].strip(), parts[1].strip()
                url_map[asin] = url
    return url_map


def load_broken_asins():
    broken = set()
    if BROKEN_DIR.exists():
        for path in BROKEN_DIR.iterdir():
            broken.add(path.name.replace(".jpg", ""))
    return broken


def get_session(proxy):
    session = getattr(_thread_local, "session", None)
    if session is not None:
        return session

    session = requests.Session()
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Connection": "keep-alive",
    })
    adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    _thread_local.session = session
    return session


def download_one(task):
    asin, url, args = task
    out = IMAGE_DIR / f"{asin}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return "exists", asin, ""

    session = get_session(args.proxy)
    last_error = ""

    for attempt in range(args.retries + 1):
        try:
            with session.get(url, timeout=args.timeout, stream=True) as resp:
                status = resp.status_code
                if status == 200:
                    tmp = out.with_suffix(".jpg.tmp")
                    with open(tmp, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=64 * 1024):
                            if chunk:
                                f.write(chunk)
                    if tmp.stat().st_size == 0:
                        tmp.unlink(missing_ok=True)
                        return "empty", asin, url
                    tmp.replace(out)
                    return "ok", asin, ""
                if status in (403, 404, 410):
                    return str(status), asin, url
                last_error = f"http_{status}"
        except Exception as exc:
            last_error = type(exc).__name__

        if attempt < args.retries:
            time.sleep(0.4 * (attempt + 1))

    return last_error or "failed", asin, url


def main():
    args = parse_args()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    url_map = load_url_map()
    broken = load_broken_asins()
    needed = None if args.all_url_images else load_needed_images()

    asins = sorted(url_map.keys() if needed is None else needed)
    missing_url = [asin for asin in asins if asin not in url_map]

    tasks = []
    skipped_broken = 0
    skipped_existing = 0
    for asin in asins:
        if asin in broken:
            skipped_broken += 1
            continue
        out = IMAGE_DIR / f"{asin}.jpg"
        if out.exists() and out.stat().st_size > 0:
            skipped_existing += 1
            continue
        url = url_map.get(asin)
        if url:
            tasks.append((asin, url, args))

    if args.limit:
        tasks = tasks[:args.limit]

    print(f"Mode: {'all URL images' if args.all_url_images else 'FashionIQ split images only'}", flush=True)
    print(f"Proxy: {args.proxy or 'disabled'}", flush=True)
    print(f"Threads: {args.threads}", flush=True)
    print(f"Needed ASINs: {len(asins)}", flush=True)
    print(f"Already downloaded: {skipped_existing}", flush=True)
    print(f"Skipped known broken: {skipped_broken}", flush=True)
    print(f"Missing URL entries: {len(missing_url)}", flush=True)
    print(f"To download: {len(tasks)}", flush=True)

    if not tasks:
        return

    counts = Counter()
    failures = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [pool.submit(download_one, task) for task in tasks]
        with tqdm(total=len(futures), ncols=120, unit="img", mininterval=0.5) as bar:
            for future in as_completed(futures):
                status, asin, detail = future.result()
                counts[status] += 1
                if status not in ("ok", "exists"):
                    failures.append((asin, status, detail))

                done = sum(counts.values())
                elapsed = max(time.time() - start, 1e-6)
                speed = done / elapsed
                bar.set_postfix({
                    "ok": counts["ok"],
                    "404": counts["404"],
                    "fail": len(failures),
                    "speed": f"{speed:.1f}/s",
                })
                bar.update(1)

    elapsed = time.time() - start
    print("\nSummary", flush=True)
    for key, value in counts.most_common():
        print(f"{key}: {value}", flush=True)
    print(f"Elapsed: {elapsed / 60:.1f} min", flush=True)
    print(f"Average speed: {len(tasks) / max(elapsed, 1e-6):.2f} img/s", flush=True)

    if failures:
        with open(FAIL_LOG, "w", encoding="utf-8") as f:
            for asin, status, detail in failures:
                f.write(f"{asin}\t{status}\t{detail}\n")
        print(f"Failures written to {FAIL_LOG}", flush=True)


if __name__ == "__main__":
    main()
