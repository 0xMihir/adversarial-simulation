import json
import requests as r
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
import sys
import heapq
import itertools
import logging
from urllib.parse import urlsplit
from random import randint
from threading import Lock
from tqdm import tqdm

error_logger = logging.getLogger('scrape_errors')
error_logger.setLevel(logging.ERROR)

def init_error_log(output_dir: str):
    """Attach a file handler that logs all errors to <output_dir>/errors.log."""
    handler = logging.FileHandler(f'{output_dir}/errors.log')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    error_logger.addHandler(handler)

def log_error(message: str):
    """Print an error to the console and record it in errors.log."""
    print(message)
    error_logger.error(message)

def load_proxies(path: str = 'proxies.txt'):
    """Load proxies from a file with lines formatted as ip:port:username:password."""
    proxies = []
    if not os.path.exists(path):
        return proxies
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(':')
            if len(parts) == 4:
                ip, port, username, password = parts
                proxy_url = f'http://{username}:{password}@{ip}:{port}'
                proxies.append({'http': proxy_url, 'https': proxy_url})
            elif len(parts) == 2:
                ip, port = parts
                proxy_url = f'http://{ip}:{port}'
                proxies.append({'http': proxy_url, 'https': proxy_url})
            else:
                print(f"Skipping malformed proxy line: {line}")
                continue
    return proxies

def proxy_to_line(proxy: dict) -> str:
    """Convert a proxy dict back to an ip:port[:username:password] line, as load_proxies expects."""
    parts = urlsplit(proxy['http'])
    if parts.username:
        return f'{parts.hostname}:{parts.port}:{parts.username}:{parts.password}'
    return f'{parts.hostname}:{parts.port}'

def test_proxy(proxy: dict) -> bool:
    """Check whether a proxy can successfully complete a request."""
    try:
        response = r.get('https://api.ipify.org', proxies=proxy, timeout=10)
        return response.status_code == 200
    except Exception:
        return False

def test_proxies(proxies: list[dict]) -> list[dict]:
    """Test all proxies concurrently and return only the ones that work."""
    if not proxies:
        return proxies
    working = []
    with ThreadPoolExecutor(max_workers=len(proxies)) as executor:
        future_to_proxy = {executor.submit(test_proxy, proxy): proxy for proxy in proxies}
        for future in tqdm(as_completed(future_to_proxy), total=len(proxies), desc="Testing proxies"):
            proxy = future_to_proxy[future]
            if future.result():
                working.append(proxy)
            else:
                print(f"Proxy {proxy['http']} failed the connectivity test.")
    return working

def working_proxies_cache_path(proxy_file: str) -> str:
    root, ext = os.path.splitext(proxy_file)
    return f'{root}-working{ext}'

def get_working_proxies(proxy_file: str) -> list[dict]:
    """Load proxies from proxy_file, using a cached working-proxies file if present.

    Skips re-testing (which is slow and can get proxies further rate-limited) on repeat
    runs by writing whichever proxies pass the connectivity test to a sibling cache file.
    """
    cache_path = working_proxies_cache_path(proxy_file)
    if os.path.exists(cache_path):
        proxies = load_proxies(cache_path)
        print(f"Loaded {len(proxies)} previously-tested working proxies from {cache_path}.")
        return proxies

    proxies = load_proxies(proxy_file)
    print(f"Loaded {len(proxies)} proxies." if proxies else "No proxies loaded; requests will use a direct connection.")
    if not proxies:
        return proxies

    proxies = test_proxies(proxies)
    if proxies:
        print(f"{len(proxies)} proxies passed the connectivity test.")
    else:
        print("No proxies passed the connectivity test.")
        return proxies

    with open(cache_path, 'w') as f:
        for proxy in proxies:
            f.write(proxy_to_line(proxy) + '\n')
    print(f"Cached working proxies to {cache_path}.")
    return proxies

class ProxyPool:
    """Round-robins proxies, prioritizing whichever has accrued the fewest errors."""

    def __init__(self, proxies: list[dict]):
        self.proxies = proxies
        self.lock = Lock()
        self.counter = itertools.count()
        # Heap of (error_count, sequence, proxy). Lower error_count = higher priority.
        # `sequence` breaks ties in insertion order and avoids comparing dicts.
        self.heap = [(0, next(self.counter), proxy) for proxy in proxies]
        heapq.heapify(self.heap)
        # Persistent error counts per proxy, independent of checkout state.
        self.error_counts: dict[int, int] = {id(proxy): 0 for proxy in proxies}
        # Proxy ids currently checked out (not in the heap).
        self.checked_out: set[int] = set()

    def get_proxy(self):
        """Check out the proxy with the fewest errors so far, or None if no proxies are configured."""
        if not self.heap:
            return None
        with self.lock:
            _, _, proxy = heapq.heappop(self.heap)
            self.checked_out.add(id(proxy))
        return proxy

    def report_result(self, proxy, success: bool):
        """Record a request outcome for a checked-out proxy, without releasing it back to the heap."""
        if proxy is None:
            return
        with self.lock:
            if not success:
                self.error_counts[id(proxy)] = self.error_counts.get(id(proxy), 0) + 1

    def release_proxy(self, proxy):
        """Return a checked-out proxy to the heap, using its accrued error count for priority."""
        if proxy is None:
            return
        with self.lock:
            self.checked_out.discard(id(proxy))
            error_count = self.error_counts.get(id(proxy), 0)
            heapq.heappush(self.heap, (error_count, next(self.counter), proxy))

    def debug_print(self):
        """Print the heap's current (error_count, proxy) pairs, sorted by priority."""
        with self.lock:
            entries = sorted(self.heap)
            checked_out = len(self.checked_out)
        state = ', '.join(f"{urlsplit(proxy['http']).hostname}:{urlsplit(proxy['http']).port}=err{count}"
                           for count, _, proxy in entries)
        print(f"[proxy heap] {len(entries)} idle, {checked_out} checked out -> {state}")

def get_crash_details(case_id: int, output_dir: str, proxy_pool: ProxyPool, proxy):
    """Fetch crash details for a given case ID and save to JSON file."""
    url = f'https://crashviewer.nhtsa.dot.gov/api/case/GetCrashDetails?caseID={case_id}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:145.0) Gecko/20100101 Firefox/145.0',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Origin': 'https://crashviewer.nhtsa.dot.gov',
        'Connection': 'keep-alive',
        'Referer': f'https://crashviewer.nhtsa.dot.gov/ciss/details/{case_id}/crash-summary-document',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Content-Length': '0',
        'TE': 'trailers',
    }

    case_dir = f'{output_dir}/{case_id}'
    os.makedirs(case_dir, exist_ok=True)
    output_file = f'{case_dir}/details.json'
    if os.path.exists(output_file):
        print(f"Details file for case {case_id} already exists. Skipping.")
        return False

    try:
        response = r.post(url, headers=headers, proxies=proxy, timeout=30)
    except Exception as e:
        log_error(f"Request error for crash details (case {case_id}): {e}")
        proxy_pool.report_result(proxy, success=False)
        return True

    if response.status_code != 200:
        log_error(f"Failed to retrieve crash details for case ID {case_id}. Status code: {response.status_code}")
        proxy_pool.report_result(proxy, success=False)
        return True

    proxy_pool.report_result(proxy, success=True)
    try:
        details = response.json()
        with open(output_file, 'w') as f:
            json.dump(details, f, indent=2)
        print(f"Saved crash details for case ID {case_id}.")
    except Exception as e:
        log_error(f"Error processing crash details for case {case_id}: {e}")
    return True

def get_scene_files(case_id: int, output_dir: str, proxy_pool: ProxyPool, proxy):
    """Fetch the scene diagram listing and download all .far/.blz files for a case."""
    url = f'https://crashviewer.nhtsa.dot.gov/api/case/GetSceneDiagram?caseID={case_id}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:145.0) Gecko/20100101 Firefox/145.0',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Origin': 'https://crashviewer.nhtsa.dot.gov',
        'Connection': 'keep-alive',
        'Referer': f'https://crashviewer.nhtsa.dot.gov/ciss/details/{case_id}/crash-summary-document',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
    }

    try:
        response = r.post(url, headers=headers, proxies=proxy, timeout=30)
    except Exception as e:
        log_error(f"Request error for case {case_id}: {e}")
        proxy_pool.report_result(proxy, success=False)
        return True

    if response.status_code != 200:
        log_error(f"Failed to retrieve scene diagram for case ID {case_id}. Status code: {response.status_code}")
        proxy_pool.report_result(proxy, success=False)
        return True

    proxy_pool.report_result(proxy, success=True)
    try:
        scene_files = response.json()
    except Exception as e:
        log_error(f"Bad JSON for case {case_id}: {e}")
        return True

    case_dir = f'{output_dir}/{case_id}'
    os.makedirs(case_dir, exist_ok=True)

    for file in scene_files:
        file_name = file.get('filename', '')
        if not (file_name.endswith('.far') or file_name.endswith('.blz')):
            continue
        ext = file_name.rsplit('.', 1)[-1]
        output_file = f'{case_dir}/scene.{ext}'
        if os.path.exists(output_file):
            print(f"File {file_name} for case ID {case_id} already exists. Skipping.")
            continue

        object_id = file.get('objectid')
        if not object_id:
            continue

        download_url = f'https://crashviewer.nhtsa.dot.gov/api/case/scenefiles/download/{case_id}?objectId={object_id}'
        file_headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:145.0) Gecko/20100101 Firefox/145.0',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Referer': f'https://crashviewer.nhtsa.dot.gov/ciss/details/{case_id}/crash-summary-scene-diagram',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        }

        try:
            file_response = r.get(download_url, headers=file_headers, proxies=proxy, timeout=60)
        except Exception as e:
            log_error(f"Download error for {file_name} (case {case_id}): {e}")
            proxy_pool.report_result(proxy, success=False)
            continue

        if file_response.status_code == 200:
            proxy_pool.report_result(proxy, success=True)
            try:
                with open(output_file, 'wb') as f:
                    f.write(file_response.content)
                print(f"Downloaded {file_name} for case ID {case_id}.")
            except Exception as e:
                log_error(f"File write error for {file_name} (case {case_id}): {e}")
        else:
            log_error(f"Failed to download {file_name} for case ID {case_id}. Status code: {file_response.status_code}")
            proxy_pool.report_result(proxy, success=False)

    return True

def case_already_scraped(case_id: int, output_dir: str) -> bool:
    """A case is fully scraped once it has details.json and a downloaded scene file."""
    case_dir = f'{output_dir}/{case_id}'
    if not os.path.exists(f'{case_dir}/details.json'):
        return False
    return any(
        entry.name.startswith('scene.')
        for entry in os.scandir(case_dir)
    ) if os.path.isdir(case_dir) else False

def process_case(case_id: int, output_dir: str, proxy_pool: ProxyPool):
    """Fetch crash details and scene files (.far/.blz) for a case, sharing one proxy."""
    proxy = proxy_pool.get_proxy()
    # proxy_pool.debug_print()
    try:
        if get_crash_details(case_id, output_dir, proxy_pool, proxy):
            sleep(randint(2, 10))  # Random sleep to avoid rate limiting
        if get_scene_files(case_id, output_dir, proxy_pool, proxy):
            sleep(randint(2, 10))  # Random sleep to avoid rate limiting
    finally:
        proxy_pool.release_proxy(proxy)

def parse_args():
    parser = argparse.ArgumentParser(description="Scrape NHTSA CISS crash details and scene files.")
    parser.add_argument('path', nargs='?', default='search_2022.2023.json', help="Path to the case search JSON file.")
    parser.add_argument('output_dir', nargs='?', default='output', help="Directory to save downloaded files to.")
    parser.add_argument('proxy_file', nargs='?', default='proxies.txt', help="Path to the proxies file.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    init_error_log(args.output_dir)
    with open(args.path, 'r') as f:
        data = json.load(f)

    case_ids = [int(case['caseId']) for case in data]
    print(f"Found {len(case_ids)} case IDs.")

    case_ids = [cid for cid in case_ids if not case_already_scraped(cid, args.output_dir)]
    print(f"{len(case_ids)} case IDs remaining after filtering already-scraped cases.")

    proxies = get_working_proxies(args.proxy_file)
    if os.path.exists(args.proxy_file) and not proxies:
        print("No proxies passed the connectivity test.")
        sys.exit(1)
    proxy_pool = ProxyPool([])

    max_attempts = 3
    pending = case_ids
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        desc = "Cases" if attempt == 1 else f"Retry {attempt - 1}/{max_attempts - 1}"
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(process_case, cid, args.output_dir, proxy_pool) for cid in pending]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=desc):
                pass
        pending = [cid for cid in pending if not case_already_scraped(cid, args.output_dir)]
        if pending and attempt < max_attempts:
            print(f"{len(pending)} case(s) still incomplete after attempt {attempt}. Retrying.")

    if pending:
        log_error(f"{len(pending)} case(s) failed after {max_attempts} attempts: {pending}")

if __name__ == '__main__':
    main()