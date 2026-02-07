import json
import requests as r
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
from random import randint

def get_crash_details(case_id: int):
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

    output_file = f'output/{case_id}_details.json'
    if os.path.exists(output_file):
        print(f"Details file for case {case_id} already exists. Skipping.")
        return

    try:
        response = r.post(url, headers=headers, timeout=30)
    except Exception as e:
        print(f"Request error for crash details (case {case_id}): {e}")
        return

    if response.status_code != 200:
        print(f"Failed to retrieve crash details for case ID {case_id}. Status code: {response.status_code}")
        return

    try:
        details = response.json()
        with open(output_file, 'w') as f:
            json.dump(details, f, indent=2)
        print(f"Saved crash details for case ID {case_id}.")
    except Exception as e:
        print(f"Error processing crash details for case {case_id}: {e}")
        return
    sleep(randint(1, 3))  # Random sleep to avoid rate limiting

def process_case(case_id: int):
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
        response = r.post(url, headers=headers, timeout=30)
    except Exception as e:
        print(f"Request error for case {case_id}: {e}")
        return

    if response.status_code != 200:
        print(f"Failed to retrieve scene diagram for case ID {case_id}. Status code: {response.status_code}")
        return

    try:
        scene_files = response.json()
    except Exception as e:
        print(f"Bad JSON for case {case_id}: {e}")
        return

    for file in scene_files:
        file_name = file.get('filename', '')
        if not (file_name.endswith('.far') or file_name.endswith('.blz')):
            continue
        if os.path.exists(f'output/{case_id}_{file_name}'):
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
            file_response = r.get(download_url, headers=file_headers, timeout=60)
        except Exception as e:
            print(f"Download error for {file_name} (case {case_id}): {e}")
            continue

        if file_response.status_code == 200:
            try:
                with open(f'output/{case_id}_{file_name}', 'wb') as f:
                    f.write(file_response.content)
                print(f"Downloaded {file_name} for case ID {case_id}.")
            except Exception as e:
                print(f"File write error for {file_name} (case {case_id}): {e}")
        else:
            print(f"Failed to download {file_name} for case ID {case_id}. Status code: {file_response.status_code}")
        sleep(0.5) 

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'search_2022.2023.json'
    with open(path, 'r') as f:
        data = json.load(f)

    case_ids = [int(case['caseId']) for case in data]
    print(f"Found {len(case_ids)} case IDs.")

    # first_case = 20462
    first_case = None
    if first_case is not None:
        try:
            start_index = case_ids.index(int(first_case))
            case_ids = case_ids[start_index:]
        except ValueError:
            print(f"First case {first_case} not found. Processing all.")

    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     futures = [executor.submit(process_case, cid) for cid in case_ids]
    #     for _ in as_completed(futures):
    #         pass

    # Fetch crash details for all cases
    print("\nFetching crash details for all cases...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_crash_details, cid) for cid in case_ids]
        for _ in as_completed(futures):
            pass

if __name__ == '__main__':
    main()