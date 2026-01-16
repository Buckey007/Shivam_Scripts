import csv
import time
import json
import requests
import itertools
import logging
from pathlib import Path

# ================= CONFIG =================

HUME_BASE = "https://api.hume.ai/v0"

API_KEYS = [
    'bVjZL9BvJb7fyqPusBVRlbMApXQrROLT60snR3STWPu72VPa', 
    'S9bTzY1CWr7M75ddD3evfzQNaHE9gOhaoGmjANjKosmSQo0v', 
    'G02wGGspSlMuG5rhtOXAC6JKcj4cmEkLT56zTJjBOkGWarKn', 
    '0MfyVhbr22QsZNOWQ3KS7EjenA3KdzdgQ0hvdjxLjaQj7RVz', 
    'mMhAllDQwUjGHwwEZf9Juam2jQXHqQXzZTQBExPMzHqmxLhg', 
    'nUTCghGcArX8xiNhnsaGNrvGTexQml0G0raCp22ojRu2sON9',
    "nUTCghGcArX8xiNhnsaGNrvGTexQml0G0raCp22ojRu2sON9",
    "shGJhk2YV1HPAIO0gN2wsXDVS4kmU4rXFmWALpUT7DGx8rnp",
    "x6EnsttAkRSAGLU0oOrAqdAgsXWAhvtw6PcKLlCJiFyHDLYR",
    "hA6aGf97JGVesmtCnCWUZub8mSUcMg5FYfa9XCqeEMHP1p6o",
    "FV71BGGjUOSEvI4Al52zbAzHAWK34gvnhPaAcDqKkvieYVxL",
    "bT2JFN2fLlkyPGDMHG0sPXQq6bDbH21BQ7T2T70CdpWRCR2g",
    "xPSPRGobuF2C7a8e2lNisrJVNPVLsKdfhc4yGelpT6QNSrGk",
    "GhTDY6g6NhvHp7M7mVC8Fmgb50HpyGYXBIaUn8Y48FDBZBa9",
    "157b9aRHupi3ETLGkX3H6pSpmsdwjHOrMaLtAMvAwVP9gSuy",
    "UmaDsWvR0M2CPy5n7PY3kbUVgyj5HSBXIh0T7xCR6ldUyKkL",
    "2qLibiqFyFu8OPWJmziz2eYlNw49ShuhW8wtNfrjY3CPQ54I",
    "hjc5N6cheCWhg2iLjybzHjQxcOi5KhspFAJBHQCa28BqDi0b",
    "bpAEgbz9jaDQW9U8fHYIf6OO5QknYRATACXeoJ4OoyVodrts",
    "R6LYldUqR51AmKMVAo569LXDK0BfzPEXkdymFjsg8WM5UWfP",
    "HTZcAMPCXqiSIAR4c1erRGryUjg4pce1VC0jC3tKpnYq5Hw2",
    "ijsnELfGi6rATkmdCyoA9657ANL3pVwc9ESzGXID3MBeSoWL",
    "Pzv3yljtdQFKrbzEARDrVpAJVD8CiSXy7VMP8Xap2AyC2vXD",
    "otzfnknlzQgUMIyJ74odoKL33L0oJKCQekSTk2ZNZ8N9lx5b",
    "iELoDmLkZPvw7sjYWm62CsawSXR58nlcMXmEDG2rDFW3W5OW",
    "ydOQCRRmoA3ElaLjAqMP2nbzwFBzY7br2QpM5CEIEdEHBCSe",
    "mkQTuSJ4bWhcGoBiUg3Yvm9UKHAkV9MbnGAcKcIZfTzoaxtV",
    "IDSGh6FsVARVpbILYrJShztHxQUAJlkGZ2kwILqhl2D87zRt"
]

BEARER = "YOUR_BEARER_TOKEN"

INPUT_CSV = "Dataset_Creation/input_csv/awkward_images.csv"
OUTPUT_CSV = "Dataset_Creation/output_csv/awkward_output.csv"
LOG_FILE = "hume_progress.log"

DEAD_KEYS_FILE = "dead_keys.txt"

FAILED_IMAGES_FILE = "failed_images.txt"


REQUEST_SLEEP = 5  # seconds (SAFE for free tier)

# ================= LOGGING =================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ================= HUME FUNCTIONS =================
def submit_hume_job(api_key, url):
    payload = {"urls": [url], "notify": False}

    r = requests.post(
        f"{HUME_BASE}/batch/jobs",
        headers={
            "X-Hume-Api-Key": api_key,
            "Authorization": f"Bearer {BEARER}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=30
    )
    r.raise_for_status()
    return r.json()["job_id"]


def get_hume_predictions(api_key, job_id, wait_sec=6):
    time.sleep(wait_sec)

    r = requests.get(
        f"{HUME_BASE}/batch/jobs/{job_id}/predictions",
        headers={"X-Hume-Api-Key": api_key},
        timeout=30
    )
    r.raise_for_status()
    return r.json()

# ================= PARSING =================

def extract_emotions(response_json):
    try:
        emotions = (
            response_json[0]["results"]["predictions"][0]
            ["models"]["face"]["grouped_predictions"][0]
            ["predictions"][0]["emotions"]
        )
        return {e["name"]: float(e["score"]) for e in emotions}
    except Exception:
        return None

# ================= RESUME =================

def load_processed_urls():
    processed = set()
    if Path(OUTPUT_CSV).exists():
        with open(OUTPUT_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["image_url"])
    return processed


def append_result(row, fieldnames):
    file_exists = Path(OUTPUT_CSV).exists()
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ================= KEY MANAGEMENT =================



def load_dead_keys():
    if not Path(DEAD_KEYS_FILE).exists():
        return set()
    return set(Path(DEAD_KEYS_FILE).read_text().splitlines())

DEAD_KEYS = load_dead_keys()


def mark_key_dead(api_key, reason):
    DEAD_KEYS.add(api_key)
    with open(DEAD_KEYS_FILE, "a") as f:
        f.write(api_key + "\n")
    logging.warning(f"API KEY DISABLED → {api_key[:6]}*** | Reason: {reason}")


def is_quota_error(err):
    msg = str(err).lower()
    return any(x in msg for x in [
        "402", "403", "quota", "usage limit", "payment required"
    ])

def mark_as_no_face(url):
    """Adds a URL to a blacklist so it's never tried again."""
    with open(FAILED_IMAGES_FILE, "a") as f:
        f.write(url + "\n")

# ================= MAIN =================

def main():
    processed_urls = load_processed_urls()

    # adding no face detacted images to processed set
    if Path(FAILED_IMAGES_FILE).exists():
        with open(FAILED_IMAGES_FILE, "r") as f:
            for line in f:
                processed_urls.add(line.strip())
    
    logging.info(f"Resuming. Already processed: {len(processed_urls)}")

    active_keys = [k for k in API_KEYS if k not in DEAD_KEYS]

    if not active_keys:
        logging.error("No active API keys available. Exiting.")
        return

    api_cycle = itertools.cycle(active_keys)

    with open(INPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader, start=1):
            url = row["uploaded_url"]

            if url in processed_urls:
                continue

            if not active_keys:
                logging.error("All API keys exhausted. STOPPING.")
                break

            api_key = next(api_cycle)

            try:
                job_id = submit_hume_job(api_key, url)
                response = get_hume_predictions(api_key, job_id)
                emotions = extract_emotions(response)

                if emotions is None:
                    mark_as_no_face(url)
                    raise ValueError("No face/emotion detected")

                append_result(
                    {
                        "image_url": url,
                        "api_key_used": api_key[:6] + "***",
                        "emotions_json": json.dumps(emotions)
                    },
                    fieldnames=["image_url", "api_key_used", "emotions_json"]
                )

                processed_urls.add(url)
                logging.info(f"[{idx}] SUCCESS → {url}")

            except Exception as e:
                if is_quota_error(e):
                    mark_key_dead(api_key, str(e))
                    active_keys = [k for k in active_keys if k != api_key]
                    api_cycle = itertools.cycle(active_keys)
                    logging.warning(f"Retrying image with next key → {url}")
                    continue
                else:
                    logging.error(f"[{idx}] FAILED → {url} | {str(e)}")

            time.sleep(REQUEST_SLEEP)

    logging.info("Processing finished.")

# ================= RUN =================

if __name__ == "__main__":
    main()