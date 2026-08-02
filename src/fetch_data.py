"""Download and verify the Azure LLM inference traces.

Resumable and idempotent: re-running only fetches what is missing or short.
Verification is on exact byte size from the GitHub release metadata, because a
truncated CSV still parses and would silently corrupt every downstream number.
"""

import sys
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
BASE = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024"

ASSETS = {
    "AzureLLMInferenceTrace_conv_1week.csv": 1135195393,
    "AzureLLMInferenceTrace_code_1week.csv": 691989454,
}

CHUNK = 1 << 20


def fetch(name: str, expected: int, attempts: int = 6) -> bool:
    dest = RAW / name
    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have == expected:
            print(f"OK    {name}  {have} bytes", flush=True)
            return True
        if have > expected:
            print(f"      {name} oversized ({have}); restarting", flush=True)
            dest.unlink()
            have = 0

        req = urllib.request.Request(f"{BASE}/{name}")
        mode = "wb"
        if have:
            req.add_header("Range", f"bytes={have}-")
            mode = "ab"
        print(f"      attempt {attempt}: {name} from byte {have}", flush=True)

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                # A server that ignores Range replies 200 and restarts the body.
                if have and resp.status != 206:
                    mode, have = "wb", 0
                start, last = time.time(), 0
                with open(dest, mode) as fh:
                    while True:
                        buf = resp.read(CHUNK)
                        if not buf:
                            break
                        fh.write(buf)
                        have += len(buf)
                        if have - last > 100 * CHUNK:
                            last = have
                            pct = 100 * have / expected
                            rate = have / (time.time() - start) / 1e6
                            print(f"        {pct:5.1f}%  {rate:.1f} MB/s", flush=True)
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            print(f"        interrupted: {exc}", flush=True)
            time.sleep(3)

    final = dest.stat().st_size if dest.exists() else 0
    print(f"FAIL  {name}  {final}/{expected}", flush=True)
    return False


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    ok = all(fetch(name, size) for name, size in ASSETS.items())
    print("=== ALL VERIFIED ===" if ok else "=== INCOMPLETE ===", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
