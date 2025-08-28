from __future__ import annotations

import re
import sys
from pathlib import Path


def check_algocore_version(filename: str) -> int:
    with Path(filename).open() as file:
        content = file.read()

    # Look for algocore in INSTALL_REQUIRES
    match = re.search(r'("algocore[^"]*")', content)
    if not match:
        print(f"Error: algocore not found in {filename}")
        return 1

    algocore_req = match[1]
    if not re.match(r'"algocore==\d+\.\d+\.\d+"', algocore_req):
        print(f"Error: algocore version must be exact (==) in {filename}. Found: {algocore_req}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(check_algocore_version("setup.py"))
