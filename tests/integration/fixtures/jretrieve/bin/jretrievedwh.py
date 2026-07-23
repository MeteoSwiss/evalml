#!/usr/bin/env python3
"""Mock jretrievedwh.py binary for integration tests.

Returns pre-stored fixture CSVs instead of contacting the DWH.
The fixture directory is read from JRETRIEVE_FIXTURE_DIR.
"""

import os
import sys
from pathlib import Path

fixture_dir = Path(os.environ["JRETRIEVE_FIXTURE_DIR"])
call_type = "meta" if "--meta-info" in sys.argv else "data"
sys.stdout.write((fixture_dir / f"{call_type}.csv").read_text())
