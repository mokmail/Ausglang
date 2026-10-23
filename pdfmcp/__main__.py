"""Allow ``python -m pdfmcp`` and ``python pdfmcp/server.py`` to start the server."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    # make the repo root importable when this file is run directly
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from pdfmcp.server import main

    main()
