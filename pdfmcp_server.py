#!/usr/bin/env python3
"""Launcher for the PDF extraction MCP server.

Kept at the repository root so it can be registered with an absolute path
(the workbench's MCP registry is shared between host and container, where the
working directory differs). Usage::

    python pdfmcp_server.py            # stdio (the workbench's MCP client)
    python pdfmcp_server.py --http     # streamable-http on :8500/mcp
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdfmcp.server import main  # noqa: E402

if __name__ == "__main__":
    main()
