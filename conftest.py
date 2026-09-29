"""Pytest bootstrap.

Ensures the repo root is on sys.path so tests can import both the `app`
package and the `scripts` modules regardless of the working directory pytest
is launched from.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
