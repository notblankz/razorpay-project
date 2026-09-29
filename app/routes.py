"""HTTP endpoints (thin wrappers around the logic modules).

Routes here should stay small: parse the request, call a logic function,
return JSON. No calculations belong in this file.
"""

from __future__ import annotations

from flask import Blueprint, jsonify

bp = Blueprint("api", __name__)


@bp.get("/health")
def health():
    """Liveness check."""
    return jsonify({"status": "ok"})
