"""Flask application factory.

Keeps the Flask surface thin: this module only wires the app together and
registers routes. All business logic lives in plain modules (portfolio,
signals, rebalance, advice) so it can be tested without a running server.
"""

from __future__ import annotations

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)

    from app.routes import bp
    app.register_blueprint(bp)

    return app
