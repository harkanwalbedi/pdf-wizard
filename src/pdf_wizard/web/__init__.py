"""PDF Wizard web UI (local, single-user mode). Start it with `pdf-wizard serve`."""

from .app import create_app

__all__ = ["create_app"]
