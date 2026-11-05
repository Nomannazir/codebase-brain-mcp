import textwrap
from pathlib import Path

import pytest

from project_brain_mcp.brain import Brain
from project_brain_mcp.config import Settings
from project_brain_mcp.embeddings import HashEmbedder

SAMPLE_FILES = {
    "app/auth.py": '''
        """Login and session handling."""
        import hashlib

        SESSION_TTL_SECONDS = 3600


        def hash_password(password: str, salt: str) -> str:
            """Hash a password with a per-user salt."""
            return hashlib.sha256((salt + password).encode()).hexdigest()


        class SessionStore:
            """Keeps active sessions in memory."""

            def __init__(self):
                self.sessions = {}

            def create_session(self, user_id: str) -> str:
                token = hashlib.sha256(user_id.encode()).hexdigest()
                self.sessions[token] = user_id
                return token
        ''',
    "app/billing.py": '''
        def calculate_invoice_total(items, tax_rate):
            """Sum the line items and add tax."""
            subtotal = sum(item.price * item.quantity for item in items)
            return round(subtotal * (1 + tax_rate), 2)
        ''',
    "README.md": '''
        # Sample App

        A tiny app used to test Project Brain.

        ## Authentication

        Passwords are hashed with a per-user salt. Sessions expire after one hour.

        ## Billing

        Invoices include tax.
        ''',
    "node_modules/ignored.js": "console.log('should not be indexed')",
}


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "sample-app"
    for rel, body in SAMPLE_FILES.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(body).lstrip())
    return root


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path / "home", embedder="hash", embed_model="unused", max_file_bytes=100_000)


@pytest.fixture
def brain(settings: Settings):
    brain = Brain(settings, HashEmbedder())
    yield brain
    brain.close()


@pytest.fixture
def anyio_backend():
    return "asyncio"
