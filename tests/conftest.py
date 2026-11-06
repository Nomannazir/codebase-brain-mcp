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
    "README.md": """
        # Sample App

        A tiny app used to test Project Brain.

        ## Authentication

        Passwords are hashed with a per-user salt. Sessions expire after one hour.

        ## Billing

        Invoices include tax.
        """,
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


def _git(root: Path, *args: str) -> None:
    import subprocess

    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "Ada",
            "GIT_AUTHOR_EMAIL": "ada@example.com",
            "GIT_COMMITTER_NAME": "Ada",
            "GIT_COMMITTER_EMAIL": "ada@example.com",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "HOME": str(root),
        },
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A repo where `discount` is written, then changed twice for stated reasons."""
    root = tmp_path / "shop"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "pricing.py").write_text(
        "def discount(price):\n    return price * 0.9\n\n\ndef tax(price):\n    return price * 0.2\n"
    )
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "Add pricing helpers")
    (root / "pricing.py").write_text(
        "def discount(price):\n    return round(price * 0.9, 2)\n\n\ndef tax(price):\n    return price * 0.2\n"
    )
    _git(root, "commit", "-q", "-am", "Round discounts to cents (#12)", "-m", "Customers saw prices like 9.000001.")
    (root / "pricing.py").write_text(
        "def discount(price, rate=0.1):\n    return round(price * (1 - rate), 2)\n\n\ndef tax(price):\n    return price * 0.2\n"
    )
    _git(root, "commit", "-q", "-am", "Make discount rate configurable")
    return root
