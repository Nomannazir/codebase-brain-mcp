import subprocess
import textwrap
from pathlib import Path

import pytest

from project_brain_mcp.brain import BrainError

CODE = {
    "shop/pricing.py": """
        def discount(price, rate=0.1):
            return round(price * (1 - rate), 2)


        def checkout_total(items):
            return sum(discount(i.price) for i in items)
        """,
    "shop/cart.py": """
        from shop.pricing import checkout_total


        class Cart:
            def __init__(self, items, currency="USD"):
                self.items = items
                self.currency = currency

            def total(self):
                return checkout_total(self.items)
        """,
    "shop/settings.py": """
        import os

        TIMEOUT = int(os.environ.get("SHOP_TIMEOUT", "30"))
        """,
    "tests/test_pricing.py": """
        from shop.pricing import discount


        def test_discount():
            assert discount(10) == 9.0
        """,
}
DOCS = """
    # Shop

    Prices go through `discount()` in `shop/pricing.py`. Totals come from `checkout_total`.

    ## Configuration

    Set `SHOP_TIMEOUT` to change the timeout, or `SHOP_RETRIES` for retries. Run with `--verbose`.

    ## Old API

    Call `apply_coupon()` from `shop/coupons.py`, or the renamed `checkout_totals()`.

    ```python
    from shop.pricing import discount, apply_coupon
    from shop.cart import Cart

    discount(100, percent=20)
    cart = Cart(items=[], currency="EUR")
    ```
    """


@pytest.fixture
def shop(tmp_path: Path) -> Path:
    root = tmp_path / "shop-repo"
    for rel, body in {**CODE, "README.md": DOCS}.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(body).lstrip())
    return root


# ---- impact_of_change ---------------------------------------------------------------------------


def test_impact_finds_direct_and_indirect_users_and_tests(brain, shop):
    brain.index(str(shop))

    report = brain.impact_of_change(None, "discount")

    direct = {(d.chunk.symbol, d.level) for d in report.dependents if d.level == 1}
    indirect = {d.chunk.symbol for d in report.dependents if d.level == 2}
    assert ("checkout_total", 1) in direct
    assert ("test_discount", 1) in direct
    assert "Cart" in indirect or "Cart.total" in indirect
    assert any(d.is_test for d in report.dependents)


def test_impact_depth_one_stops_at_direct_users(brain, shop):
    brain.index(str(shop))

    report = brain.impact_of_change(None, "discount", depth=1)

    assert {d.level for d in report.dependents} == {1}


def test_impact_of_unknown_symbol_is_a_clear_error(brain, shop):
    brain.index(str(shop))

    with pytest.raises(BrainError, match="No function"):
        brain.impact_of_change(None, "does_not_exist")


# ---- find_doc_drift -----------------------------------------------------------------------------


def _findings(brain, shop):
    brain.index(str(shop))
    report = brain.find_doc_drift(None, check_history=False)
    return report, {(f.kind, f.claim): f for f in report.findings}


def test_drift_ignores_claims_that_are_true(brain, shop):
    _, found = _findings(brain, shop)
    claims = {claim for _, claim in found}

    for true_claim in ("`discount()`", "`shop/pricing.py`", "`checkout_total`", "`SHOP_TIMEOUT`"):
        assert true_claim not in claims


def test_drift_flags_missing_files_symbols_settings_and_flags(brain, shop):
    _, found = _findings(brain, shop)

    assert found[("missing_file", "`shop/coupons.py`")].severity == "medium"
    assert found[("missing_symbol", "`apply_coupon()`")]
    renamed = found[("missing_symbol", "`checkout_totals()`")]
    assert renamed.severity == "high" and "checkout_total" in renamed.suggestion
    assert ("unknown_setting", "`SHOP_RETRIES`") in found
    assert ("unknown_flag", "`--verbose`") in found


def test_drift_checks_code_examples_against_real_signatures(brain, shop):
    _, found = _findings(brain, shop)

    wrong = found[("wrong_parameters", "discount(percent=…)")]
    assert wrong.severity == "high" and "price, rate" in wrong.detail
    assert ("broken_import", "from shop.pricing import apply_coupon") in found
    assert not any(kind == "wrong_parameters" and "Cart" in claim for kind, claim in found)  # valid call


def test_drift_cites_the_doc_section(brain, shop):
    _, found = _findings(brain, shop)

    assert found[("unknown_flag", "`--verbose`")].doc.startswith("README.md:")


def test_drift_flags_docs_older_than_the_code_they_describe(brain, shop):
    env = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "HOME": str(shop),
        "GIT_AUTHOR_NAME": "Ada",
        "GIT_AUTHOR_EMAIL": "a@example.com",
        "GIT_COMMITTER_NAME": "Ada",
        "GIT_COMMITTER_EMAIL": "a@example.com",
    }

    def git(*args, date=None):
        extra = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date} if date else {}
        subprocess.run(["git", *args], cwd=shop, check=True, capture_output=True, env={**env, **extra})

    git("init", "-q", "-b", "main")
    git("add", ".")
    git("commit", "-q", "-m", "initial", date="2024-01-01T00:00:00+00:00")
    pricing = shop / "shop" / "pricing.py"
    pricing.write_text(
        pricing.read_text().replace("round(price * (1 - rate), 2)", "max(0, round(price * (1 - rate), 2))")
    )
    git("commit", "-q", "-am", "Never return negative prices", date="2025-06-01T00:00:00+00:00")
    (shop / "CHANGES.md").write_text("# Changes\n\n- Removed `apply_discount_legacy()`.\n")
    brain.index(str(shop))

    report = brain.find_doc_drift(None)

    stale = [f for f in report.findings if f.kind == "docs_older_than_code"]
    assert stale and stale[0].severity == "low" and "Never return negative prices" in stale[0].detail


RST = """
Signing
=======

Use :class:`~shop.cart.Cart` and :func:`shop.pricing.discount`. The old
:func:`shop.pricing.discounted` is gone. ``checkout_total`` still works, and
``os.urandom`` is someone else's API.

.. code-block:: python

    discount(5, percentage=10)

>>> from shop.pricing import checkout_total
>>> checkout_total([])
0
"""


def test_drift_understands_restructuredtext(brain, shop):
    (shop / "docs").mkdir()
    (shop / "docs" / "guide.rst").write_text(textwrap.dedent(RST).lstrip())
    brain.index(str(shop))

    found = {
        (f.kind, f.claim): f for f in brain.find_doc_drift(None, path_prefix="docs/", check_history=False).findings
    }

    assert ("missing_symbol", "`shop.pricing.discounted`") in found
    assert ("wrong_parameters", "discount(percentage=…)") in found
    claims = {claim for _, claim in found}
    assert not any(name in c for c in claims for name in ("Cart", "checkout_total", "urandom"))


def test_changelogs_are_never_drift(brain, shop):
    (shop / "CHANGES.md").write_text("# Changes\n\n- Removed `apply_discount_legacy()` and `OldCart`.\n")
    brain.index(str(shop))

    report = brain.find_doc_drift(None, check_history=False)

    assert not any(f.doc.startswith("CHANGES.md") for f in report.findings)


def test_constructor_calls_are_checked_against_init_case_sensitively(brain, shop):
    (shop / "tests" / "test_cart.py").write_text("def cart(fixture):\n    return fixture\n")
    brain.index(str(shop))

    found = {(f.kind, f.claim) for f in brain.find_doc_drift(None, check_history=False).findings}

    assert not any(kind == "wrong_parameters" and claim.startswith("Cart(") for kind, claim in found)
