import math
import textwrap

import pytest

from codebase_brain_mcp import evals
from codebase_brain_mcp.evals import Dataset, DatasetError, Query, Target
from codebase_brain_mcp.store import StoredChunk


def chunk(path, symbol=None, start=1):
    return StoredChunk(
        id=start, path=path, start_line=start, end_line=start + 5, kind="function", symbol=symbol, text=""
    )


# ---- matching and metrics -----------------------------------------------------------------------


def test_target_matches_symbol_its_methods_or_whole_file():
    assert Target("a.py", "Signer").matches(chunk("a.py", "Signer.sign"))
    assert not Target("a.py", "Signer").matches(chunk("a.py", "SignerFactory"))
    assert not Target("a.py", "Signer").matches(chunk("b.py", "Signer"))
    assert Target("docs/x.rst").matches(chunk("docs/x.rst"))


def test_target_matches_a_doc_section_and_its_subsections():
    assert Target("readme.md", "API > retry").matches(chunk("readme.md", "API > retry > delay"))
    assert not Target("readme.md", "API > retry").matches(chunk("readme.md", "API > retryOptions"))


def test_first_relevant_rank_and_reciprocal_rank():
    ranked = [chunk("x.py", "a"), chunk("y.py", "b"), chunk("z.py", "c")]

    assert evals.first_relevant_rank(ranked, [Target("y.py", "b")]) == 2
    assert evals.first_relevant_rank(ranked, [Target("q.py")]) is None


def test_ndcg_credits_each_target_once():
    targets = [Target("a.py", "f"), Target("b.py", "g")]
    perfect = [chunk("a.py", "f"), chunk("b.py", "g")]
    duplicated = [chunk("a.py", "f"), chunk("a.py", "f", 9), chunk("b.py", "g")]

    assert evals.ndcg(perfect, targets, 10) == pytest.approx(1.0)
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert evals.ndcg(duplicated, targets, 10) == pytest.approx(expected)
    assert evals.ndcg([], targets, 10) == 0.0


def test_percentile_and_bootstrap_are_deterministic():
    assert evals.percentile([1, 2, 3, 4], 50) == 2.5
    low, high = evals.bootstrap_ci([0.0, 1.0] * 20)
    assert 0.3 < low < 0.5 < high < 0.7
    assert evals.bootstrap_ci([0.0, 1.0] * 20) == (low, high)
    assert evals.bootstrap_ci([1.0] * 10) == (1.0, 1.0)


# ---- dataset ------------------------------------------------------------------------------------


def write(tmp_path, body):
    path = tmp_path / "d.toml"
    path.write_text(textwrap.dedent(body))
    return path


def test_load_dataset_and_reject_duplicates(tmp_path):
    good = write(
        tmp_path,
        """
        name = "x"
        repo = "https://example.com/x"
        commit = "abc"
        [[query]]
        id = "q1"
        category = "identifier"
        text = "foo"
        targets = [{ path = "a.py", symbol = "foo" }]
        """,
    )
    dataset = evals.load_dataset(good)
    assert dataset.queries[0].targets == (Target("a.py", "foo"),)

    dup = write(
        tmp_path,
        good.read_text()
        + """
[[query]]
id = "q1"
category = "docs"
text = "bar"
targets = [{ path = "b.md" }]
""",
    )
    with pytest.raises(DatasetError, match="Duplicate"):
        evals.load_dataset(dup)


def test_dataset_rot_is_caught(brain, sample_repo):
    brain.index(str(sample_repo))
    dataset = Dataset(
        name="s",
        repo="",
        commit="",
        queries=(
            Query("ok", "identifier", "hash", (Target("app/auth.py", "hash_password"),)),
            Query("gone", "identifier", "x", (Target("app/auth.py", "deleted_function"),)),
        ),
    )

    with pytest.raises(DatasetError, match="no symbol deleted_function"):
        evals.validate_targets(brain, "sample-app", dataset)


# ---- end to end (offline: hash embedder, local repo) -------------------------------------------


def test_run_reports_every_mode_and_renders(brain, sample_repo):
    brain.index(str(sample_repo))
    dataset = Dataset(
        name="sample",
        repo="",
        commit="0" * 40,
        queries=(
            Query(
                "id", "identifier", "calculate_invoice_total", (Target("app/billing.py", "calculate_invoice_total"),)
            ),
            Query("bh", "behaviour", "how are passwords hashed", (Target("app/auth.py", "hash_password"),)),
            Query("doc", "docs", "when do sessions expire", (Target("README.md"),)),
        ),
    )

    report = evals.run(brain, "sample-app", dataset)

    assert set(report.summaries) == {"hybrid", "keyword", "vector"}
    assert report.summaries["hybrid"].hit_at[5] == 1.0
    assert sum(report.paired["hybrid vs keyword"]) == 3
    markdown = evals.to_markdown(report)
    assert "| **hybrid** |" in markdown and "By query type" in markdown
    assert evals.to_json(report)["queries"]["hybrid"][0]["rank"] == 1
