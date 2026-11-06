import pytest

from project_brain_mcp.brain import BrainError
from project_brain_mcp.history import pr_numbers


def test_index_reads_history_once_then_incrementally(brain, git_repo):
    first = brain.index(str(git_repo))
    again = brain.index(str(git_repo))

    assert (first.commits_added, first.commits_total) == (3, 3)
    assert again.commits_added == 0


def test_explain_history_of_a_symbol_lists_its_changes_newest_first(brain, git_repo):
    brain.index(str(git_repo))

    report = brain.explain_history(None, symbol="discount")

    assert (report.path, report.start_line, report.end_line) == ("pricing.py", 1, 2)
    subjects = [c.commit.subject for c in report.changes]
    assert subjects == ["Make discount rate configurable", "Round discounts to cents (#12)", "Add pricing helpers"]
    assert report.changes[1].commit.prs == (12,)
    assert "9.000001" in report.changes[1].commit.body
    assert "round(" in report.changes[1].diff
    assert report.blame[0].author == "Ada"
    assert report.pull_requests == []  # local repo: no GitHub remote, so no network calls


def test_explain_history_of_lines_ignores_unrelated_changes(brain, git_repo):
    brain.index(str(git_repo))

    report = brain.explain_history(None, path="pricing.py", start=5, end=6)

    assert [c.commit.subject for c in report.changes] == ["Add pricing helpers"]


def test_search_history_finds_commits_by_message(brain, git_repo):
    brain.index(str(git_repo))

    commits = brain.search_history("rounding cents")

    assert commits[0].subject == "Round discounts to cents (#12)"


def test_explain_history_errors_are_clear(brain, git_repo, sample_repo):
    brain.index(str(git_repo))
    with pytest.raises(BrainError, match="No function"):
        brain.explain_history("shop", symbol="nope")
    with pytest.raises(BrainError, match="isn't a file"):
        brain.explain_history("shop", path="../secret.txt")

    brain.index(str(sample_repo))
    with pytest.raises(BrainError, match="isn't a git repository"):
        brain.explain_history("sample-app", path="app/auth.py")


def test_pr_numbers_from_common_message_styles():
    assert pr_numbers("Fix login (#42)") == (42,)
    assert pr_numbers("Merge pull request #7 from user/branch") == (7,)
    assert pr_numbers("Fix issue 12 without a PR") == ()


def test_github_pr_lookup_is_cached_and_failures_fall_back_to_links(brain, git_repo, monkeypatch):
    from project_brain_mcp import history
    from project_brain_mcp.history import PullRequest

    calls = []

    def fake_fetch(slug, number, token):
        calls.append(number)
        return PullRequest(number=number, url=f"https://github.com/{slug}/pull/{number}", title="Round to cents")

    monkeypatch.setattr(history, "github_slug", lambda root: "acme/shop")
    monkeypatch.setattr(history, "fetch_pull_request", fake_fetch)
    brain.index(str(git_repo))

    first = brain.explain_history(None, symbol="discount")
    second = brain.explain_history(None, symbol="discount")

    assert first.pull_requests[0].title == "Round to cents" and second.pull_requests[0].title == "Round to cents"
    assert calls == [12]  # second call served from the cache


def test_symbol_resolves_to_the_implementation_not_overload_stubs(brain, git_repo):
    import subprocess

    (git_repo / "parse.py").write_text(
        "from typing import overload\n\n\n"
        "@overload\ndef parse(x: int) -> int: ...\n\n\n"
        "@overload\ndef parse(x: str) -> str: ...\n\n\n"
        "def parse(x):\n    if isinstance(x, int):\n        return x\n    return str(x)\n"
    )
    subprocess.run(["git", "add", "."], cwd=git_repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Ada", "-c", "user.email=ada@example.com", "commit", "-q", "-m", "Add parse"],
        cwd=git_repo,
        check=True,
    )
    brain.index(str(git_repo))

    report = brain.explain_history(None, symbol="parse")

    assert (report.start_line, report.end_line) == (12, 15)
