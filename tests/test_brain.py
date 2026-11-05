import pytest

from project_brain_mcp.brain import BrainError


def test_index_skips_ignored_dirs_and_reports_counts(brain, sample_repo):
    report = brain.index(str(sample_repo))

    assert report.repo == "sample-app"
    assert report.files_seen == 3  # node_modules is skipped
    assert report.files_indexed == 3 and report.chunks_total > 3


def test_reindex_only_touches_changed_and_removed_files(brain, sample_repo):
    brain.index(str(sample_repo))
    (sample_repo / "app" / "billing.py").write_text("def refund():\n    return 0\n")
    (sample_repo / "README.md").unlink()

    report = brain.index(str(sample_repo))

    assert (report.files_indexed, report.files_unchanged, report.files_removed) == (1, 1, 1)
    assert not report.full_rebuild


def test_search_finds_the_function_and_cites_lines(brain, sample_repo):
    brain.index(str(sample_repo))

    hits = brain.search("how is the password hashed", k=3)

    assert hits[0].chunk.symbol == "hash_password"
    assert hits[0].chunk.citation.startswith("app/auth.py:")
    assert hits[0].keyword_rank is not None


def test_keyword_search_matches_identifier_parts(brain, sample_repo):
    brain.index(str(sample_repo))

    hits = brain.search("invoice total", k=1)

    assert hits[0].chunk.symbol == "calculate_invoice_total"


def test_filters_by_path_and_kind(brain, sample_repo):
    brain.index(str(sample_repo))

    docs = brain.search("password", path_prefix="README", k=5)
    methods = brain.search("session", kinds=["method"], k=5)

    assert docs and all(h.chunk.path == "README.md" for h in docs)
    assert all(h.chunk.kind == "method" for h in methods)


def test_read_lines_refuses_paths_outside_the_repo(brain, sample_repo):
    brain.index(str(sample_repo))

    assert "def hash_password" in brain.read_lines(None, "app/auth.py", 1, 10)
    with pytest.raises(BrainError):
        brain.read_lines(None, "../../etc/passwd", 1, 5)


def test_repo_is_required_when_several_are_indexed(brain, sample_repo, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.py").write_text("def x():\n    return 1\n")
    brain.index(str(sample_repo))
    brain.index(str(other))

    with pytest.raises(BrainError, match="Several repos"):
        brain.search("x")
    assert brain.search("x", repo="other")


def test_bad_names_and_sources_are_rejected(brain, tmp_path):
    from project_brain_mcp.sources import SourceError

    with pytest.raises(BrainError):
        brain.index(str(tmp_path), name="../escape")
    with pytest.raises(SourceError):
        brain.index("https://example.com/repo.git --upload-pack=evil")
    with pytest.raises(SourceError):
        brain.index(str(tmp_path / "missing"))
