"""Retrieval evaluation: does search put the right code in front of the model?

A dataset is a TOML file of queries, each with the chunks a good answer must come from, against a
repository pinned to one commit. For every search mode we measure where the first relevant result
lands, then report standard IR metrics with bootstrap confidence intervals, per-category breakdowns,
paired win/tie/loss counts between modes, and search latency.
"""

from __future__ import annotations

import math
import random
import statistics
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import sources
from .brain import SEARCH_MODES, Brain, SearchMode
from .chunking import DEFINITION_KINDS
from .store import StoredChunk

CUTOFFS = (1, 3, 5, 10)
DEPTH = 10
"""How many results each query retrieves; every metric is computed at or below this depth."""
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 7


class DatasetError(ValueError):
    """The dataset file is malformed, or names code that doesn't exist at the pinned commit."""


@dataclass(frozen=True)
class Target:
    path: str
    symbol: str | None = None

    def matches(self, chunk: StoredChunk) -> bool:
        if chunk.path != self.path:
            return False
        if self.symbol is None:
            return True
        # A method belongs to its class ("Signer.sign"), a subsection to its section ("API > retry").
        symbol = chunk.symbol or ""
        return symbol == self.symbol or symbol.startswith((f"{self.symbol}.", f"{self.symbol} > "))


@dataclass(frozen=True)
class Query:
    id: str
    category: str
    text: str
    targets: tuple[Target, ...]


@dataclass(frozen=True)
class Dataset:
    name: str
    repo: str
    commit: str
    queries: tuple[Query, ...]


@dataclass(frozen=True)
class QueryResult:
    query: Query
    mode: str
    first_relevant_rank: int | None
    """1-based rank of the first relevant result within DEPTH, or None if none was found."""
    targets_found: int
    ndcg: float
    latency_ms: float
    top: tuple[str, ...]
    """Citations of the top results, for reading failures."""

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_relevant_rank if self.first_relevant_rank else 0.0

    def hit(self, k: int) -> bool:
        return self.first_relevant_rank is not None and self.first_relevant_rank <= k

    @property
    def recall(self) -> float:
        return self.targets_found / len(self.query.targets)


@dataclass(frozen=True)
class ModeSummary:
    mode: str
    queries: int
    hit_at: dict[int, float]
    mrr: float
    ndcg: float
    recall: float
    mrr_ci: tuple[float, float]
    hit5_ci: tuple[float, float]
    latency_p50_ms: float
    latency_p95_ms: float
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalReport:
    dataset: Dataset
    embedder: str
    summaries: dict[str, ModeSummary]
    results: dict[str, list[QueryResult]]
    paired: dict[str, tuple[int, int, int]]
    """'hybrid vs keyword' -> (wins, ties, losses) by reciprocal rank, per query."""


# ---- dataset ----------------------------------------------------------------------------------


def load_dataset(path: Path) -> Dataset:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        queries = tuple(
            Query(
                id=q["id"],
                category=q["category"],
                text=q["text"],
                targets=tuple(Target(path=t["path"], symbol=t.get("symbol")) for t in q["targets"]),
            )
            for q in raw["query"]
        )
        dataset = Dataset(name=raw["name"], repo=raw["repo"], commit=raw["commit"], queries=queries)
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        raise DatasetError(f"Can't read dataset {path}: {exc}") from exc
    ids = [q.id for q in dataset.queries]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise DatasetError(f"Duplicate query ids: {', '.join(duplicates)}")
    if empty := [q.id for q in dataset.queries if not q.targets]:
        raise DatasetError(f"Queries without targets: {', '.join(empty)}")
    return dataset


def prepare(brain: Brain, dataset: Dataset) -> str:
    """Clone the repo, check out the pinned commit, index it, and check every target exists there.
    Returns the index name."""
    root = sources.resolve(dataset.repo, brain.settings.clone_dir / "evals")
    sources.run_git(root, "checkout", "-q", "--detach", dataset.commit)
    name = f"eval-{dataset.name}"
    brain.index(str(root), name=name)
    validate_targets(brain, name, dataset)
    return name


def validate_targets(brain: Brain, repo: str, dataset: Dataset) -> None:
    """Fail loudly if the dataset names files or symbols that aren't in the index (dataset rot)."""
    store = brain.store_for(repo)
    files = store.file_paths()
    symbols = set(store.symbols((*DEFINITION_KINDS, "section")))
    missing = []
    for query in dataset.queries:
        for target in query.targets:
            if target.path not in files:
                missing.append(f"{query.id}: no file {target.path}")
            elif target.symbol and (target.symbol, target.path) not in symbols:
                missing.append(f"{query.id}: no symbol {target.symbol} in {target.path}")
    if missing:
        raise DatasetError("Dataset doesn't match the code:\n  " + "\n  ".join(missing))


# ---- running ----------------------------------------------------------------------------------


def run(brain: Brain, repo: str, dataset: Dataset, modes: Sequence[SearchMode] = SEARCH_MODES) -> EvalReport:
    brain.search("warm up", repo=repo, k=1)  # load the model before timing anything
    results: dict[str, list[QueryResult]] = {}
    for mode in modes:
        results[mode] = [_run_query(brain, repo, query, mode) for query in dataset.queries]
    summaries = {mode: summarise(mode, rows) for mode, rows in results.items()}
    paired = {}
    if "hybrid" in results:
        for other in modes:
            if other != "hybrid":
                paired[f"hybrid vs {other}"] = compare(results["hybrid"], results[other])
    return EvalReport(
        dataset=dataset, embedder=brain.embedder.model_id, summaries=summaries, results=results, paired=paired
    )


def _run_query(brain: Brain, repo: str, query: Query, mode: SearchMode) -> QueryResult:
    started = time.perf_counter()
    hits = brain.search(query.text, repo=repo, k=DEPTH, mode=mode)
    latency = (time.perf_counter() - started) * 1000
    chunks = [hit.chunk for hit in hits]
    return QueryResult(
        query=query,
        mode=mode,
        first_relevant_rank=first_relevant_rank(chunks, query.targets),
        targets_found=sum(any(t.matches(c) for c in chunks) for t in query.targets),
        ndcg=ndcg(chunks, query.targets, DEPTH),
        latency_ms=latency,
        top=tuple(c.citation for c in chunks[:3]),
    )


# ---- metrics (pure functions) ------------------------------------------------------------------


def first_relevant_rank(chunks: Sequence[StoredChunk], targets: Sequence[Target]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if any(t.matches(chunk) for t in targets):
            return rank
    return None


def ndcg(chunks: Sequence[StoredChunk], targets: Sequence[Target], k: int) -> float:
    """Binary-gain nDCG@k. Each target earns credit once, at the first result that matches it, so
    returning five chunks of the same function doesn't inflate the score."""
    credited: set[int] = set()
    dcg = 0.0
    for rank, chunk in enumerate(chunks[:k], start=1):
        newly = [i for i, t in enumerate(targets) if i not in credited and t.matches(chunk)]
        if newly:
            credited.add(newly[0])
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(targets), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_ci(values: Sequence[float], samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED):
    """95% percentile-bootstrap confidence interval of the mean, resampling queries."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)  # noqa: S311 (statistics, not security: a seeded RNG keeps CIs reproducible)
    n = len(values)
    means = [statistics.fmean(rng.choices(values, k=n)) for _ in range(samples)]
    return (percentile(means, 2.5), percentile(means, 97.5))


def summarise(mode: str, rows: Sequence[QueryResult]) -> ModeSummary:
    rr = [r.reciprocal_rank for r in rows]
    hit5 = [float(r.hit(5)) for r in rows]
    latencies = [r.latency_ms for r in rows]
    by_category: dict[str, dict[str, float]] = {}
    for category in sorted({r.query.category for r in rows}):
        subset = [r for r in rows if r.query.category == category]
        by_category[category] = {
            "queries": len(subset),
            "hit@1": statistics.fmean(float(r.hit(1)) for r in subset),
            "hit@5": statistics.fmean(float(r.hit(5)) for r in subset),
            "mrr": statistics.fmean(r.reciprocal_rank for r in subset),
        }
    return ModeSummary(
        mode=mode,
        queries=len(rows),
        hit_at={k: statistics.fmean(float(r.hit(k)) for r in rows) for k in CUTOFFS},
        mrr=statistics.fmean(rr),
        ndcg=statistics.fmean(r.ndcg for r in rows),
        recall=statistics.fmean(r.recall for r in rows),
        mrr_ci=bootstrap_ci(rr),
        hit5_ci=bootstrap_ci(hit5),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        by_category=by_category,
    )


def compare(a: Sequence[QueryResult], b: Sequence[QueryResult]) -> tuple[int, int, int]:
    """Per-query wins, ties and losses of ``a`` against ``b`` by reciprocal rank."""
    wins = ties = losses = 0
    for x, y in zip(a, b, strict=True):
        if math.isclose(x.reciprocal_rank, y.reciprocal_rank):
            ties += 1
        elif x.reciprocal_rank > y.reciprocal_rank:
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


# ---- reporting --------------------------------------------------------------------------------


def to_markdown(report: EvalReport) -> str:
    d = report.dataset
    lines = [
        f"## Retrieval eval: {d.name} @ `{d.commit[:10]}`",
        "",
        f"{len(d.queries)} queries · embedder `{report.embedder}` · top {DEPTH} results per query · "
        f"95% bootstrap CIs over queries ({BOOTSTRAP_SAMPLES} resamples)",
        "",
        "| Mode | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 (95% CI) | nDCG@10 | Recall@10 | p50 / p95 latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in report.summaries.values():
        lines.append(
            f"| **{s.mode}** | {_pct(s.hit_at[1])} | {_pct(s.hit_at[3])} | {_pct(s.hit_at[5])} | "
            f"{_pct(s.hit_at[10])} | {s.mrr:.3f} ({s.mrr_ci[0]:.3f}–{s.mrr_ci[1]:.3f}) | {s.ndcg:.3f} | "
            f"{_pct(s.recall)} | {s.latency_p50_ms:.0f} / {s.latency_p95_ms:.0f} ms |"
        )
    categories = sorted({c for s in report.summaries.values() for c in s.by_category})
    lines += [
        "",
        "**By query type** (Hit@5 / MRR)",
        "",
        "| Mode | "
        + " | ".join(
            f"{c} ({int(next(iter(report.summaries.values())).by_category[c]['queries'])})" for c in categories
        )
        + " |",
        "|---|" + "---|" * len(categories),
    ]
    for s in report.summaries.values():
        cells = [f"{_pct(s.by_category[c]['hit@5'])} / {s.by_category[c]['mrr']:.2f}" for c in categories]
        lines.append(f"| {s.mode} | " + " | ".join(cells) + " |")
    if report.paired:
        lines += ["", "**Paired, per query** (by reciprocal rank)", ""]
        for name, (wins, ties, losses) in report.paired.items():
            lines.append(f"- {name}: {wins} better, {ties} same, {losses} worse")
    misses = [r for r in report.results.get("hybrid", []) if not r.hit(5)]
    if misses:
        lines += ["", "**Hybrid misses (not in top 5)**", ""]
        for r in misses:
            got = ", ".join(f"`{c}`" for c in r.top) or "nothing"
            lines.append(f'- `{r.query.id}` "{r.query.text}": top results {got}')
    return "\n".join(lines) + "\n"


def to_json(report: EvalReport) -> dict:
    return {
        "dataset": report.dataset.name,
        "commit": report.dataset.commit,
        "embedder": report.embedder,
        "depth": DEPTH,
        "summaries": {
            mode: {
                "hit_at": {str(k): v for k, v in s.hit_at.items()},
                "mrr": s.mrr,
                "mrr_ci": list(s.mrr_ci),
                "hit5_ci": list(s.hit5_ci),
                "ndcg": s.ndcg,
                "recall": s.recall,
                "latency_p50_ms": s.latency_p50_ms,
                "latency_p95_ms": s.latency_p95_ms,
                "by_category": s.by_category,
            }
            for mode, s in report.summaries.items()
        },
        "paired": {name: dict(zip(("wins", "ties", "losses"), v, strict=True)) for name, v in report.paired.items()},
        "queries": {
            mode: [
                {
                    "id": r.query.id,
                    "category": r.query.category,
                    "rank": r.first_relevant_rank,
                    "ndcg": round(r.ndcg, 4),
                    "top": list(r.top),
                }
                for r in rows
            ]
            for mode, rows in report.results.items()
        },
    }


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"
