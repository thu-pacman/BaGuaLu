"""Analyze per-rank MoE token pressure from exported imbalance logs.

Structured ``moe_token_event`` alloc/free lines can reconstruct exact live-token
deltas per rank. Legacy ``step/layer/rank/received_tokens`` lines are still
accepted and fall back to the rolling-window pressure estimate.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


DEFAULT_LOG_PATH = Path("outputs/moe-imbalance-minimal-187619.out")
DEFAULT_OUTPUT_DIR = Path("outputs/moe_token_pressure_187619")

MOE_TOKEN_EVENT_RE = re.compile(
    r"moe_token_event:\s*"
    r"event_id=(?P<event_id>\d+),\s*"
    r"step=(?P<step>[^,\s]+),\s*"
    r"iteration=(?P<iteration>[^,\s]+),\s*"
    r"microbatch=(?P<microbatch>[^,\s]+),\s*"
    r"phase=(?P<phase>[^,\s]+),\s*"
    r"event=(?P<event>[^,\s]+),\s*"
    r"layer=(?P<layer>[^,\s]+),\s*"
    r"rank=(?P<rank>\d+),\s*"
    r"received_tokens=(?P<tokens>\d+)\s*$"
)
TOKEN_LINE_RE = re.compile(
    r"\|\s*step:\s*(?P<step>\d+),\s*"
    r"layer:\s*(?P<layer>\d+),\s*"
    r"rank:\s*(?P<rank>\d+),\s*"
    r"received_tokens:\s*(?P<tokens>\d+)\s*$"
)
SUMMARY_LINE_RE = re.compile(
    r"\|\s*step:\s*(?P<step>\d+),\s*"
    r"layer:\s*(?P<layer>\d+),\s*"
    r"average_received_tokens:\s*(?P<average>[+\-0-9.eEinfINF]+),\s*"
    r"max_received_tokens:\s*(?P<max>\d+),\s*"
    r"imbalance_ratio:\s*(?P<ratio>[+\-0-9.eEinfINF]+)\s*$"
)


@dataclass(frozen=True)
class TokenRecord:
    line_number: int
    step: int
    layer: int
    rank: int
    received_tokens: int
    iteration: Optional[int] = None
    microbatch: Optional[int] = None
    phase: Optional[str] = None
    event: Optional[str] = None
    event_id: Optional[int] = None


@dataclass(frozen=True)
class LayerSummary:
    line_number: int
    step: int
    layer: int
    average_received_tokens: float
    max_received_tokens: int
    imbalance_ratio: float


@dataclass(frozen=True)
class ParseResult:
    token_records: list[TokenRecord]
    layer_summaries: list[LayerSummary]
    skipped_metric_lines: int


@dataclass(frozen=True)
class EventWindow:
    rank: int
    window_index: int
    record_count: int
    token_sum: int
    start_line: int
    end_line: int
    start_step: int
    start_layer: int
    end_step: int
    end_layer: int
    steps: tuple[int, ...]
    layers: tuple[int, ...]
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class RankSummary:
    rank: int
    records: int
    steps: int
    event_windows: int
    event_window_size: int
    event_peak_tokens: int
    event_peak_start_line: int
    event_peak_end_line: int
    event_peak_start_step: int
    event_peak_start_layer: int
    event_peak_end_step: int
    event_peak_end_layer: int
    event_peak_layers: tuple[int, ...]
    event_peak_token_values: tuple[int, ...]
    event_trough_tokens: int
    event_trough_start_step: int
    event_trough_start_layer: int
    event_trough_end_step: int
    event_trough_end_layer: int
    event_trough_layers: tuple[int, ...]
    event_trough_token_values: tuple[int, ...]
    event_mean_tokens: float
    event_peak_to_mean_ratio: float
    event_peak_to_trough_ratio: float


@dataclass(frozen=True)
class LiveTokenSample:
    line_number: int
    rank: int
    event_id: int
    step: int
    iteration: Optional[int]
    microbatch: Optional[int]
    phase: str
    event: str
    layer: int
    delta_tokens: int
    live_tokens: int


@dataclass(frozen=True)
class LiveRankSummary:
    rank: int
    events: int
    peak_live_tokens: int
    peak_line: int
    peak_event_id: int
    peak_step: int
    peak_iteration: Optional[int]
    peak_microbatch: Optional[int]
    peak_layer: int
    min_live_tokens: int
    final_live_tokens: int


@dataclass(frozen=True)
class LifecycleIssue:
    rank: int
    event_id: int
    step: int
    iteration: Optional[int]
    microbatch: Optional[int]
    layer: int
    alloc_tokens: int
    free_tokens: int
    net_tokens: int
    alloc_count: int
    free_count: int
    first_alloc_line: int
    last_alloc_line: int
    first_free_line: int
    last_free_line: int
    status: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_from_repo(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def parse_float(value: str) -> float:
    if value.lower() == "inf":
        return float("inf")
    if value.lower() == "-inf":
        return float("-inf")
    return float(value)


def parse_int_field(value: str, default: Optional[int] = None) -> Optional[int]:
    if value.lower() == "unknown":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def parse_moe_log(log_path: Path) -> ParseResult:
    token_records: list[TokenRecord] = []
    layer_summaries: list[LayerSummary] = []
    skipped_metric_lines = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if "received_tokens" not in line and "imbalance_ratio" not in line:
                continue

            event_match = MOE_TOKEN_EVENT_RE.search(line)
            if event_match:
                step = parse_int_field(event_match.group("step"), default=0)
                iteration = parse_int_field(event_match.group("iteration"), default=step)
                layer = parse_int_field(event_match.group("layer"), default=-1)
                token_records.append(
                    TokenRecord(
                        line_number=line_number,
                        step=step or 0,
                        iteration=iteration,
                        microbatch=parse_int_field(
                            event_match.group("microbatch"), default=None
                        ),
                        phase=event_match.group("phase"),
                        event=event_match.group("event"),
                        event_id=int(event_match.group("event_id")),
                        layer=layer if layer is not None else -1,
                        rank=int(event_match.group("rank")),
                        received_tokens=int(event_match.group("tokens")),
                    )
                )
                continue

            token_match = TOKEN_LINE_RE.search(line)
            if token_match:
                token_records.append(
                    TokenRecord(
                        line_number=line_number,
                        step=int(token_match.group("step")),
                        layer=int(token_match.group("layer")),
                        rank=int(token_match.group("rank")),
                        received_tokens=int(token_match.group("tokens")),
                    )
                )
                continue

            summary_match = SUMMARY_LINE_RE.search(line)
            if summary_match:
                layer_summaries.append(
                    LayerSummary(
                        line_number=line_number,
                        step=int(summary_match.group("step")),
                        layer=int(summary_match.group("layer")),
                        average_received_tokens=parse_float(summary_match.group("average")),
                        max_received_tokens=int(summary_match.group("max")),
                        imbalance_ratio=parse_float(summary_match.group("ratio")),
                    )
                )
                continue

            skipped_metric_lines += 1

    return ParseResult(
        token_records=token_records,
        layer_summaries=layer_summaries,
        skipped_metric_lines=skipped_metric_lines,
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def records_by_rank(records: Iterable[TokenRecord]) -> dict[int, list[TokenRecord]]:
    grouped: dict[int, list[TokenRecord]] = defaultdict(list)
    for record in records:
        grouped[record.rank].append(record)
    return dict(grouped)


def pressure_records(records: Iterable[TokenRecord]) -> list[TokenRecord]:
    records = list(records)
    has_structured_events = any(record.event is not None for record in records)
    return [
        record
        for record in records
        if (
            record.event in {"alloc", "dispatch"}
            if has_structured_events
            else record.event is None
        )
    ]


def lifecycle_records(records: Iterable[TokenRecord]) -> list[TokenRecord]:
    return [
        record
        for record in records
        if record.event_id is not None and record.event in {"alloc", "free"}
    ]


def build_event_windows(records: Sequence[TokenRecord], window_size: int) -> list[EventWindow]:
    windows: list[EventWindow] = []
    grouped = records_by_rank(records)

    for rank in sorted(grouped):
        rank_records = sorted(grouped[rank], key=lambda record: record.line_number)
        if not rank_records:
            continue

        actual_window_size = min(window_size, len(rank_records))
        for start in range(0, len(rank_records) - actual_window_size + 1):
            window_records = rank_records[start : start + actual_window_size]
            windows.append(
                EventWindow(
                    rank=rank,
                    window_index=start,
                    record_count=len(window_records),
                    token_sum=sum(record.received_tokens for record in window_records),
                    start_line=window_records[0].line_number,
                    end_line=window_records[-1].line_number,
                    start_step=window_records[0].step,
                    start_layer=window_records[0].layer,
                    end_step=window_records[-1].step,
                    end_layer=window_records[-1].layer,
                    steps=tuple(record.step for record in window_records),
                    layers=tuple(record.layer for record in window_records),
                    tokens=tuple(record.received_tokens for record in window_records),
                )
            )

    return windows


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("inf")
    return numerator / denominator


def summarize_ranks(
    records: Sequence[TokenRecord],
    windows: Sequence[EventWindow],
    window_size: int,
) -> list[RankSummary]:
    grouped_records = records_by_rank(records)
    grouped_windows: dict[int, list[EventWindow]] = defaultdict(list)
    for window in windows:
        grouped_windows[window.rank].append(window)

    summaries: list[RankSummary] = []
    for rank in sorted(grouped_records):
        rank_records = grouped_records[rank]
        rank_windows = grouped_windows.get(rank, [])
        if not rank_windows:
            continue

        peak = max(rank_windows, key=lambda window: window.token_sum)
        trough = min(rank_windows, key=lambda window: window.token_sum)
        mean_tokens = sum(window.token_sum for window in rank_windows) / len(rank_windows)

        summaries.append(
            RankSummary(
                rank=rank,
                records=len(rank_records),
                steps=len({record.step for record in rank_records}),
                event_windows=len(rank_windows),
                event_window_size=min(window_size, len(rank_records)),
                event_peak_tokens=peak.token_sum,
                event_peak_start_line=peak.start_line,
                event_peak_end_line=peak.end_line,
                event_peak_start_step=peak.start_step,
                event_peak_start_layer=peak.start_layer,
                event_peak_end_step=peak.end_step,
                event_peak_end_layer=peak.end_layer,
                event_peak_layers=peak.layers,
                event_peak_token_values=peak.tokens,
                event_trough_tokens=trough.token_sum,
                event_trough_start_step=trough.start_step,
                event_trough_start_layer=trough.start_layer,
                event_trough_end_step=trough.end_step,
                event_trough_end_layer=trough.end_layer,
                event_trough_layers=trough.layers,
                event_trough_token_values=trough.tokens,
                event_mean_tokens=mean_tokens,
                event_peak_to_mean_ratio=safe_ratio(peak.token_sum, mean_tokens),
                event_peak_to_trough_ratio=safe_ratio(peak.token_sum, trough.token_sum),
            )
        )

    return summaries


def build_live_token_samples(records: Sequence[TokenRecord]) -> list[LiveTokenSample]:
    live_by_rank: dict[int, int] = defaultdict(int)
    samples: list[LiveTokenSample] = []

    for record in sorted(lifecycle_records(records), key=lambda item: item.line_number):
        delta = record.received_tokens if record.event == "alloc" else -record.received_tokens
        live_by_rank[record.rank] += delta
        samples.append(
            LiveTokenSample(
                line_number=record.line_number,
                rank=record.rank,
                event_id=record.event_id if record.event_id is not None else -1,
                step=record.step,
                iteration=record.iteration,
                microbatch=record.microbatch,
                phase=record.phase or "unknown",
                event=record.event or "unknown",
                layer=record.layer,
                delta_tokens=delta,
                live_tokens=live_by_rank[record.rank],
            )
        )

    return samples


def summarize_live_ranks(samples: Sequence[LiveTokenSample]) -> list[LiveRankSummary]:
    grouped: dict[int, list[LiveTokenSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.rank].append(sample)

    summaries: list[LiveRankSummary] = []
    for rank in sorted(grouped):
        rank_samples = grouped[rank]
        peak = max(rank_samples, key=lambda item: item.live_tokens)
        trough = min(rank_samples, key=lambda item: item.live_tokens)
        summaries.append(
            LiveRankSummary(
                rank=rank,
                events=len(rank_samples),
                peak_live_tokens=peak.live_tokens,
                peak_line=peak.line_number,
                peak_event_id=peak.event_id,
                peak_step=peak.step,
                peak_iteration=peak.iteration,
                peak_microbatch=peak.microbatch,
                peak_layer=peak.layer,
                min_live_tokens=trough.live_tokens,
                final_live_tokens=rank_samples[-1].live_tokens,
            )
        )

    return summaries


def lifecycle_issues(records: Sequence[TokenRecord]) -> list[LifecycleIssue]:
    grouped: dict[tuple[int, int], list[TokenRecord]] = defaultdict(list)
    for record in lifecycle_records(records):
        grouped[(record.rank, record.event_id if record.event_id is not None else -1)].append(
            record
        )

    issues: list[LifecycleIssue] = []
    for (rank, event_id), group in sorted(grouped.items()):
        alloc_records = [record for record in group if record.event == "alloc"]
        free_records = [record for record in group if record.event == "free"]
        alloc_tokens = sum(record.received_tokens for record in alloc_records)
        free_tokens = sum(record.received_tokens for record in free_records)
        alloc_count = len(alloc_records)
        free_count = len(free_records)

        if (
            alloc_tokens == free_tokens
            and alloc_count == free_count
            and alloc_count > 0
            and free_count > 0
        ):
            continue

        representative = group[0]
        if alloc_count == 0:
            status = "free_without_alloc"
        elif free_count == 0:
            status = "alloc_without_free"
        elif alloc_tokens != free_tokens:
            status = "token_mismatch"
        else:
            status = "count_mismatch"

        alloc_lines = [record.line_number for record in alloc_records]
        free_lines = [record.line_number for record in free_records]
        issues.append(
            LifecycleIssue(
                rank=rank,
                event_id=event_id,
                step=representative.step,
                iteration=representative.iteration,
                microbatch=representative.microbatch,
                layer=representative.layer,
                alloc_tokens=alloc_tokens,
                free_tokens=free_tokens,
                net_tokens=alloc_tokens - free_tokens,
                alloc_count=alloc_count,
                free_count=free_count,
                first_alloc_line=min(alloc_lines) if alloc_lines else 0,
                last_alloc_line=max(alloc_lines) if alloc_lines else 0,
                first_free_line=min(free_lines) if free_lines else 0,
                last_free_line=max(free_lines) if free_lines else 0,
                status=status,
            )
        )

    return issues


def join_ints(values: Sequence[int]) -> str:
    return " ".join(str(value) for value in values)


def write_token_records(path: Path, records: Sequence[TokenRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "line_number",
                "rank",
                "step",
                "layer",
                "received_tokens",
                "iteration",
                "microbatch",
                "phase",
                "event",
                "event_id",
            ]
        )
        for record in sorted(records, key=lambda item: item.line_number):
            writer.writerow(
                [
                    record.line_number,
                    record.rank,
                    record.step,
                    record.layer,
                    record.received_tokens,
                    record.iteration if record.iteration is not None else "",
                    record.microbatch if record.microbatch is not None else "",
                    record.phase or "",
                    record.event or "",
                    record.event_id if record.event_id is not None else "",
                ]
            )


def write_layer_summaries(path: Path, summaries: Sequence[LayerSummary]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "line_number",
                "step",
                "layer",
                "average_received_tokens",
                "max_received_tokens",
                "imbalance_ratio",
            ]
        )
        for summary in sorted(summaries, key=lambda item: item.line_number):
            writer.writerow(
                [
                    summary.line_number,
                    summary.step,
                    summary.layer,
                    summary.average_received_tokens,
                    summary.max_received_tokens,
                    summary.imbalance_ratio,
                ]
            )


def write_event_windows(path: Path, windows: Sequence[EventWindow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "rank",
                "window_index",
                "record_count",
                "token_sum",
                "start_line",
                "end_line",
                "start_step",
                "start_layer",
                "end_step",
                "end_layer",
                "steps",
                "layers",
                "tokens",
            ]
        )
        for window in sorted(windows, key=lambda item: (item.rank, item.window_index)):
            writer.writerow(
                [
                    window.rank,
                    window.window_index,
                    window.record_count,
                    window.token_sum,
                    window.start_line,
                    window.end_line,
                    window.start_step,
                    window.start_layer,
                    window.end_step,
                    window.end_layer,
                    join_ints(window.steps),
                    join_ints(window.layers),
                    join_ints(window.tokens),
                ]
            )


def write_rank_summaries(path: Path, summaries: Sequence[RankSummary]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "rank",
                "records",
                "steps",
                "event_windows",
                "event_window_size",
                "event_peak_tokens",
                "event_peak_start_line",
                "event_peak_end_line",
                "event_peak_start_step",
                "event_peak_start_layer",
                "event_peak_end_step",
                "event_peak_end_layer",
                "event_peak_layers",
                "event_peak_token_values",
                "event_trough_tokens",
                "event_trough_start_step",
                "event_trough_start_layer",
                "event_trough_end_step",
                "event_trough_end_layer",
                "event_trough_layers",
                "event_trough_token_values",
                "event_mean_tokens",
                "event_peak_to_mean_ratio",
                "event_peak_to_trough_ratio",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.rank,
                    summary.records,
                    summary.steps,
                    summary.event_windows,
                    summary.event_window_size,
                    summary.event_peak_tokens,
                    summary.event_peak_start_line,
                    summary.event_peak_end_line,
                    summary.event_peak_start_step,
                    summary.event_peak_start_layer,
                    summary.event_peak_end_step,
                    summary.event_peak_end_layer,
                    join_ints(summary.event_peak_layers),
                    join_ints(summary.event_peak_token_values),
                    summary.event_trough_tokens,
                    summary.event_trough_start_step,
                    summary.event_trough_start_layer,
                    summary.event_trough_end_step,
                    summary.event_trough_end_layer,
                    join_ints(summary.event_trough_layers),
                    join_ints(summary.event_trough_token_values),
                    round(summary.event_mean_tokens, 2),
                    round(summary.event_peak_to_mean_ratio, 4),
                    round(summary.event_peak_to_trough_ratio, 4),
                ]
            )


def write_live_token_samples(path: Path, samples: Sequence[LiveTokenSample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "line_number",
                "rank",
                "event_id",
                "step",
                "iteration",
                "microbatch",
                "phase",
                "event",
                "layer",
                "delta_tokens",
                "live_tokens",
            ]
        )
        for sample in sorted(samples, key=lambda item: item.line_number):
            writer.writerow(
                [
                    sample.line_number,
                    sample.rank,
                    sample.event_id,
                    sample.step,
                    sample.iteration if sample.iteration is not None else "",
                    sample.microbatch if sample.microbatch is not None else "",
                    sample.phase,
                    sample.event,
                    sample.layer,
                    sample.delta_tokens,
                    sample.live_tokens,
                ]
            )


def write_live_rank_summaries(path: Path, summaries: Sequence[LiveRankSummary]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "rank",
                "events",
                "peak_live_tokens",
                "peak_line",
                "peak_event_id",
                "peak_step",
                "peak_iteration",
                "peak_microbatch",
                "peak_layer",
                "min_live_tokens",
                "final_live_tokens",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.rank,
                    summary.events,
                    summary.peak_live_tokens,
                    summary.peak_line,
                    summary.peak_event_id,
                    summary.peak_step,
                    summary.peak_iteration if summary.peak_iteration is not None else "",
                    summary.peak_microbatch if summary.peak_microbatch is not None else "",
                    summary.peak_layer,
                    summary.min_live_tokens,
                    summary.final_live_tokens,
                ]
            )


def write_lifecycle_issues(path: Path, issues: Sequence[LifecycleIssue]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "rank",
                "event_id",
                "step",
                "iteration",
                "microbatch",
                "layer",
                "alloc_tokens",
                "free_tokens",
                "net_tokens",
                "alloc_count",
                "free_count",
                "first_alloc_line",
                "last_alloc_line",
                "first_free_line",
                "last_free_line",
                "status",
            ]
        )
        for issue in issues:
            writer.writerow(
                [
                    issue.rank,
                    issue.event_id,
                    issue.step,
                    issue.iteration if issue.iteration is not None else "",
                    issue.microbatch if issue.microbatch is not None else "",
                    issue.layer,
                    issue.alloc_tokens,
                    issue.free_tokens,
                    issue.net_tokens,
                    issue.alloc_count,
                    issue.free_count,
                    issue.first_alloc_line,
                    issue.last_alloc_line,
                    issue.first_free_line,
                    issue.last_free_line,
                    issue.status,
                ]
            )


def write_split_rank_series(output_dir: Path, records: Sequence[TokenRecord]) -> None:
    rank_dir = output_dir / "rank_series"
    rank_dir.mkdir(parents=True, exist_ok=True)

    for rank, rank_records in sorted(records_by_rank(records).items()):
        path = rank_dir / f"rank_{rank:03d}.csv"
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "event_index",
                    "line_number",
                    "step",
                    "layer",
                    "received_tokens",
                    "iteration",
                    "microbatch",
                    "phase",
                    "event",
                    "event_id",
                ]
            )
            for event_index, record in enumerate(
                sorted(rank_records, key=lambda item: item.line_number)
            ):
                writer.writerow(
                    [
                        event_index,
                        record.line_number,
                        record.step,
                        record.layer,
                        record.received_tokens,
                        record.iteration if record.iteration is not None else "",
                        record.microbatch if record.microbatch is not None else "",
                        record.phase or "",
                        record.event or "",
                        record.event_id if record.event_id is not None else "",
                    ]
                )


def write_text_summary(
    path: Path,
    log_path: Path,
    parse_result: ParseResult,
    summaries: Sequence[RankSummary],
    live_summaries: Sequence[LiveRankSummary],
    lifecycle_unmatched: Sequence[LifecycleIssue],
    output_dir: Path,
    window_size: int,
    top: int,
) -> None:
    top_ranks = sorted(summaries, key=lambda item: item.event_peak_tokens, reverse=True)[:top]
    top_live_ranks = sorted(
        live_summaries, key=lambda item: item.peak_live_tokens, reverse=True
    )[:top]

    lines = [
        "MoE token pressure report",
        "",
        f"input_log: {log_path}",
        f"output_dir: {output_dir}",
        f"window_size: {window_size}",
        f"token_records: {len(parse_result.token_records)}",
        f"lifecycle_ranks: {len(live_summaries)}",
        f"unmatched_lifecycle_events: {len(lifecycle_unmatched)}",
        "unmatched_lifecycle_net_tokens: {}".format(
            sum(issue.net_tokens for issue in lifecycle_unmatched)
        ),
        f"layer_summaries: {len(parse_result.layer_summaries)}",
        f"skipped_metric_lines: {parse_result.skipped_metric_lines}",
        "",
        "files:",
        "  rank_tokens.csv",
        "  rank_event_windows.csv",
        "  rank_summary.csv",
        "  rank_live_tokens.csv",
        "  rank_live_summary.csv",
        "  rank_lifecycle_unmatched.csv",
        "  layer_imbalance_summary.csv",
        "  rank_series/rank_XXX.csv",
        "",
    ]

    if top_live_ranks:
        lines.extend(
            [
                "top ranks by exact live-token peak:",
                "rank,peak_live_tokens,peak_event_id,peak_step,peak_microbatch,peak_layer,min_live_tokens,final_live_tokens",
            ]
        )
        for summary in top_live_ranks:
            lines.append(
                ",".join(
                    [
                        str(summary.rank),
                        str(summary.peak_live_tokens),
                        str(summary.peak_event_id),
                        str(summary.peak_step),
                        "" if summary.peak_microbatch is None else str(summary.peak_microbatch),
                        str(summary.peak_layer),
                        str(summary.min_live_tokens),
                        str(summary.final_live_tokens),
                    ]
                )
            )
        lines.append("")

    if lifecycle_unmatched:
        by_status: dict[str, int] = defaultdict(int)
        for issue in lifecycle_unmatched:
            by_status[issue.status] += 1
        lines.extend(
            [
                "lifecycle unmatched status counts:",
                ",".join(f"{status}={count}" for status, count in sorted(by_status.items())),
                "",
                "top unmatched lifecycle entries by abs(net_tokens):",
                "rank,event_id,step,microbatch,layer,net_tokens,status",
            ]
        )
        for issue in sorted(
            lifecycle_unmatched, key=lambda item: abs(item.net_tokens), reverse=True
        )[:top]:
            lines.append(
                ",".join(
                    [
                        str(issue.rank),
                        str(issue.event_id),
                        str(issue.step),
                        "" if issue.microbatch is None else str(issue.microbatch),
                        str(issue.layer),
                        str(issue.net_tokens),
                        issue.status,
                    ]
                )
            )
        lines.append("")

    lines.extend(
        [
            "top ranks by event-window peak:",
            "rank,peak_tokens,layers,tokens,mean_tokens,peak_to_mean",
        ]
    )

    for summary in top_ranks:
        lines.append(
            ",".join(
                [
                    str(summary.rank),
                    str(summary.event_peak_tokens),
                    '"' + join_ints(summary.event_peak_layers) + '"',
                    '"' + join_ints(summary.event_peak_token_values) + '"',
                    str(round(summary.event_mean_tokens, 2)),
                    str(round(summary.event_peak_to_mean_ratio, 4)),
                ]
            )
        )

    lines.extend(
        [
            "",
            "interpretation:",
            "  rank_live_tokens.csv reconstructs exact live-token deltas when the log",
            "  includes moe_token_event alloc/free lifecycle events.",
            "  rank_lifecycle_unmatched.csv lists alloc/free pairs that do not reconcile.",
            "  alloc_without_free at the end of a log usually means the run stopped with",
            "  in-flight activations still live or the log was truncated.",
            "  min_live_tokens below zero means the log line order is not causal enough",
            "  for exact reconstruction and should be investigated.",
            "  rank_event_windows.csv rolls over each rank's observed token events in log order.",
            f"  A window size of {window_size} estimates how many received tokens are live if",
            f"  {window_size} adjacent",
            "  dispatch events can overlap in the pipeline.",
            "  If lifecycle_ranks is 0, this report only has the rolling pressure estimate.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(
    output_dir: Path,
    log_path: Path,
    parse_result: ParseResult,
    window_size: int,
    split_ranks: bool,
    top: int,
) -> tuple[list[RankSummary], list[LiveRankSummary]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    records_for_pressure = pressure_records(parse_result.token_records)
    windows = build_event_windows(records_for_pressure, window_size)
    summaries = summarize_ranks(records_for_pressure, windows, window_size)
    live_samples = build_live_token_samples(parse_result.token_records)
    live_summaries = summarize_live_ranks(live_samples)
    lifecycle_unmatched = lifecycle_issues(parse_result.token_records)

    write_token_records(output_dir / "rank_tokens.csv", parse_result.token_records)
    write_layer_summaries(output_dir / "layer_imbalance_summary.csv", parse_result.layer_summaries)
    write_event_windows(output_dir / "rank_event_windows.csv", windows)
    write_rank_summaries(output_dir / "rank_summary.csv", summaries)
    write_live_token_samples(output_dir / "rank_live_tokens.csv", live_samples)
    write_live_rank_summaries(output_dir / "rank_live_summary.csv", live_summaries)
    write_lifecycle_issues(output_dir / "rank_lifecycle_unmatched.csv", lifecycle_unmatched)
    if split_ranks:
        write_split_rank_series(output_dir, parse_result.token_records)
    write_text_summary(
        output_dir / "summary.txt",
        log_path,
        parse_result,
        summaries,
        live_summaries,
        lifecycle_unmatched,
        output_dir,
        window_size,
        top,
    )

    return summaries, live_summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MoE received-token logs and reconstruct or estimate per-rank token pressure."
        )
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"log file to parse, default: {DEFAULT_LOG_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for CSV reports, default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--window-size",
        type=positive_int,
        default=8,
        help="number of adjacent token events to sum for each rank",
    )
    parser.add_argument(
        "--top",
        type=positive_int,
        default=12,
        help="number of high-peak ranks to include in summary.txt and stdout",
    )
    parser.add_argument(
        "--no-split-ranks",
        action="store_true",
        help="skip rank_series/rank_XXX.csv files",
    )
    return parser


def print_stdout_summary(
    summaries: Sequence[RankSummary],
    live_summaries: Sequence[LiveRankSummary],
    output_dir: Path,
    top: int,
) -> None:
    print(f"wrote reports to {output_dir}")
    if live_summaries:
        print("top ranks by exact live-token peak:")
        print("rank  peak_live_tokens  event_id  step  microbatch  min_live  final_live")
        for summary in sorted(
            live_summaries, key=lambda item: item.peak_live_tokens, reverse=True
        )[:top]:
            microbatch = (
                "unknown"
                if summary.peak_microbatch is None
                else str(summary.peak_microbatch)
            )
            print(
                f"{summary.rank:>4}  "
                f"{summary.peak_live_tokens:>16}  "
                f"{summary.peak_event_id:>8}  "
                f"{summary.peak_step:>4}  "
                f"{microbatch:>10}  "
                f"{summary.min_live_tokens:>8}  "
                f"{summary.final_live_tokens:>10}"
            )
    print("top ranks by event-window peak:")
    print("rank  peak_tokens  peak_layers        mean_tokens  peak/mean")
    for summary in sorted(summaries, key=lambda item: item.event_peak_tokens, reverse=True)[:top]:
        peak_layers = join_ints(summary.event_peak_layers)
        print(
            f"{summary.rank:>4}  "
            f"{summary.event_peak_tokens:>11}  "
            f"{peak_layers:<17}  "
            f"{summary.event_mean_tokens:>11.2f}  "
            f"{summary.event_peak_to_mean_ratio:>9.4f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = resolve_from_repo(args.log_path)
    output_dir = resolve_from_repo(args.output_dir)

    if not log_path.exists():
        raise SystemExit(f"log file does not exist: {log_path}")

    parse_result = parse_moe_log(log_path)
    if not parse_result.token_records:
        raise SystemExit(f"no MoE token records found in: {log_path}")

    summaries, live_summaries = write_reports(
        output_dir=output_dir,
        log_path=log_path,
        parse_result=parse_result,
        window_size=args.window_size,
        split_ranks=not args.no_split_ranks,
        top=args.top,
    )
    print_stdout_summary(summaries, live_summaries, output_dir, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
