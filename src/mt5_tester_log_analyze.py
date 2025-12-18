"""Analyze MetaTrader 5 Strategy Tester log files for tick-data quality.

This helps you quantify how much of a backtest used synthetic tick generation,
by parsing lines like:
  - "XAUUSD : YYYY.MM.DD 23:59 - no real ticks within a day"
  - "real ticks absent for ... whole days"

Usage (PowerShell):
  python src\mt5_tester_log_analyze.py "C:\\...\\20251213.log"

Tip:
  MT5 prints the log path at the end of a test, e.g.:
    log file "C:\\Users\\...\\logs\\20251213.log" written
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional


NO_REAL_TICKS_RE = re.compile(
    r"^\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\s+Core \d+\s+"
    r"(?P<symbol>[^:]+)\s+:\s+(?P<day>\d{4}\.\d{2}\.\d{2})\s+23:59\s+-\s+no real ticks within a day\s*$"
)

SUMMARY_RE = re.compile(
    r"real ticks absent for\s+(?P<minutes>\d+)\s+minutes of\s+(?P<total>\d+)\s+total minute bars",
    re.IGNORECASE,
)

SUMMARY_DAYS_RE = re.compile(
    r"real ticks absent for\s+(?P<days>\d+)\s+whole days",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LogStats:
    symbols: Counter[str]
    days_no_real_ticks: list[date]
    absent_minutes: Optional[int]
    total_minutes: Optional[int]
    absent_days_summary: Optional[int]


def _parse_day(d: str) -> date:
    y, m, dd = d.split(".")
    return date(int(y), int(m), int(dd))


def iter_lines(path: Path) -> Iterable[str]:
    # MT5 logs are usually ANSI/UTF-8-ish. Try utf-8 first, then fall back.
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            yield from f
    except OSError:
        # Re-raise for a clear error.
        raise


def analyze_log(path: Path) -> LogStats:
    symbols = Counter()
    days: list[date] = []
    absent_minutes: Optional[int] = None
    total_minutes: Optional[int] = None
    absent_days_summary: Optional[int] = None

    for line in iter_lines(path):
        line = line.rstrip("\n")

        m = NO_REAL_TICKS_RE.match(line)
        if m:
            symbols[m.group("symbol").strip()] += 1
            days.append(_parse_day(m.group("day")))
            continue

        if absent_minutes is None:
            sm = SUMMARY_RE.search(line)
            if sm:
                absent_minutes = int(sm.group("minutes"))
                total_minutes = int(sm.group("total"))

        if absent_days_summary is None:
            sd = SUMMARY_DAYS_RE.search(line)
            if sd:
                absent_days_summary = int(sd.group("days"))

    days.sort()
    return LogStats(
        symbols=symbols,
        days_no_real_ticks=days,
        absent_minutes=absent_minutes,
        total_minutes=total_minutes,
        absent_days_summary=absent_days_summary,
    )


def _fmt_pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{(100.0 * n / d):.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze MT5 Strategy Tester log tick-data quality.")
    ap.add_argument("log", type=Path, help="Path to MT5 tester log .log file")
    ap.add_argument(
        "--top-days",
        type=int,
        default=10,
        help="Print the first/last N 'no real ticks' days (default: 10)",
    )
    args = ap.parse_args()

    path: Path = args.log
    if not path.exists():
        raise SystemExit(f"Log not found: {path}")

    stats = analyze_log(path)

    print(f"Log: {path}")

    if stats.absent_minutes is not None and stats.total_minutes is not None:
        present = stats.total_minutes - stats.absent_minutes
        print(
            "Tick coverage summary: "
            f"absent={stats.absent_minutes} / total={stats.total_minutes} "
            f"({_fmt_pct(stats.absent_minutes, stats.total_minutes)} absent, "
            f"{_fmt_pct(present, stats.total_minutes)} present)"
        )

    if stats.absent_days_summary is not None:
        print(f"Whole days with no real ticks (summary line): {stats.absent_days_summary}")

    if stats.symbols:
        print("Symbols with 'no real ticks within a day' lines:")
        for sym, cnt in stats.symbols.most_common():
            print(f"  - {sym}: {cnt} day(s)")

    if stats.days_no_real_ticks:
        print(f"Distinct 'no real ticks' days found: {len(set(stats.days_no_real_ticks))}")
        uniq = sorted(set(stats.days_no_real_ticks))
        n = max(0, int(args.top_days))
        if n > 0:
            head = uniq[:n]
            tail = uniq[-n:] if len(uniq) > n else []
            print("First days:")
            for d in head:
                print(f"  - {d.isoformat()}")
            if tail:
                print("Last days:")
                for d in tail:
                    print(f"  - {d.isoformat()}")

    # Quick heuristic suggestion for a "clean" window:
    # If we have the summary minutes, we can at least tell if this test is mostly synthetic.
    if stats.absent_minutes is not None and stats.total_minutes is not None:
        absent_ratio = stats.absent_minutes / max(1, stats.total_minutes)
        if absent_ratio > 0.50:
            print(
                "Suggestion: This period is heavily synthetic (>50% minutes missing real ticks). "
                "Consider benchmarking on a shorter window where real ticks are mostly present."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
